"""Explicit collective primitives; no model discovery or Trainer dependency."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Function
from torch.distributed.tensor import DTensor

from .ownership import ParameterMapping


class CopyToParallel(Function):
    """Replicated input to column shards: sum input gradients on backward."""

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        gradient = gradient.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


class ReduceFromParallel(Function):
    """Sum partial outputs; identical downstream losses need no backward sum."""

    @staticmethod
    def forward(ctx, value, group):
        result = value.contiguous().clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class AllToAll(Function):
    """Variable token exchange whose backward reverses the same split map."""

    @staticmethod
    def forward(ctx, value, send, receive, group):
        ctx.send, ctx.receive, ctx.group = send, receive, group
        output = value.new_empty((sum(receive), *value.shape[1:]))
        dist.all_to_all_single(output, value.contiguous(), receive, send, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        output = gradient.new_empty((sum(ctx.send), *gradient.shape[1:]))
        dist.all_to_all_single(
            output, gradient.contiguous(), ctx.send, ctx.receive, group=ctx.group
        )
        return output, None, None, None


class ParallelLinear(nn.Module):
    """Local linear storage with explicit replicated input/output semantics."""

    def __init__(self, source: nn.Linear, group: dist.ProcessGroup, column: bool):
        super().__init__()
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        axis = 0 if column else 1
        self.weight = nn.Parameter(
            source.weight.chunk(size, axis)[rank].clone(),
            requires_grad=source.weight.requires_grad,
        )
        self.bias = None
        if source.bias is not None:
            self.bias = nn.Parameter(
                source.bias.chunk(size)[rank].clone()
                if column
                else source.bias.clone(),
                requires_grad=source.bias.requires_grad,
            )
        self.group, self.column = group, column

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.column:
            value = CopyToParallel.apply(value, self.group)
            return torch.nn.functional.linear(value, self.weight, self.bias)
        output = ReduceFromParallel.apply(
            torch.nn.functional.linear(value, self.weight), self.group
        )
        return output if self.bias is None else output + self.bias


class ExpertDispatcher(nn.Module):
    """Route assignments to expert owners, execute, reverse, then apply scores.

    Experts own gate/up/down mathematics. Dispatch only understands expert IDs,
    ownership and the exchange permutation. Score gradients stay on the source.
    """

    def __init__(self, experts: nn.Module, group: dist.ProcessGroup):
        super().__init__()
        # Keep expert parameters at their original FQNs in the model tree.
        object.__setattr__(self, "experts", experts)
        self.group = group
        self.size, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.local_experts = experts.num_experts // self.size
        start = self.rank * self.local_experts
        for name in ("gate_up_proj", "down_proj"):
            value = getattr(experts, name)
            setattr(
                experts,
                name,
                nn.Parameter(
                    value[start : start + self.local_experts].clone(),
                    requires_grad=value.requires_grad,
                ),
            )
        experts.num_experts = self.local_experts
        self.exchanges = 0
        self.empty_batches = 0

    def forward(self, hidden, indices, scores):
        flat_ids = indices.flatten()
        permutation = flat_ids.argsort(stable=True)
        owners = flat_ids // self.local_experts
        counts = torch.bincount(owners, minlength=self.size)
        received_counts = torch.empty_like(counts)
        dist.all_to_all_single(received_counts, counts, group=self.group)
        send, receive = counts.tolist(), received_counts.tolist()
        self.empty_batches += int(sum(receive) == 0)
        token_ids = permutation // indices.shape[-1]
        routed = AllToAll.apply(hidden[token_ids], send, receive, self.group)
        sorted_ids = flat_ids[permutation].contiguous()
        received_ids = sorted_ids.new_empty(sum(receive))
        dist.all_to_all_single(
            received_ids, sorted_ids, receive, send, group=self.group
        )
        local_ids = (received_ids - self.rank * self.local_experts).unsqueeze(-1)
        result = self.experts(routed, local_ids, routed.new_ones((len(routed), 1)))
        returned = AllToAll.apply(result, receive, send, self.group)
        restored = torch.zeros_like(returned).index_copy(0, permutation, returned)
        self.exchanges += 2
        return (
            restored.reshape(*indices.shape, hidden.shape[-1]) * scores[..., None]
        ).sum(1)


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Read local FSDP storage without private DTensor attributes."""
    return value.to_local() if isinstance(value, DTensor) else value


@dataclass
class ParallelRuntime:
    """Runtime groups and parameter layouts, fixed before optimizer creation."""

    tp_group: dist.ProcessGroup | None = None
    ep_group: dist.ProcessGroup | None = None
    data_group: dist.ProcessGroup | None = None
    norm_group: dist.ProcessGroup | None = None
    tp_size: int = 1
    ep_size: int = 1
    fsdp_size: int = 1
    data_rank: int = 0
    parameter_mappings: tuple[ParameterMapping, ...] = ()

    @property
    def data_size(self) -> int:
        return self.ep_size * self.fsdp_size

    def sum_data(self, value: torch.Tensor) -> torch.Tensor:
        result = value.detach().clone()
        if self.data_size > 1:
            dist.all_reduce(result, group=self.data_group)
        return result

    def synchronize_and_clip(self, model: nn.Module, maximum: float = 1.0) -> float:
        """Normalize EP ownership, sum partial TP norms, then clip exactly once."""
        parameters = list(model.named_parameters())
        ownership = {entry.canonical_name: entry for entry in self.parameter_mappings}
        if self.data_size * self.tp_size > 1 and set(ownership) != {
            name for name, _ in parameters
        }:
            raise ValueError(
                "Distributed gradients require complete parameter ownership"
            )
        device = next(model.parameters()).device
        squared = torch.zeros((), device=device)
        for name, parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = local_tensor(parameter.grad)
            mapping = ownership.get(name)
            if mapping is not None and mapping.target is not parameter:
                raise ValueError(f"Stale runtime parameter mapping: {name}")
            partial = mapping.source.tp_gradient_sum if mapping else False
            if partial and self.tp_size > 1:
                dist.all_reduce(gradient, group=self.tp_group)
            if self.ep_size > 1:
                if mapping.ep_replicas == self.ep_size:
                    dist.all_reduce(gradient, group=self.ep_group)
                gradient.div_(self.ep_size)
            # FSDP owns disjoint shards. Other axes can replicate this tensor.
            replicas = mapping.tp_replicas * mapping.ep_replicas if mapping else 1
            squared += gradient.float().square().sum() / replicas
        if self.tp_size * self.ep_size * self.fsdp_size > 1:
            dist.all_reduce(squared, group=self.norm_group)
        norm = squared.sqrt()
        scale = (maximum / (norm + 1e-6)).clamp(max=1)
        for _, parameter in parameters:
            if parameter.grad is not None:
                local_tensor(parameter.grad).mul_(scale)
        return float(norm)
