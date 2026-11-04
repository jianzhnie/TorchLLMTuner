"""Diagnostic vocab-parallel output arithmetic with unchanged parameter storage."""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import loss_parallel

from llmtuner.parallel.tensor_parallel.linear_ops import all_gather_along


def install_output_layout_control(trainer):
    """Install only for explicit diagnostic replay, never normal training."""
    dims = trainer.parallel_dims
    if dims.tp <= 1 or dims.cp != 1 or dims.pp != 1:
        raise ValueError("Output layout control requires TP and no CP/PP")
    head = trainer.model_parts[0].lm_head
    if head.bias is not None or getattr(head, "compute_dtype", None) is not None:
        raise ValueError("Output control requires an uncast, bias-free linear head")
    mesh = dims.get_mesh("tp")
    group = mesh.get_group()
    size, rank = mesh.size(), mesh.get_local_rank()
    if head.weight.shape[0] % size:
        raise ValueError("Output control requires evenly divisible vocabulary")

    def forward(module, inputs):
        weight = module.weight
        if isinstance(weight, DTensor):
            # FSDP's pre-forward hook has already materialized this DP shard.
            # The TP-replicated storage remains unchanged; slicing contributes
            # zeros for the other vocabulary slices, later summed by Trainer's
            # existing replicated-TP gradient reduction.
            weight = weight.to_local()
        gathered = all_gather_along(inputs, 0, group)
        return F.linear(gathered, weight.chunk(size, dim=0)[rank])

    def loss_sum(logits, labels, **kwargs):
        pieces = [torch.empty_like(labels) for _ in range(size)]
        dist.all_gather(pieces, labels.contiguous(), group=group)
        labels = torch.cat(pieces)
        value = DTensor.from_local(logits, mesh, [Shard(-1)], run_check=False)
        total = F.cross_entropy(value.float(), labels, reduction="sum")
        total = total.to_local()
        # The loss mesh includes TP because ordinary replay partitions tokens.
        # Report each replica's share, while retaining the full loss derivative
        # for the disjoint vocabulary weights and partial input gradients.
        return total + total.detach() * (1.0 / size - 1.0)

    head.forward = MethodType(forward, head)
    trainer.loss_sum = loss_sum
    original_context = trainer.param_context

    @contextmanager
    def output_context():
        # PyTorch's loss-parallel handlers must remain installed for backward,
        # not merely for the cross-entropy forward.
        with original_context(), loss_parallel():
            yield

    trainer.param_context = output_context


def local_token_logits(logits, mesh):
    """Reconstruct only for diagnostic snapshots; never feeds backward."""
    pieces = [torch.empty_like(logits) for _ in range(mesh.size())]
    dist.all_gather(pieces, logits.detach().contiguous(), group=mesh.get_group())
    return torch.cat(pieces, dim=-1).chunk(mesh.size(), dim=0)[mesh.get_local_rank()]
