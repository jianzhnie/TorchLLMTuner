"""Resolve EP routing and packed-storage bindings from model declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from ...errors import UnsupportedCombinationError
from ...models.contracts import MoEBlockContract


def _require_moe_contract(block: nn.Module, *, requirement: str) -> MoEBlockContract:
    contract = getattr(block, "moe_contract", None)
    if contract is None:
        raise UnsupportedCombinationError(
            f"{requirement} a model-owned MoEBlockContract"
        )
    if not isinstance(contract, MoEBlockContract):
        raise TypeError("moe_contract must be a MoEBlockContract")
    return contract


def _validate_expert_bias(owner: nn.Module) -> None:
    if any(
        name == "bias" or name.endswith("_bias")
        for name, _ in owner.named_parameters(recurse=False)
    ):
        raise NotImplementedError(
            "EP grouped experts do not support expert bias vectors"
        )


@dataclass(frozen=True, slots=True)
class MoERoutingBinding:
    """Model routing choices resolved before allocating replacement weights."""

    source: nn.Module
    router: nn.Module
    shared: nn.Module | None
    top_k: int
    score_func: str
    route_norm: bool
    route_scale: float
    num_expert_groups: int | None
    num_limited_groups: int | None
    shared_gate: nn.Module | None = None

    def validate(self, block: nn.Module) -> None:
        """Reject changed routing modules or declarations before conversion."""
        if block is not self.source:
            raise ValueError("MoE routing binding belongs to another block")
        contract = getattr(block, "moe_contract", None)
        if not isinstance(contract, MoEBlockContract):
            raise ValueError("Stale EP routing contract")
        router, _, shared = contract.bind(block)
        shared_gate = (
            block._modules[contract.shared_expert_gate]
            if contract.shared_expert_gate is not None
            else None
        )
        if (
            contract.group_routing == "single_best"
            or router is not self.router
            or shared is not self.shared
            or shared_gate is not self.shared_gate
            or any(
                getattr(contract, name) != getattr(self, name)
                for name in (
                    "top_k",
                    "score_func",
                    "route_norm",
                    "route_scale",
                    "num_expert_groups",
                    "num_limited_groups",
                )
            )
        ):
            raise ValueError("Stale EP routing binding")


def resolve_moe_routing(block: nn.Module) -> MoERoutingBinding:
    """Bind routing from the model-owned contract without inspecting forward."""
    contract = _require_moe_contract(block, requirement="MoE conversion requires")
    if contract.group_routing == "single_best":
        raise NotImplementedError(
            "EP group-limited greedy selects a single best expert per group; "
            "the grouped dispatcher implements top-2 score summation"
        )
    router, _, shared = contract.bind(block)
    return MoERoutingBinding(
        source=block,
        router=router,
        shared=shared,
        shared_gate=(
            block._modules[contract.shared_expert_gate]
            if contract.shared_expert_gate is not None
            else None
        ),
        top_k=contract.top_k,
        score_func=contract.score_func,
        route_norm=contract.route_norm,
        route_scale=contract.route_scale,
        num_expert_groups=contract.num_expert_groups,
        num_limited_groups=contract.num_limited_groups,
    )


class PackedExpertViews(NamedTuple):
    """Views of an explicitly declared fused gate/up expert storage."""

    gate_EFD: torch.Tensor
    up_EFD: torch.Tensor
    down_EDF: torch.Tensor
    num_experts: int


@dataclass(frozen=True, slots=True)
class PackedExpertBinding:
    """Validated gate/up and down parameters; no copied or stacked storage.

    Resolve after TP, and bind only to the concrete source block. Views are
    created when conversion executes, so current parameter values are copied.
    """

    source: nn.Module
    owner: nn.Module
    gate_up: nn.Parameter
    down: nn.Parameter
    shape: tuple[int, int, int]

    def validate(self, block: nn.Module) -> None:
        """Reject replaced parameters or geometry before allocating conversion state."""
        contract = getattr(block, "moe_contract", None)
        if (
            block is not self.source
            or not isinstance(contract, MoEBlockContract)
            or contract.activation != "silu"
            or contract.num_experts != self.shape[0]
            or block._modules.get("experts") is not self.owner
            or self.owner._parameters.get("gate_up_proj") is not self.gate_up
            or self.owner._parameters.get("down_proj") is not self.down
            or tuple(self.gate_up.shape) != self.shape
            or tuple(self.down.shape)
            != (self.shape[0], self.shape[2], self.shape[1] // 2)
        ):
            raise ValueError("Stale EP packed expert binding")
        _validate_expert_bias(self.owner)
        router, _, _ = contract.bind(block)
        weight = router._parameters.get("weight")
        if (
            not isinstance(weight, nn.Parameter)
            or tuple(weight.shape) != (self.shape[0], self.shape[2])
            or weight.device != self.gate_up.device
            or self.down.device != self.gate_up.device
        ):
            raise ValueError("Stale EP packed expert router geometry or device")

    def projections(self, block: nn.Module) -> PackedExpertViews:
        """Create only role-specific views of the validated live parameters."""
        self.validate(block)
        hidden = self.shape[1] // 2
        return PackedExpertViews(
            self.gate_up[:, :hidden],
            self.gate_up[:, hidden:],
            self.down,
            self.shape[0],
        )


def resolve_packed_experts(block: nn.Module, *, degree: int) -> PackedExpertBinding:
    """Bind fused parameter storage described by the model contract."""
    if type(degree) is not int or degree < 1:
        raise ValueError("EP degree must be a positive integer")
    contract = _require_moe_contract(block, requirement="EP packed experts require")
    owner = block._modules.get("experts")
    if owner is None:
        raise ValueError("Declared fused experts require an experts child module")
    gate_up = owner._parameters.get("gate_up_proj")
    down = owner._parameters.get("down_proj")
    if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
        raise ValueError(
            "Declared fused experts require gate_up_proj and down_proj parameters"
        )
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("EP fused gate/up requires rank-3 expert parameters")
    _validate_expert_bias(owner)
    if contract.activation != "silu":
        raise UnsupportedCombinationError(
            f"EP grouped experts require SiLU, but the model declares "
            f"{contract.activation!r}; retain the native expert path"
        )
    num_experts = contract.num_experts
    router, _, _ = contract.bind(block)
    weight = router._parameters.get("weight")
    if not isinstance(weight, nn.Parameter) or tuple(weight.shape) != (
        num_experts,
        gate_up.shape[2],
    ):
        raise ValueError(
            "Declared MoE router weight must have shape "
            "[num_experts, hidden_size] matching packed expert storage"
        )
    if len({weight.device, gate_up.device, down.device}) != 1:
        raise UnsupportedCombinationError(
            "Declared MoE router and experts are on different devices"
        )
    if gate_up.shape[0] != num_experts or down.shape[0] != num_experts:
        raise ValueError("Declared MoE expert count differs from packed storage")
    if gate_up.shape[1] % 2 != 0 or tuple(down.shape[1:]) != (
        gate_up.shape[2],
        gate_up.shape[1] // 2,
    ):
        raise ValueError("Declared fused gate/up expert geometry is invalid")
    if num_experts % degree:
        raise ValueError(
            f"EP degree {degree} does not divide num_experts={num_experts}"
        )
    if gate_up.shape[1] == 0 or gate_up.shape[2] == 0:
        raise ValueError("EP expert dimensions must be positive")
    return PackedExpertBinding(block, owner, gate_up, down, tuple(gate_up.shape))


__all__ = [
    "MoERoutingBinding",
    "PackedExpertBinding",
    "PackedExpertViews",
    "resolve_moe_routing",
    "resolve_packed_experts",
]
