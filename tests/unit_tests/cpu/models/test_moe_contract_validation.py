"""Invalid model declarations fail before parallel execution can coerce them."""

import pytest
import torch
from torch import nn

from llmtuner.models.contracts import MoEBlockContract
from llmtuner.parallel.expert_parallel.block_conversion import resolve_moe_routing


@pytest.mark.parametrize(
    "overrides",
    [
        {"router": 42},
        {"shared_expert": 42},
        {"num_expert_groups": 2},
        {"num_limited_groups": 1},
        {"num_expert_groups": True, "num_limited_groups": 1},
        {"num_expert_groups": 3, "num_limited_groups": 1},
        {"num_expert_groups": 2, "num_limited_groups": 3},
        {"num_expert_groups": 4, "num_limited_groups": 1},
        {"num_experts": True},
        {"num_experts": 4.5},
        {"top_k": True},
        {"top_k": 1.5},
        {"top_k": 0},
        {"top_k": 5},
        {"route_norm": "false"},
        {"router_logits_in_output": "false"},
        {"router_logits_in_output": 0},
        {"router_logits_in_output": None},
        {"route_scale": True},
        {"route_scale": 0},
        {"route_scale": -1},
        {"route_scale": float("nan")},
        {"route_scale": float("inf")},
        {"activation": ""},
        {"activation": " "},
        {"activation": None},
        {"activation": True},
        {"score_func": []},
        {"score_func": {}},
        {"score_func": None},
        {"score_func": True},
        {"score_func": "unknown"},
    ],
)
def test_moe_contract_rejects_invalid_routing_math(overrides):
    values = dict(router="gate", num_experts=4, top_k=2, route_norm=True)
    with pytest.raises(ValueError):
        MoEBlockContract(**(values | overrides))


@pytest.mark.parametrize("scale", [1, 0.5, 2.5])
def test_moe_contract_accepts_positive_finite_route_scale(scale):
    contract = MoEBlockContract(
        router="gate", num_experts=4, top_k=2, route_norm=False, route_scale=scale
    )
    assert contract.route_scale == scale


def test_grouped_contract_preserves_valid_geometry():
    contract = MoEBlockContract(
        router="gate",
        num_experts=8,
        top_k=4,
        route_norm=True,
        num_expert_groups=4,
        num_limited_groups=2,
    )
    assert contract.num_expert_groups == 4
    assert contract.num_limited_groups == 2


@pytest.mark.parametrize("kind", ["missing", "none", "tensor", "unregistered"])
def test_declared_shared_expert_requires_registered_module(kind):
    block = nn.Module()
    block.gate = nn.Linear(4, 4)
    if kind == "none":
        block.add_module("shared", None)
    elif kind == "tensor":
        block.shared = torch.ones(4)
    elif kind == "unregistered":
        object.__setattr__(block, "shared", nn.Linear(4, 4))
    contract = MoEBlockContract(
        router="gate", num_experts=4, top_k=2, route_norm=True, shared_expert="shared"
    )
    with pytest.raises(ValueError, match="shared expert.*must be a child module"):
        contract.bind(block)


def test_shared_expert_binding_preserves_module_identity():
    block = nn.Module()
    block.gate = nn.Linear(4, 4)
    block.shared = nn.Linear(4, 4)
    contract = MoEBlockContract(
        router="gate", num_experts=4, top_k=2, route_norm=True, shared_expert="shared"
    )
    router, name, shared = contract.bind(block)
    assert router is block.gate and name == "shared" and shared is block.shared


@pytest.mark.parametrize("gate", [42, "", "nested.gate"])
def test_shared_gate_requires_direct_child_name(gate):
    with pytest.raises(ValueError, match="shared expert gate"):
        MoEBlockContract(
            router="gate",
            num_experts=4,
            top_k=2,
            route_norm=True,
            shared_expert="shared",
            shared_expert_gate=gate,
        )


def test_shared_gate_requires_declared_shared_expert():
    with pytest.raises(ValueError, match="shared expert gate"):
        MoEBlockContract(
            router="gate",
            num_experts=4,
            top_k=2,
            route_norm=True,
            shared_expert_gate="shared_gate",
        )


def test_shared_gate_binding_uses_declared_registered_name():
    block = nn.Module()
    block.gate = nn.Linear(4, 4)
    block.shared = nn.Linear(4, 4)
    contract = MoEBlockContract(
        router="gate",
        num_experts=4,
        top_k=2,
        route_norm=True,
        shared_expert="shared",
        shared_expert_gate="blend",
    )
    with pytest.raises(ValueError, match="shared expert gate.*child module"):
        contract.bind(block)
    block.blend = nn.Linear(4, 1)
    block.moe_contract = contract
    binding = resolve_moe_routing(block)
    assert binding.shared_gate is block.blend
    assert binding.shared is block.shared
