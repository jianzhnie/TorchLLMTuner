"""Expert-parallel wiring: the single-rank guards.

The swap and multi-rank all-to-all live in
``tests/test_ep_model_replacement.py`` and ``tests/ep_wiring_equivalence.py``.
What can be checked without a mesh is the
refusal path: EP is wired, so without the sparse mesh's EP group it must refuse
loudly rather than silently swap to a replicated (local) dispatcher -- and with
EP off it must leave the model untouched.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch

from llmtuner.config import ParallelConfig
from llmtuner.parallel.expert_parallel import apply_ep


def test_apply_ep_is_a_no_op_when_ep_is_off() -> None:
    """EP=1 must hand the model back untouched rather than raise."""
    model = torch.nn.Linear(4, 4)
    cfg = ParallelConfig(expert_parallel_size=1)

    assert apply_ep(model, cfg) is model


def test_apply_ep_requires_an_ep_group_when_ep_is_on() -> None:
    """Without the sparse mesh's EP group it must refuse loudly rather than
    silently swap to a replicated (local) dispatcher."""
    cfg = ParallelConfig(expert_parallel_size=2)

    with pytest.raises(ValueError, match="EP process group"):
        apply_ep(torch.nn.Linear(4, 4), cfg)


def test_expert_layout_rejects_inconsistent_down_width():
    from llmtuner.parallel.expert_parallel.hf_inspection import fused_experts_of

    block = torch.nn.Module()
    block.experts = torch.nn.Module()
    block.experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 6))
    block.experts.down_proj = torch.nn.Parameter(torch.randn(2, 6, 3))
    with pytest.raises(ValueError, match="unrecognized expert weight shapes"):
        fused_experts_of(block)


@pytest.mark.parametrize(
    "names", [("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")]
)
def test_legacy_expert_inspection_preserves_freezing_without_grad_tracking(names):
    from llmtuner.parallel.expert_parallel.hf_inspection import fused_experts_of

    block = torch.nn.Module()
    block.experts = torch.nn.ModuleList()
    for _ in range(2):
        expert = torch.nn.Module()
        for name, shape, trainable in zip(
            names, ((6, 4), (6, 4), (4, 6)), (True, False, True), strict=True
        ):
            projection = torch.nn.Linear(*shape, bias=False)
            projection.weight.requires_grad_(trainable)
            setattr(expert, name, projection)
        block.experts.append(expert)
    with torch.no_grad():
        result = fused_experts_of(block)
    assert tuple(tensor.requires_grad for tensor in result[:3]) == (True, False, True)
    for name, tensor in zip(names, result[:3], strict=True):
        expected = torch.stack([getattr(e, name).weight for e in block.experts])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "names", [("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")]
)
@pytest.mark.parametrize(
    "invalid",
    [
        "activation",
        "gate_bias",
        "up_bias",
        "down_bias",
        "frozen_gate",
        "frozen_up",
        "mixed_freezing",
        "shared_expert_projection",
    ],
)
def test_legacy_expert_contract_checks_later_experts(names, invalid):
    from llmtuner.parallel.expert_parallel.hf_inspection import fused_experts_of

    block = torch.nn.Module()
    block.experts = torch.nn.ModuleList()
    for index in range(2):
        expert = torch.nn.Module()
        expert.act_fn = (
            torch.nn.ReLU()
            if index == 1 and invalid == "activation"
            else torch.nn.SiLU()
        )
        for name, role, shape in zip(
            names,
            ("gate_bias", "up_bias", "down_bias"),
            ((4, 6), (4, 6), (6, 4)),
            strict=True,
        ):
            setattr(
                expert,
                name,
                torch.nn.Linear(*shape, bias=index == 1 and invalid == role),
            )
        block.experts.append(expert)
    if invalid == "mixed_freezing":
        from llmtuner.errors import UnsupportedCombinationError

        getattr(block.experts[1], names[0]).weight.requires_grad_(False)
        with pytest.raises(UnsupportedCombinationError, match="requires_grad"):
            fused_experts_of(block)
        return
    if invalid == "shared_expert_projection":
        from llmtuner.errors import UnsupportedCombinationError

        getattr(block.experts[1], names[0]).weight = getattr(
            block.experts[0], names[0]
        ).weight
        with pytest.raises(UnsupportedCombinationError, match="shared.*expert"):
            fused_experts_of(block)
        return
    if invalid in ("frozen_gate", "frozen_up"):
        frozen = 0 if invalid == "frozen_gate" else 1
        for expert in block.experts:
            getattr(expert, names[frozen]).weight.requires_grad_(False)
        result = fused_experts_of(block)
        for index, tensor in enumerate(result[:3]):
            assert tensor.requires_grad == (index != frozen)
            assert torch.equal(
                tensor,
                torch.stack([getattr(e, names[index]).weight for e in block.experts]),
            )
        return
    with pytest.raises(NotImplementedError, match="activation|bias"):
        fused_experts_of(block)
