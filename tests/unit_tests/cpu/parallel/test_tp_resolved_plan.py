"""TP preflight is local and its bindings cannot silently become stale."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences
from llmtuner.parallel.plan import ParallelPlan
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    ColwiseLinearNoGather,
    apply_tp,
    resolve_tp_plan,
)
from tests.unit_tests.cpu.parallel.test_tp_moe import _MoeModel


def test_resolving_tp_preserves_model_storage_rng_and_forward():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    parameter = model[0].weight
    forward = model[0].forward
    rng = torch.get_rng_state().clone()
    plan = resolve_tp_plan(
        model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
    )
    assert plan.targets[0].module is model[0]
    assert plan.targets[0].spec.kind == "colwise"
    assert plan.stage == "tp"
    assert model[0].weight is parameter
    assert model[0].forward == forward
    assert torch.equal(torch.get_rng_state(), rng)
    plan.validate_binding(model)


def test_invalid_moe_shared_geometry_fails_during_pure_preflight():
    model = _MoeModel()
    block = model.layers[0]["mlp"]
    shared = nn.Module()
    shared.gate_proj = nn.Linear(16, 7, bias=False)
    shared.up_proj = nn.Linear(16, 7, bias=False)
    shared.down_proj = nn.Linear(7, 16, bias=False)
    block.shared_expert = shared
    parameters = tuple(model.parameters())
    with pytest.raises(ValueError, match="not divisible"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert not getattr(block, "_tp_moe_boundary", False)


@pytest.mark.parametrize("replacement", ["module", "parameter", "alias", "new_module"])
def test_resolved_tp_rejects_replaced_bindings_before_mesh_access(replacement):
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    if replacement == "module":
        model[0] = nn.Linear(8, 12, bias=False)
    elif replacement == "parameter":
        model[0].weight = nn.Parameter(model[0].weight.detach().clone())
    elif replacement == "alias":
        model.register_parameter("external_alias", model[0].weight)
    else:
        model.add_module("extra", nn.Identity())
    with pytest.raises(ValueError, match="Stale plan"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)


def test_tp_preflight_rejects_invalid_geometry_without_replacing_parameters():
    model = nn.Sequential(nn.Linear(8, 11, bias=False))
    parameter = model[0].weight
    with pytest.raises(ValueError, match="not divisible"):
        resolve_tp_plan(
            model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
        )
    assert model[0].weight is parameter


def test_resolved_tp_rejects_ambiguous_or_mismatched_requests():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    with pytest.raises(ValueError, match="either plan or resolved_plan"):
        apply_tp(model, object(), cfg, plan={"0": "colwise"}, resolved_plan=plan)
    with pytest.raises(ValueError, match="degree"):
        apply_tp(
            model, object(), ParallelConfig(tensor_parallel_size=4), resolved_plan=plan
        )
    with pytest.raises(ValueError, match="Expected a TP plan"):
        apply_tp(
            model,
            object(),
            cfg,
            resolved_plan=ParallelPlan("ep", (), 2, "alltoall"),
        )


def test_attention_realizer_and_boundary_are_resolved_before_execution():
    attention = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        attention.add_module(name, nn.Linear(8, 8, bias=False))
    model = nn.Sequential(attention)
    plan = resolve_tp_plan(
        model,
        cfg=ParallelConfig(tensor_parallel_size=2),
        plan={f"0.{name}": "colwise" for name in ("q_proj", "k_proj", "v_proj")},
    )
    boundary = next(target for target in plan.targets if target.kind == "attention")
    assert boundary.module is attention
    assert boundary.layout == "gather"
    assert all(
        target.spec.implementation is ColwiseLinearNoGather
        for target in plan.targets
        if target.kind == "dense"
    )


def test_tp_parameter_mapping_rebinds_fsdp_exclusions():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    original = model[0]
    weight = original.weight
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        fsdp_config=FSDPConfig(fsdp_apply_modules={"": {"ignored_params": {weight}}}),
    )
    references = FSDPPatternReferences(model, cfg)
    axis = SimpleNamespace(
        device_type="cpu",
        size=lambda: 2,
        get_group=lambda: None,
        get_local_rank=lambda: 0,
    )
    apply_tp(
        model,
        {"tp": axis},
        cfg,
        plan={"0": "colwise"},
        on_replacement=references.record_replacements,
    )
    assert model[0] is not original
    assert model[0].weight is not weight
    bound = references.bind(model)
    assert bound.fsdp_config.fsdp_apply_modules[""]["ignored_params"] == {
        model[0].weight
    }
