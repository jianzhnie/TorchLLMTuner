"""PP-local EP declarations must retain globally checked conversion semantics."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel import parallelize
from llmtuner.parallel.expert_parallel.plan import (
    resolve_ep_plan,
    validate_pipeline_ep_plan,
)
from llmtuner.parallel.pipeline_parallel.partition import prune_declared_stage
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def plans():
    config = model_config(num_hidden_layers=2, mlp_only_layers=[])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    whole = Qwen3MoeForCausalLM(config)
    global_plan = resolve_ep_plan(whole, degree=2, implementation="alltoall")
    part = deepcopy(whole)
    prune_declared_stage(
        part, {"model.embed_tokens", "model.layers.0", "model.rotary_emb"}
    )
    part.configure_pipeline_stage(first=True, last=False)
    local_plan = resolve_ep_plan(part, degree=2, implementation="alltoall")
    return whole, part, global_plan, local_plan


def test_pipeline_ep_accepts_fresh_binding_without_mutation():
    whole, part, global_plan, local_plan = plans()
    state = {name: value.clone() for name, value in part.state_dict().items()}
    rng = torch.get_rng_state().clone()
    validate_pipeline_ep_plan(global_plan, local_plan, part)
    assert len(global_plan.targets) == 2
    assert len(local_plan.targets) == 1
    assert local_plan.targets[0].module is part.model.layers[0].mlp
    assert local_plan.targets[0].module is not whole.model.layers[0].mlp
    for name, value in part.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("change", ["missing", "layout", "routing", "geometry"])
def test_pipeline_ep_rejects_changed_local_conversion(change):
    _, part, global_plan, local_plan = plans()
    target = local_plan.targets[0]
    if change == "missing":
        targets = ()
    elif change == "layout":
        targets = (replace(target, layout="another_layout"),)
    elif change == "routing":
        binding = replace(
            target.spec, routing=replace(target.spec.routing, route_scale=2.0)
        )
        targets = (replace(target, spec=binding),)
    else:
        binding = replace(
            target.spec,
            packed_experts=replace(target.spec.packed_experts, shape=(4, 16, 16)),
        )
        targets = (replace(target, spec=binding),)
    with pytest.raises(ValueError, match="Pipeline EP"):
        validate_pipeline_ep_plan(
            global_plan, replace(local_plan, targets=targets), part
        )


def test_pipeline_ep_rejects_live_model_routing_drift_after_partition():
    _, part, global_plan, _ = plans()
    block = part.model.layers[0].mlp
    block.moe_contract = replace(block.moe_contract, route_scale=2.0)
    local_plan = resolve_ep_plan(part, degree=2, implementation="alltoall")
    with pytest.raises(ValueError, match="conversion semantics changed"):
        validate_pipeline_ep_plan(global_plan, local_plan, part)


@pytest.mark.parametrize(
    "change", [{"stage": "tp"}, {"degree": 4}, {"implementation": "mc2"}]
)
def test_pipeline_ep_rejects_changed_execution(change):
    _, part, global_plan, local_plan = plans()
    with pytest.raises(ValueError, match="same execution"):
        validate_pipeline_ep_plan(global_plan, replace(local_plan, **change), part)


def test_pipeline_ep_rejects_global_instance_bindings():
    _, part, global_plan, _ = plans()
    with pytest.raises(ValueError, match="Stale plan"):
        validate_pipeline_ep_plan(global_plan, global_plan, part)


@pytest.mark.parametrize("invalid_part", [0, 1])
def test_pipeline_checks_all_local_ep_plans_before_transform(monkeypatch, invalid_part):
    whole, part, _, _ = plans()
    parts = [deepcopy(part), deepcopy(part)]
    block = parts[invalid_part].model.layers[0].mlp
    block.moe_contract = replace(block.moe_contract, route_scale=2.0)
    cfg = ParallelConfig(pipeline_parallel_size=2)
    cfg.ep_config.fixed_router = True
    topology = SimpleNamespace(
        pp_enabled=True,
        get_optional_mesh=lambda name: None,
        spmd_dense_mesh=lambda: None,
    )
    monkeypatch.setattr(
        parallelize, "validate_fsdp_environment", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(parallelize, "validate_microbatches", lambda *args: None)
    monkeypatch.setattr(
        parallelize,
        "apply_pp",
        lambda *args, **kwargs: (
            [SimpleNamespace(submod=part) for part in parts],
            parts,
            True,
            False,
        ),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Local transformation began before all EP plans were checked")

    for name in ("apply_tp", "apply_ep", "apply_ac", "apply_fsdp"):
        monkeypatch.setattr(parallelize, name, unexpected)
    with pytest.raises(ValueError, match="Pipeline EP conversion semantics changed"):
        parallelize.parallelize_model(
            whole,
            cfg=cfg,
            mesh=None,
            parallel_dims=topology,
            global_batch_size=2,
        )
