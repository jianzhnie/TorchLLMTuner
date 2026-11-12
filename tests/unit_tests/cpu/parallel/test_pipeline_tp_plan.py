"""PP-local TP targets must retain their globally validated semantics."""

from copy import deepcopy
from dataclasses import replace

import pytest
from torch import nn

from llmtuner.config.parallel_config import ParallelConfig, TensorParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.pipeline_parallel.partition import prune_declared_stage
from llmtuner.parallel.plan import ParallelPlan, PlanTarget
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    resolve_tp_plan,
    validate_pipeline_tp_plan,
)
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def plans():
    whole = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    part = deepcopy(whole)
    part[1] = nn.Identity()
    global_plan = ParallelPlan(
        "tp",
        (
            PlanTarget("0", "dense", whole[0], "colwise", "column_kernel"),
            PlanTarget("1", "dense", whole[1], "rowwise", "row_kernel"),
        ),
        2,
        "unbound",
    )
    local_plan = replace(
        global_plan,
        targets=(replace(global_plan.targets[0], module=part[0]),),
    )
    return whole, part, global_plan, local_plan


def test_pipeline_tp_accepts_fresh_local_bindings_and_nonlocal_targets():
    whole, part, global_plan, local_plan = plans()
    validate_pipeline_tp_plan(global_plan, local_plan, part)
    assert local_plan.targets[0].module is part[0]
    assert local_plan.targets[0].module is not whole[0]


@pytest.mark.parametrize("change", ["missing", "unexpected", "layout", "kernel"])
def test_pipeline_tp_rejects_local_target_drift(change):
    _, part, global_plan, local_plan = plans()
    target = local_plan.targets[0]
    if change == "missing":
        targets = ()
    elif change == "unexpected":
        targets = local_plan.targets + (PlanTarget("1", "dense", part[1]),)
    elif change == "layout":
        targets = (replace(target, layout="rowwise"),)
    else:
        targets = (replace(target, spec="another_kernel"),)
    with pytest.raises(ValueError, match="Pipeline TP"):
        validate_pipeline_tp_plan(
            global_plan, replace(local_plan, targets=targets), part
        )


@pytest.mark.parametrize("change", [{"stage": "ep"}, {"degree": 4}])
def test_pipeline_tp_rejects_another_execution(change):
    _, part, global_plan, local_plan = plans()
    with pytest.raises(ValueError, match="same stage and degree"):
        validate_pipeline_tp_plan(global_plan, replace(local_plan, **change), part)


def test_pipeline_tp_rejects_global_references_in_local_plan():
    _, part, global_plan, _ = plans()
    with pytest.raises(ValueError, match="Stale plan binding"):
        validate_pipeline_tp_plan(global_plan, global_plan, part)


def test_owned_qwen_pipeline_tp_plans_preserve_global_targets():
    config = model_config(num_hidden_layers=2, mlp_only_layers=[1])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    whole = Qwen3MoeForCausalLM(config)
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        tp_config=TensorParallelConfig(
            tp_colwise_modules=["model.layers.0.self_attn.q_proj"],
            tp_rowwise_modules=["model.layers.1.self_attn.o_proj"],
        ),
    )
    global_plan = resolve_tp_plan(whole, cfg=cfg)
    for index, names in enumerate(
        (
            {"model.embed_tokens", "model.layers.0", "model.rotary_emb"},
            {"model.layers.1", "model.norm", "lm_head", "model.rotary_emb"},
        )
    ):
        part = deepcopy(whole)
        prune_declared_stage(part, names)
        part.configure_pipeline_stage(first=index == 0, last=index == 1)
        local_plan = resolve_tp_plan(part, cfg=cfg)
        validate_pipeline_tp_plan(global_plan, local_plan, part)
        assert local_plan.targets
        assert all(
            target.module is part.get_submodule(target.path)
            for target in local_plan.targets
        )


@pytest.mark.parametrize("scope", ["global", "local"])
def test_pipeline_tp_rejects_duplicate_target_ownership(scope):
    _, part, global_plan, local_plan = plans()
    if scope == "global":
        global_plan = replace(
            global_plan, targets=global_plan.targets + (global_plan.targets[0],)
        )
    else:
        local_plan = replace(local_plan, targets=local_plan.targets * 2)
    with pytest.raises(ValueError, match="duplicate targets"):
        validate_pipeline_tp_plan(global_plan, local_plan, part)


@pytest.mark.parametrize("requirement", [False, None, 1])
def test_pipeline_tp_rejects_changed_token_history_contract(requirement):
    _, part, global_plan, local_plan = plans()
    global_plan = replace(global_plan, token_history_requirements=(("0", True),))
    part[0].requires_full_token_history = requirement
    local_plan = replace(local_plan, token_history_requirements=(("0", requirement),))
    # Each plan is internally valid. The new guard detects divergence between
    # the independently resolved global and local contracts.
    local_plan.validate_binding(part)
    with pytest.raises(ValueError, match="token history requirement changed"):
        validate_pipeline_tp_plan(global_plan, local_plan, part)
