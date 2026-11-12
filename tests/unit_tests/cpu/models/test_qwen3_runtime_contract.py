"""Parallel consumers of the explicit Qwen3 model contract."""

from types import SimpleNamespace

import pytest

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_plan
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config
from tests.unit_tests.cpu.models.test_qwen3_parallel_contract import small_config


def test_wrapper_prefers_model_plan_over_hf_generated_patterns():
    model = Qwen3MoeForCausalLM(model_config())
    assert "model.layers.0.self_attn.q_proj" in model.tp_config
    assert not any("invalid_hf_target" in path for path in model.tp_config)


def test_default_tp_preflight_consumes_model_geometry_contract():
    model = Qwen3MoeForCausalLM(model_config(moe_intermediate_size=13))
    with pytest.raises(ValueError, match="Expert width"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    plan = resolve_tp_plan(
        model,
        cfg=ParallelConfig(tensor_parallel_size=2, expert_parallel_size=2),
    )
    assert plan.count("moe") == 0  # EP owns the experts in the joint layout.


def test_partial_tp_plan_does_not_validate_unused_dense_width():
    model = Qwen3MoeForCausalLM(model_config(num_experts=0, intermediate_size=13))
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="Dense width"):
        resolve_tp_plan(model, cfg=cfg)
    plan = resolve_tp_plan(
        model, cfg=cfg, plan={"model.layers.0.self_attn.q_proj": "colwise"}
    )
    assert plan.count("dense") == 1
    target = next(target for target in plan.targets if target.kind == "dense")
    assert target.path == "model.layers.0.self_attn.q_proj"


def test_ep_only_run_uses_model_capability_validation():
    config = small_config()
    config.num_experts = 3
    model = Qwen3MoeForCausalLM(config)
    with pytest.raises(ValueError, match="Expert count.*EP degree"):
        apply_ep(
            model,
            ParallelConfig(expert_parallel_size=2),
            ep_group=SimpleNamespace(size=lambda: 2),
        )
