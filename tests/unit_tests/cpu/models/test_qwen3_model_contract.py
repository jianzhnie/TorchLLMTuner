"""Qwen3 declarations describe real storage without changing model state."""

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_plan


def model_config(**kwargs):
    values = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[1],
        tie_word_embeddings=True,
    )
    return Qwen3MoeConfig(**(values | kwargs))


def test_exact_plan_preserves_rng_aliases_and_state_keys():
    model = Qwen3MoeForCausalLM(model_config())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    plan = model.tp_config
    assert plan["model.layers.0.mlp.experts.gate_up_proj"] == "packed_colwise"
    assert plan["model.layers.1.mlp.gate_proj"] == "colwise"
    assert not any("*" in name for name in plan)
    assert "model.embed_tokens" not in plan
    assert model.enable_weight_tying
    for path in plan:
        if path.endswith(("experts.gate_up_proj", "experts.down_proj")):
            assert isinstance(model.get_parameter(path), nn.Parameter)
        else:
            assert isinstance(model.get_submodule(path), nn.Module)
    with pytest.raises(TypeError):
        plan["lm_head"] = "rowwise"
    model.validate_parallel(tp_degree=2, ep_degree=1)
    assert tuple(model.state_dict()) == keys
    assert torch.equal(rng, torch.get_rng_state())


def test_wrapper_prefers_model_plan_over_hf_generated_patterns():
    model = HFTransformerModel(model_config())
    model.model._tp_config = {"invalid_hf_target": "colwise"}
    assert "model.model.layers.0.self_attn.q_proj" in model.tp_config
    assert not any("invalid_hf_target" in path for path in model.tp_config)


@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
@pytest.mark.parametrize("axis", ["tp_degree", "ep_degree"])
def test_parallel_contract_rejects_invalid_degrees(axis, degree):
    model = Qwen3MoeForCausalLM(model_config())
    values = {"tp_degree": 1, "ep_degree": 1, axis: degree}
    with pytest.raises(ValueError, match="positive integer"):
        model.validate_parallel(**values)


def test_parallel_contract_checks_actual_packed_storage():
    model = Qwen3MoeForCausalLM(model_config())
    model.model.layers[0].mlp.experts.gate_up_proj = nn.Parameter(
        torch.empty(4, 12, 16)
    )
    with pytest.raises(ValueError, match="gate/up storage"):
        model.validate_parallel(tp_degree=1, ep_degree=1)


def test_joint_layout_does_not_require_expert_width_divisible_by_tp():
    model = Qwen3MoeForCausalLM(model_config(moe_intermediate_size=13))
    model.validate_parallel(tp_degree=2, ep_degree=2)
    with pytest.raises(ValueError, match="Expert width"):
        model.validate_parallel(tp_degree=2, ep_degree=1)


def test_meta_parallel_contract_uses_shapes_without_materialization():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(model_config())
    model.validate_parallel(tp_degree=2, ep_degree=2)
    assert model.model.embed_tokens.weight.is_meta


def test_default_tp_preflight_consumes_model_geometry_contract():
    model = HFTransformerModel(model_config(moe_intermediate_size=13))
    with pytest.raises(ValueError, match="Expert width"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    plan = resolve_tp_plan(
        model,
        cfg=ParallelConfig(tensor_parallel_size=2, expert_parallel_size=2),
    )
    assert plan.count("moe") == 0  # EP owns the experts in the joint layout.


def test_partial_tp_plan_does_not_validate_unused_dense_width():
    model = HFTransformerModel(model_config(num_experts=0, intermediate_size=13))
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="Dense width"):
        resolve_tp_plan(model, cfg=cfg)
    plan = resolve_tp_plan(
        model, cfg=cfg, plan={"model.model.layers.0.self_attn.q_proj": "colwise"}
    )
    assert plan.count("dense") == 1
    target = next(target for target in plan.targets if target.kind == "dense")
    assert target.path == "model.model.layers.0.self_attn.q_proj"
