"""Qwen3 owns the complete MoE semantics consumed by TP and EP."""

import pytest
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.block_conversion import (
    resolve_moe_routing,
    resolve_packed_experts,
)
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks


def small_config():
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )


def test_qwen3_ep_never_uses_legacy_structure_or_source_probes(monkeypatch):
    model = Qwen3MoeForCausalLM(small_config())
    block = model.model.layers[0].mlp

    routing = resolve_moe_routing(block)
    packed = resolve_packed_experts(block, degree=1)
    assert routing.router is block.gate
    assert routing.top_k == 2
    assert routing.score_func == "softmax"
    assert routing.route_norm == model.config.norm_topk_prob
    assert routing.route_scale == 1.0
    assert routing.num_expert_groups is None
    assert routing.shared is None
    assert packed.gate_up is block.experts.gate_up_proj

    assert swap_moe_blocks(model.model) == 1
    plan = model.model.ep_parallel_plan
    assert plan.stage == "ep"
    assert plan.targets[0].path == "layers.0.mlp"
    assert plan.targets[0].module is block
    assert plan.targets[0].layout == "fused_gate_up"


def test_qwen3_contract_rejects_inconsistent_bound_structure():
    model = Qwen3MoeForCausalLM(small_config())
    block = model.model.layers[0].mlp
    block._modules.pop("gate")
    with pytest.raises(ValueError, match="router"):
        resolve_moe_routing(block)
