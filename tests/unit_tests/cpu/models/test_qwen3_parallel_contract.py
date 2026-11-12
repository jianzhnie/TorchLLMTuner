"""Qwen3 owns the complete MoE semantics consumed by TP and EP."""

import pytest
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel import hf_inspection
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks


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

    def reject(*args, **kwargs):
        raise AssertionError("Qwen3 must use its explicit MoE contract")

    monkeypatch.setattr(hf_inspection.inspect, "getsource", reject)
    monkeypatch.setattr(hf_inspection, "_fused_experts_of_per_expert_list", reject)
    monkeypatch.setattr(hf_inspection, "ignores_norm_topk_prob", reject)
    monkeypatch.setattr(hf_inspection, "read_int_attr", reject)

    assert hf_inspection.is_hf_moe_block(block)
    assert hf_inspection.router_of(block) is block.gate
    assert hf_inspection.resolve_top_k(block) == 2
    assert hf_inspection.resolve_score_func(block, block.gate) == "softmax"
    assert (
        hf_inspection.read_route_norm(block, block.gate) == model.config.norm_topk_prob
    )
    assert hf_inspection.read_route_scale(block, block.gate) == 1.0
    assert hf_inspection.read_expert_groups(block, block.gate) == (None, None)
    assert hf_inspection.resolve_shared_expert(block) == ("shared_expert", None)
    assert not hf_inspection.returns_router_logits(block)

    assert swap_hf_moe_blocks(model.model) == 1
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
        hf_inspection.is_hf_moe_block(block)
