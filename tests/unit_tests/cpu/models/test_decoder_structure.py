"""Explicit structural binding must preserve storage and initialization."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM


def small_config():
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=True,
    )


def test_binding_preserves_rng_parameters_and_state():
    model = Qwen3MoeForCausalLM(small_config())
    state = tuple(model.state_dict())
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    bound = model.decoder_structure.bind(model)
    assert bound.decoder is model.model
    assert bound.blocks is model.model.layers
    assert bound.embedding.weight is bound.head.weight
    assert tuple(model.state_dict()) == state
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_declaration_rejects_missing_or_misowned_targets():
    model = Qwen3MoeForCausalLM(small_config())
    with pytest.raises(AttributeError):
        replace(model.decoder_structure, norm="model.missing").bind(model)
    with pytest.raises(ValueError, match="direct child"):
        replace(model.decoder_structure, embedding="lm_head").bind(model)
    with pytest.raises(TypeError, match="ModuleList"):
        replace(model.decoder_structure, blocks="model.norm").bind(model)


def test_production_wrapper_uses_declaration_without_candidate_search(monkeypatch):
    def reject_inspection(*args, **kwargs):
        raise AssertionError("Explicit model must not search candidate attributes")

    monkeypatch.setattr(
        "llmtuner.models.hf.model.resolve_submodule_name", reject_inspection
    )
    model = HFTransformerModel(small_config())
    assert model.tok_embeddings is model.model.model.embed_tokens
    assert model.enable_weight_tying
    assert not any(key.startswith("_decoder.") for key in model.state_dict())
    replacement = nn.LayerNorm(16)
    model.model.model.norm = replacement
    assert model.norm is replacement
    assert model.model.decoder_structure.bind(model.model).norm is replacement

    from llmtuner.parallel.context_parallel import context_parallel

    monkeypatch.setattr(context_parallel, "_ATTN_MODULE_NAMES", ())
    attentions = context_parallel._attention_modules(model)
    assert attentions == [layer.self_attn for layer in model.layers]


def test_meta_binding_does_not_materialize_parameters():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(small_config())
    bound = model.decoder_structure.bind(model)
    assert bound.embedding.weight.is_meta
    assert bound.embedding.weight is bound.head.weight
