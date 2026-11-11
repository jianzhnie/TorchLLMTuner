"""Explicit Qwen model subclasses preserve native HF forward semantics."""

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel as NativeQwen35TextModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextModel as NativeQwen35MoeTextModel,
)

from llmtuner.models.qwen.qwen3_5.multimodal import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextModel,
)
from llmtuner.models.qwen.qwen3_5_moe.multimodal import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextModel,
)


def small_config(model_type: str):
    return AutoConfig.for_model(
        model_type,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )


def test_explicit_dense_text_model_matches_native() -> None:
    config = small_config("qwen3_5_text")
    native = NativeQwen35TextModel(config).eval()
    explicit = Qwen3_5TextModel(config).eval()
    explicit.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = native(input_ids=tokens, use_cache=False).last_hidden_state
        actual = explicit(input_ids=tokens, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_explicit_moe_text_model_matches_native() -> None:
    config = small_config("qwen3_5_moe_text")
    native = NativeQwen35MoeTextModel(config).eval()
    explicit = Qwen3_5MoeTextModel(config).eval()
    explicit.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = native(input_ids=tokens, use_cache=False).last_hidden_state
        actual = explicit(input_ids=tokens, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "model_type,decoder_type",
    [
        ("qwen3_5_text", Qwen3_5DecoderLayer),
        ("qwen3_5_moe_text", Qwen3_5MoeDecoderLayer),
    ],
)
def test_invalid_layer_type_fails_before_parameter_allocation(model_type, decoder_type):
    config = small_config(model_type)
    config.layer_types[0] = "unsupported_attention"
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Unsupported Qwen3.5 layer type"):
        decoder_type(config, 0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "model_type,decoder_type,slot",
    [
        ("qwen3_5_text", Qwen3_5DecoderLayer, None),
        ("qwen3_5_moe_text", Qwen3_5MoeDecoderLayer, "mlp"),
    ],
)
def test_multimodal_text_decoder_declares_moe_slot(model_type, decoder_type, slot):
    layer = decoder_type(small_config(model_type), 0)
    assert layer.moe_block_name == slot
    if slot is not None:
        assert layer._modules[slot] is layer.mlp
