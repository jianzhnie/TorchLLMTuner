"""Runtime options must reach the text config of composite HF models."""

import pytest
from transformers import PretrainedConfig

from llmtuner.models.hf.factory import unwrap_text_config


@pytest.mark.parametrize(
    "name,value",
    [
        ("attn_mask_type", "block_causal"),
        ("compute_dtype", "float32"),
        ("experts_implementation", "eager"),
    ],
)
def test_composite_config_passes_explicit_runtime_options(name, value):
    text = PretrainedConfig()
    text.architectures = ["LlamaForCausalLM"]
    top = PretrainedConfig()
    top.text_config = text
    setattr(top, name, value)
    assert unwrap_text_config(top) is text
    assert getattr(text, name) == value


def test_composite_config_preserves_text_options_when_parent_has_none():
    text = PretrainedConfig()
    text.architectures = ["LlamaForCausalLM"]
    text.compute_dtype = "bfloat16"
    top = PretrainedConfig()
    top.text_config = text
    assert unwrap_text_config(top).compute_dtype == "bfloat16"


@pytest.mark.parametrize("seq_len, expected", [(16, 32), (64, 64)])
def test_local_composite_model_extends_decoder_context(tmp_path, seq_len, expected):
    from transformers import AutoConfig

    from llmtuner.models.hf.factory import build_model_config

    saved = AutoConfig.for_model(
        "qwen2_5_vl",
        text_config={
            "vocab_size": 257,
            "hidden_size": 96,
            "intermediate_size": 192,
            "num_hidden_layers": 3,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    )
    saved.save_pretrained(tmp_path)
    actual = build_model_config(str(tmp_path), seq_len=seq_len)
    assert actual.text_config.max_position_embeddings == expected
    assert actual.vision_config.to_dict() == saved.vision_config.to_dict()
