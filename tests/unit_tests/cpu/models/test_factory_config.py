"""Runtime options must reach the text config of composite HF models."""

import pytest
from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from llmtuner.config import LLMTunerConfig, ModelConfig
from llmtuner.models.configuration import (
    build_model_config,
    build_model_config_for,
    unwrap_text_config,
)


@pytest.mark.parametrize("expert_width", [2048, 3072])
def test_canonical_expert_width_overrides_generic_alias_default(expert_width):
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="deepseek_v4",
            arch_overrides={"moe_intermediate_size": expert_width},
        )
    )
    actual = build_model_config_for(config)
    assert actual.moe_intermediate_size == expert_width
    assert actual.intermediate_size == expert_width
    assert config.model.arch_overrides == {"moe_intermediate_size": expert_width}


def test_explicit_alias_still_sets_expert_width():
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="deepseek_v4",
            arch_overrides={"intermediate_size": 96},
        )
    )
    actual = build_model_config_for(config)
    assert actual.moe_intermediate_size == 96


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
    saved = Qwen3_5Config(
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


@pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe"])
@pytest.mark.parametrize("saved_limit,expected", [(16, 64), (128, 128)])
def test_offline_and_saved_config_apply_same_text_context_requirement(
    tmp_path, model_type, saved_limit, expected
):
    overrides = (
        {"text_config": {"max_position_embeddings": saved_limit}}
        if model_type == "qwen3_5_moe"
        else {"max_position_embeddings": saved_limit}
    )
    offline = build_model_config(model_type, seq_len=64, arch_overrides=overrides)
    offline.save_pretrained(tmp_path)
    restored = build_model_config(str(tmp_path), seq_len=64)
    for config in (offline, restored):
        text = getattr(config, "text_config", config)
        assert text.max_position_embeddings == expected
    if model_type == "qwen3_5_moe":
        assert offline.vision_config.to_dict() == restored.vision_config.to_dict()
        assert overrides == {"text_config": {"max_position_embeddings": saved_limit}}
    else:
        assert overrides == {"max_position_embeddings": saved_limit}
