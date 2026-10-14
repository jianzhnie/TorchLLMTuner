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
