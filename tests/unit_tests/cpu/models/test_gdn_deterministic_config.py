"""Configuration propagation only; does not accept NPU determinism."""

from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.models.hf.factory import apply_runtime_config, unwrap_text_config


def test_deterministic_requires_boolean_and_npu_backend():
    assert ModelConfig().gdn_deterministic_reduction is False
    with pytest.raises(ConfigError, match="boolean"):
        ModelConfig(gdn_deterministic_reduction="true")
    with pytest.raises(ConfigError, match="requires npu_triton"):
        ModelConfig(gdn_deterministic_reduction=True)


def test_deterministic_factory_propagates_to_native_text_config():
    cfg = SimpleNamespace(
        model=ModelConfig(
            linear_attention_backend="npu_triton", gdn_deterministic_reduction=True
        ),
        dataloader=SimpleNamespace(attention_mask_type=lambda: "causal"),
        norm_backend="native",
        qkv_projection="separate",
        rotary_init_device="constructor",
        attention_backend="auto",
    )
    outer = PretrainedConfig()
    outer.text_config = PretrainedConfig()
    apply_runtime_config(outer, cfg)
    assert unwrap_text_config(outer).gdn_deterministic_reduction is True
    assert unwrap_text_config(outer).gdn_numerical_policy == "native"
