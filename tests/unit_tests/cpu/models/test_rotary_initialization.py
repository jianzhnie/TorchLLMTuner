"""Explicit training-device initialization of HF runtime RoPE buffers."""

from tests.caps import require_env

require_env("flex_attention", "spmd_types")

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.factory import build_model_config_for, unwrap_text_config
from llmtuner.models.hf.materialization import (
    initialize_rotary_on_device,
    materialize_meta_model,
    validate_rotary_initialization,
)
from llmtuner.models.hf.model import HFTransformerModel


def make_config(policy="target"):
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            rotary_init_device=policy,
            arch_overrides={"head_dim": 8},
        )
    )


@pytest.mark.parametrize("policy", ["constructor", "target"])
def test_rotary_policy_reaches_hf_text_config(policy):
    config = make_config(policy)
    hf_config = build_model_config_for(config)
    assert config.rotary_init_device == policy
    assert hf_config.rotary_init_device == policy
    composite = PretrainedConfig()
    composite.text_config = hf_config
    composite.rotary_init_device = policy
    assert unwrap_text_config(composite).rotary_init_device == policy


def test_invalid_rotary_policy_is_rejected():
    with pytest.raises(ConfigError, match="rotary_init_device"):
        ModelConfig(rotary_init_device="npu")


@pytest.mark.parametrize("policy", ["constructor", "target"])
def test_rotary_initialization_preserves_parameters_and_buffer_aliases(policy):
    model = HFTransformerModel(build_model_config_for(make_config(policy)))
    rotary = model.model.model.rotary_emb
    parameters = {name: id(value) for name, value in model.named_parameters()}
    keys = tuple(model.state_dict())
    rotary.inv_freq.fill_(7)
    rotary.original_inv_freq.fill_(9)
    model.register_buffer("rotary_alias", rotary.inv_freq, persistent=False)
    initialize_rotary_on_device(model, torch.device("cpu"))
    assert model.rotary_alias is rotary.inv_freq
    assert parameters == {name: id(value) for name, value in model.named_parameters()}
    assert keys == tuple(model.state_dict())
    if policy == "constructor":
        assert (rotary.inv_freq == 7).all()
        assert (rotary.original_inv_freq == 9).all()
    else:
        expected = Qwen3RotaryEmbedding(rotary.config, device=torch.device("cpu"))
        torch.testing.assert_close(rotary.inv_freq, expected.inv_freq, atol=0, rtol=0)
        torch.testing.assert_close(
            rotary.original_inv_freq, expected.original_inv_freq, atol=0, rtol=0
        )
        assert rotary.inv_freq is not rotary.original_inv_freq


def test_meta_materialization_uses_target_rotary_policy():
    config = build_model_config_for(make_config())
    with torch.device("meta"):
        model = HFTransformerModel(config)
    materialize_meta_model(model, torch.device("cpu"))
    rotary = model.model.model.rotary_emb
    expected = Qwen3RotaryEmbedding(config, device=torch.device("cpu"))
    assert not rotary.inv_freq.is_meta
    assert not rotary.original_inv_freq.is_meta
    torch.testing.assert_close(rotary.inv_freq, expected.inv_freq, atol=0, rtol=0)
    torch.testing.assert_close(
        rotary.original_inv_freq, expected.original_inv_freq, atol=0, rtol=0
    )


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
def test_target_rotary_rejects_other_families(family):
    config = AutoConfig.for_model(family)
    config.rotary_init_device = "target"
    with pytest.raises(UnsupportedCombinationError, match="dense Qwen3 default"):
        HFTransformerModel(config)


def test_target_rotary_rejects_scaled_rope():
    config = build_model_config_for(make_config())
    config.rope_parameters["rope_type"] = "dynamic"
    with pytest.raises(UnsupportedCombinationError, match="dense Qwen3 default"):
        validate_rotary_initialization(config)


def test_target_rotary_rejects_custom_rotary_class():
    class CustomRotary(Qwen3RotaryEmbedding):
        pass

    config = build_model_config_for(make_config())
    rotary = CustomRotary(config)
    original = rotary.inv_freq
    with pytest.raises(UnsupportedCombinationError, match="native Qwen3"):
        initialize_rotary_on_device(rotary, torch.device("cpu"))
    assert rotary.inv_freq is original


def test_target_rotary_resets_all_runtime_frequency_state():
    config = build_model_config_for(make_config())
    rotary = Qwen3RotaryEmbedding(config).to(torch.bfloat16)
    rotary.max_seq_len_cached = 123
    rotary.original_max_seq_len = 124
    rotary.attention_scaling = 2.0
    initialize_rotary_on_device(rotary, torch.device("cpu"))
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.original_inv_freq.dtype == torch.float32
    assert rotary.max_seq_len_cached == config.max_position_embeddings
    assert rotary.original_max_seq_len == config.max_position_embeddings
    assert rotary.attention_scaling == 1.0
