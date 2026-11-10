"""Qwen3 dense/MoE nonpersistent RoPE must survive HF/meta versus native resume."""

import pytest
import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.hf.materialization import (
    _construct_rotary,
    _rotary_reconstruction_device,
    materialize_meta_model,
    validate_rotary_initialization,
)


def test_moe_explicit_target_policy_remains_unsupported():
    config = Qwen3MoeConfig(head_dim=128)
    config.rotary_init_device = "target"
    with pytest.raises(UnsupportedCombinationError, match="requires dense Qwen3"):
        validate_rotary_initialization(config)


@pytest.mark.parametrize(
    "config_type,rotary_type",
    [(Qwen3Config, Qwen3RotaryEmbedding), (Qwen3MoeConfig, Qwen3MoeRotaryEmbedding)],
)
def test_constructor_policy_reconstructs_cpu_arithmetic_for_accelerator_transfer(
    config_type, rotary_type
):
    config = config_type(head_dim=128)
    rotary = _construct_rotary(rotary_type, config, torch.device("meta"))
    assert _rotary_reconstruction_device(rotary, torch.device("cuda")) == (
        torch.device("cpu")
    )
    config.rotary_init_device = "target"
    assert _rotary_reconstruction_device(rotary, torch.device("cuda")) == (
        torch.device("cuda")
    )
    # Other native architectures retain their existing construction contract.
    assert _rotary_reconstruction_device(nn.Module(), torch.device("cuda")) == (
        torch.device("cuda")
    )


@pytest.mark.parametrize(
    "config_type,rotary_type",
    [(Qwen3Config, Qwen3RotaryEmbedding), (Qwen3MoeConfig, Qwen3MoeRotaryEmbedding)],
)
def test_meta_reconstruction_matches_constructor_buffers_and_bf16_caches(
    config_type, rotary_type
):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        config = config_type(head_dim=128)
        reference = _construct_rotary(rotary_type, config, torch.device("cpu"))
        model = nn.Module()
        model.rotary = _construct_rotary(rotary_type, config, torch.device("meta"))
        model.register_buffer(
            "frequency_alias", model.rotary.inv_freq, persistent=False
        )
        assert model.state_dict() == {}
        materialize_meta_model(model, torch.device("cpu"))
        assert model.frequency_alias is model.rotary.inv_freq
        assert model.rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(
            model.rotary.inv_freq, reference.inv_freq, rtol=0, atol=0
        )
        value = torch.zeros(1, 4096, 128, dtype=torch.bfloat16)
        positions = torch.arange(4096)[None]
        torch.testing.assert_close(
            model.rotary(value, positions), reference(value, positions), rtol=0, atol=0
        )
    finally:
        torch.set_default_dtype(previous)
