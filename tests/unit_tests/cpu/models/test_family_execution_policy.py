"""DeepSeek/Kimi policy selectors must describe the model actually constructed."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.deepseek.deepseek_v4.modeling import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM, KimiTextDecoder
from tests.unit_tests.cpu.models.test_owned_family_construction import small_config


def kimi_config():
    return KimiLinearConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )


FAMILIES = [
    (DeepseekV4ForCausalLM, small_config),
    (DeepseekV4Model, small_config),
    (KimiLinearForCausalLM, kimi_config),
    (KimiTextDecoder, kimi_config),
]
DEFAULTS = {
    "norm_backend": "native",
    "qkv_projection": "separate",
    "rotary_init_device": "constructor",
    "rotary_compute_dtype": None,
    "rotary_cache_dtype": None,
}


@pytest.mark.parametrize("model_type,config_factory", FAMILIES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("norm_backend", "npu"),
        ("norm_backend", "torch"),
        ("norm_backend", "residual"),
        ("qkv_projection", "fused"),
        ("rotary_init_device", "target"),
        ("rotary_compute_dtype", "float32"),
        ("rotary_cache_dtype", "bfloat16"),
        ("rotary_cache_dtype", "float32"),
    ],
)
def test_unsupported_policy_fails_before_allocation(
    model_type, config_factory, field, value, monkeypatch
):
    config = config_factory()
    setattr(config, field, value)
    before = deepcopy(config.to_dict())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported policy reached allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    with pytest.raises(UnsupportedCombinationError, match=field):
        model_type(config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert config.to_dict() == before


@pytest.mark.parametrize("model_type,config_factory", FAMILIES)
def test_explicit_native_policy_preserves_parameters_rng_and_output(
    model_type, config_factory
):
    config = config_factory()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    selected_config = deepcopy(config)
    for name, value in DEFAULTS.items():
        setattr(selected_config, name, value)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(63)
        reference = model_type(config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(63)
        candidate = model_type(selected_config)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    actual, expected = candidate.state_dict(), reference.state_dict()
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    tokens = torch.tensor([[1, 7, 3, 9]])
    left = candidate(tokens, use_cache=False)
    right = reference(tokens, use_cache=False)
    left = left.logits if hasattr(left, "logits") else left.last_hidden_state
    right = right.logits if hasattr(right, "logits") else right.last_hidden_state
    assert left.isfinite().all() and left.count_nonzero() > 0
    torch.testing.assert_close(left, right, rtol=0, atol=0)
