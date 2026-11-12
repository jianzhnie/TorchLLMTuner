"""Explicit Qwen3.5 must not silently ignore requested numerical policies."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextModel,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config

DEFAULTS = {
    "norm_backend": "native",
    "qkv_projection": "separate",
    "rotary_init_device": "constructor",
    "rotary_compute_dtype": None,
    "rotary_cache_dtype": None,
}


@pytest.mark.parametrize("model_type", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
@pytest.mark.parametrize(
    "field,value",
    [
        ("norm_backend", "torch"),
        ("norm_backend", "npu"),
        ("norm_backend", "residual"),
        ("qkv_projection", "fused"),
        ("rotary_init_device", "target"),
        ("rotary_compute_dtype", "float32"),
        ("rotary_cache_dtype", "bfloat16"),
        ("rotary_cache_dtype", "float32"),
    ],
)
def test_unsupported_policy_fails_before_allocation_and_rng(
    model_type, field, value, monkeypatch
):
    config = text_config(Qwen3_5MoeTextConfig)
    setattr(config, field, value)
    before = deepcopy(config.to_dict())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported policy reached model allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    with pytest.raises(UnsupportedCombinationError, match=field):
        model_type(config)
    assert config.to_dict() == before
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("model_type", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
def test_explicit_native_settings_preserve_construction_and_forward(model_type):
    config = text_config(Qwen3_5MoeTextConfig)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    explicit = deepcopy(config)
    for name, value in DEFAULTS.items():
        setattr(explicit, name, value)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        native = model_type(config)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(83)
        selected = model_type(explicit)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    actual, expected = selected.state_dict(), native.state_dict()
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    tokens = torch.tensor([[1, 7, 3, 9]])
    # Training explicitly disables cache; the all-GDN fixture has no KV layer.
    left, right = selected(tokens, use_cache=False), native(tokens, use_cache=False)
    left = left.logits if hasattr(left, "logits") else left.last_hidden_state
    right = right.logits if hasattr(right, "logits") else right.last_hidden_state
    assert left.isfinite().all() and left.count_nonzero() > 0
    torch.testing.assert_close(left, right, rtol=0, atol=0)
