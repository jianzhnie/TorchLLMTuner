"""Model-owned partial/mRoPE preserves buffers and dynamic growth/reset."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextRotaryEmbedding as ReferenceRotary,
)

from llmtuner.models.qwen.qwen3_5_moe.model import initialize_model_weights
from llmtuner.models.qwen.qwen3_5_moe.rotary import Qwen3_5MoeTextRotaryEmbedding
from tests.unit_tests.cpu.models.test_qwen35_rotary_initialization import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("multiaxis", [False, True])
def test_owned_rotary_matches_reference_without_construction_or_forward_delegation(
    monkeypatch, dtype, dynamic, multiaxis
):
    config = text_config(Qwen3_5MoeTextConfig, dynamic=dynamic)
    config.max_position_embeddings = 8
    expected = ReferenceRotary(deepcopy(config))

    def forbidden(*args, **kwargs):
        pytest.fail("Owned rotary delegated construction or forward")

    monkeypatch.setattr(ReferenceRotary, "__init__", forbidden)
    rng = torch.get_rng_state().clone()
    actual = Qwen3_5MoeTextRotaryEmbedding(deepcopy(config))
    assert torch.equal(torch.get_rng_state(), rng)
    assert not actual.state_dict()
    assert actual._non_persistent_buffers_set == expected._non_persistent_buffers_set
    for length in (3, 12, 4):
        positions = torch.arange(length)[None].expand(2, -1)
        if multiaxis:
            positions = torch.stack((positions, positions + 1, positions * 2))
        hidden = torch.ones(2, length, config.hidden_size, dtype=dtype)
        target = expected(hidden, positions)
        observed = actual(hidden, positions)
        for left, right in zip(observed, target, strict=True):
            assert left.dtype == dtype
            assert left.isfinite().all() and left.count_nonzero() > 0
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert actual.max_seq_len_cached == expected.max_seq_len_cached
        for name in ("inv_freq", "original_inv_freq"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0
            )
    # Exercise the owned implementation after disabling the inherited method.
    monkeypatch.setattr(ReferenceRotary, "forward", forbidden)
    result = actual(hidden, positions)
    for left, right in zip(result, observed, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("frequency_loaded", [False, True])
@pytest.mark.parametrize("original_loaded", [False, True])
def test_owned_frequency_initialization_respects_individual_buffer_load_markers(
    dynamic, frequency_loaded, original_loaded
):
    config = text_config(Qwen3_5MoeTextConfig, dynamic=dynamic)
    expected = ReferenceRotary(deepcopy(config))
    actual = Qwen3_5MoeTextRotaryEmbedding(deepcopy(config))
    names = ("inv_freq", "original_inv_freq")
    buffers = tuple(getattr(actual, name) for name in names)
    for buffer, loaded in zip(
        buffers, (frequency_loaded, original_loaded), strict=True
    ):
        buffer.fill_(0.25)
        buffer._is_hf_initialized = loaded
    rng = torch.get_rng_state().clone()
    assert initialize_model_weights(actual, 0.02)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, buffer, loaded in zip(
        names, buffers, (frequency_loaded, original_loaded), strict=True
    ):
        assert getattr(actual, name) is buffer
        target = torch.full_like(buffer, 0.25) if loaded else getattr(expected, name)
        torch.testing.assert_close(buffer, target, rtol=0, atol=0)
    assert not actual.state_dict()
