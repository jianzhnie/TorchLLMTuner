"""Qwen3.5 owns decoder execution while preserving reference cache and outputs."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextModel as ReferenceDecoder,
)

from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


def models():
    config = text_config(ReferenceModel.config_class)
    config.num_hidden_layers = 2
    config.layer_types = ["linear_attention", "full_attention"]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(61)
    baseline = ReferenceModel(deepcopy(config))
    owned = Qwen3_5MoeForCausalLM(deepcopy(config))
    owned.load_state_dict(baseline.state_dict(), strict=True)
    return baseline.model, owned.model


@pytest.mark.parametrize("mrope", [False, True])
def test_decoder_forward_and_hidden_capture_do_not_use_reference_forward(
    monkeypatch, mrope
):
    baseline, owned = models()
    tokens = torch.tensor([[1, 5, 2, 7]])
    positions = torch.arange(4)[None]
    if mrope:
        positions = torch.stack(
            (positions, positions + 1, positions * 2, positions * 3)
        )
    expected = baseline(
        tokens, position_ids=positions, use_cache=False, output_hidden_states=True
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit decoder delegated its forward to the reference")

    monkeypatch.setattr(ReferenceDecoder, "forward", forbidden)
    actual = owned(
        tokens, position_ids=positions, use_cache=False, output_hidden_states=True
    )
    torch.testing.assert_close(
        actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0
    )
    assert len(actual.hidden_states) == len(expected.hidden_states) == 3
    for left, right in zip(actual.hidden_states, expected.hidden_states, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    probe = torch.randn_like(actual.last_hidden_state)
    actual.last_hidden_state.backward(probe)
    expected.last_hidden_state.backward(probe)
    nonzero = 0
    for name, parameter in owned.named_parameters():
        other = dict(baseline.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        if parameter.grad is not None:
            nonzero += bool(parameter.grad.count_nonzero())
    assert nonzero > 0


def test_decoder_cache_continuation_matches_reference():
    baseline, owned = models()
    for model in (baseline, owned):
        model.eval()
    caches = [None, None]
    for tokens in (torch.tensor([[1, 5, 2]]), torch.tensor([[7]])):
        results = []
        with torch.no_grad():
            for index, model in enumerate((baseline, owned)):
                output = model(tokens, use_cache=True, past_key_values=caches[index])
                caches[index] = output.past_key_values
                results.append(output.last_hidden_state)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        assert bool(results[1].isfinite().all() and results[1].count_nonzero())
        assert caches[0].get_seq_length() == caches[1].get_seq_length()
