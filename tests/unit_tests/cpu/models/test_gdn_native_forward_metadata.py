"""Explicit native GDN metadata and causal-execution contracts."""

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
from tests.unit_tests.cpu.models.test_qwen35_owned_decoder import models


@pytest.mark.parametrize("causal", [None, True])
@pytest.mark.parametrize("masked", [False, True])
def test_native_decoder_metadata_matches_reference_during_real_execution(
    causal, masked
):
    baseline, owned = models()
    captures = [{}, {}]
    handles = []
    for index, model in enumerate((baseline, owned)):
        if causal is not None:
            model.config.is_causal = causal

        def capture(module, args, kwargs, index=index):
            captures[index].update(kwargs)

        handles.append(
            model.layers[0].linear_attn.register_forward_pre_hook(
                capture, with_kwargs=True
            )
        )
    tokens = torch.tensor([[1, 5, 2, 7], [3, 6, 4, 8]])
    keep = (
        torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool) if masked else None
    )
    try:
        expected = baseline(
            tokens, attention_mask=keep, use_cache=False, is_causal=causal
        ).last_hidden_state
        actual = owned(
            tokens, attention_mask=keep, use_cache=False, is_causal=causal
        ).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    for captured in captures:
        assert {"hidden_states", "cache_params", "attention_mask"} <= captured.keys()
        assert captured["cache_params"] is None
    for field in ("hidden_states", "attention_mask"):
        left, right = captures[0][field], captures[1][field]
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    references = dict(baseline.named_parameters())
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad, references[name].grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("model_type", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
@pytest.mark.parametrize("causal", [False, 1, "false"])
def test_noncausal_config_rejects_before_allocation(model_type, causal, monkeypatch):
    config = text_config(Qwen3_5MoeTextConfig)
    config.is_causal = causal
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        nn.Embedding, "__init__", lambda *a, **kw: pytest.fail("invalid GDN allocated")
    )
    with pytest.raises(UnsupportedCombinationError, match="requires causal execution"):
        model_type(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("model_type", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
@pytest.mark.parametrize("causal", [False, 1, "false"])
def test_noncausal_forward_rejects_before_embedding(model_type, causal):
    model = model_type(text_config(Qwen3_5MoeTextConfig))

    def forbidden(*args):
        pytest.fail("Invalid GDN causal metadata reached embedding")

    handle = model.get_input_embeddings().register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(
            UnsupportedCombinationError, match="requires causal execution"
        ):
            model(torch.tensor([[1, 3, 7, 2]]), is_causal=causal, use_cache=False)
    finally:
        handle.remove()
