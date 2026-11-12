"""Native Qwen3.5 recurrent masks must not imply document-state resets."""

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    validate_recurrent_mask,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


def test_packed_construction_fails_without_consuming_rng():
    config = text_config(Qwen3_5MoeTextConfig)
    config.attn_mask_type = "block_causal"
    before = torch.get_rng_state().clone()
    with pytest.raises(UnsupportedCombinationError, match="recurrent-state resets"):
        Qwen3_5MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), before)


def test_packed_preparation_rejects_before_building_dense_mask(monkeypatch):
    config = text_config(Qwen3_5MoeTextConfig)
    model = Qwen3_5MoeForCausalLM(config)
    config.attn_mask_type = "block_causal"

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported packed GDN reached generic input preparation")

    monkeypatch.setattr(
        "llmtuner.models.qwen.qwen3_5_moe.model.prepare_text_inputs", forbidden
    )
    tokens = torch.tensor([1, 2, 3, 4])
    with pytest.raises(UnsupportedCombinationError, match="packed documents"):
        model.prepare_inputs(
            {
                "input": tokens,
                "labels": tokens,
                "positions": torch.tensor([0, 1, 0, 1]),
            },
            parallel_dims=None,
        )


@pytest.mark.parametrize("keyword", ["attention_masks", "attention_mask"])
@pytest.mark.parametrize("hidden_only", [False, True])
def test_pairwise_masks_reject_before_embedding(keyword, hidden_only):
    model = Qwen3_5MoeForCausalLM(text_config(Qwen3_5MoeTextConfig))

    def forbidden(*args):
        pytest.fail("Unsupported recurrent mask reached embedding execution")

    hook = model.tok_embeddings.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(UnsupportedCombinationError, match="token keep mask"):
            model(
                torch.tensor([1, 2, 3, 4]),
                skip_lm_head=hidden_only,
                **{keyword: torch.zeros(1, 1, 4, 4)},
            )
    finally:
        hook.remove()


@pytest.mark.parametrize("packed", [False, True])
def test_full_attention_only_keeps_pairwise_mask_support(packed):
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = ["full_attention"]
    config.attn_mask_type = "block_causal" if packed else "causal"
    validate_recurrent_mask(config, torch.zeros(1, 1, 4, 4))


def test_full_attention_only_packed_logits_match_separate_documents():
    config = text_config(Qwen3_5MoeTextConfig)
    config.layer_types = ["full_attention"]
    config.attn_mask_type = "block_causal"
    config._attn_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).eval()
    tokens = torch.tensor([1, 3, 7, 2])
    positions = torch.tensor([0, 1, 0, 1])
    inputs, _, extra = model.prepare_inputs(
        {"input": tokens, "labels": tokens, "positions": positions},
        parallel_dims=None,
    )
    with torch.no_grad():
        actual = model(inputs, **extra).logits
        expected = torch.cat(
            [model(doc, positions=torch.arange(2)).logits for doc in tokens.split(2)]
        )
    assert bool(torch.isfinite(actual).all() and actual.count_nonzero())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hidden_only", [False, True])
@pytest.mark.parametrize("kind", ["positions", "mask"])
def test_duplicate_input_spellings_reject_before_embedding(hidden_only, kind):
    model = Qwen3_5MoeForCausalLM(text_config(Qwen3_5MoeTextConfig))
    kwargs = (
        {"positions": torch.arange(4), "position_ids": torch.arange(4)[None]}
        if kind == "positions"
        else {
            "attention_masks": torch.ones(1, 4, dtype=torch.bool),
            "attention_mask": torch.zeros(1, 4, dtype=torch.bool),
        }
    )

    def forbidden(*args):
        pytest.fail("Ambiguous inputs reached embedding")

    handle = model.tok_embeddings.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(ValueError, match="Specify only one"):
            model(torch.tensor([1, 3, 7, 2]), skip_lm_head=hidden_only, **kwargs)
    finally:
        handle.remove()


@pytest.mark.parametrize("hidden_only", [False, True])
def test_reference_attention_mask_is_not_overwritten_by_padding_alias(hidden_only):
    model = Qwen3_5MoeForCausalLM(text_config(Qwen3_5MoeTextConfig)).eval()
    tokens = torch.tensor([[1, 3, 7, 2]])
    keep = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        expected = model(tokens, attention_mask=keep, skip_lm_head=hidden_only)
        actual = model(
            tokens,
            attention_mask=keep,
            padding_mask=torch.zeros_like(keep),
            skip_lm_head=hidden_only,
        )
    if not hidden_only:
        actual, expected = actual.logits, expected.logits
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
