"""Explicit causal models reject ambiguous input spellings before execution."""

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.models.deepseek.deepseek_v4.modeling import DeepseekV4ForCausalLM
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from tests.unit_tests.cpu.models.test_family_execution_policy import kimi_config
from tests.unit_tests.cpu.models.test_owned_family_construction import small_config


def qwen_config():
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )


FAMILIES = [
    (Qwen3MoeForCausalLM, qwen_config),
    (DeepseekV4ForCausalLM, small_config),
    (KimiLinearForCausalLM, kimi_config),
]


@pytest.mark.parametrize("model_type,config_factory", FAMILIES)
@pytest.mark.parametrize(
    "alias,canonical",
    [("positions", "position_ids"), ("attention_masks", "attention_mask")],
)
def test_duplicate_input_spelling_fails_before_decoder(
    model_type, config_factory, alias, canonical, monkeypatch
):
    model = model_type(config_factory())

    def forbidden(*args, **kwargs):
        raise AssertionError("Ambiguous input reached decoder")

    monkeypatch.setattr(model.model, "forward", forbidden)
    tokens = torch.tensor([[1, 3, 2]])
    with pytest.raises(ValueError, match=f"only one of {alias} and {canonical}"):
        model(tokens, **{alias: tokens, canonical: tokens})


@pytest.mark.parametrize("model_type,config_factory", FAMILIES[:2])
@pytest.mark.parametrize("hidden_only", [False, True])
def test_canonical_mask_is_not_overwritten_by_padding_metadata(
    model_type, config_factory, hidden_only
):
    config = config_factory()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = model_type(config).eval()
    tokens = torch.tensor([[1, 3, 2]])
    # Auxiliary metadata must not replace an explicitly supplied attention mask.
    mask = torch.tensor([[1, 1, 0]])
    with torch.no_grad():
        reference = model(tokens, attention_mask=mask, skip_lm_head=hidden_only)
        candidate = model(
            tokens,
            attention_mask=mask,
            padding_mask=torch.zeros_like(tokens).bool(),
            skip_lm_head=hidden_only,
        )
    if not hidden_only:
        reference, candidate = reference.logits, candidate.logits
    assert reference.numel() > 0 and reference.abs().sum() > 0
    assert torch.equal(candidate, reference)
