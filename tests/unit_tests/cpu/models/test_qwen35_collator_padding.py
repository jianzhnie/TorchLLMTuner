"""Real collator/explicit-model mask propagation; CPU math, never NPU fallback."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.datasets.collators import TextCollator
from llmtuner.datasets.contracts import TextSequence
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM


def collated(valid=93):
    context = SimpleNamespace(num_tokens_per_batch=128, max_context_length=128)
    row = TextSequence(
        input_ids=(torch.arange(valid) % 29 + 1).numpy(),
        labels=(torch.arange(valid) % 29 + 1).numpy(),
    )
    return TextCollator(context=context)([row])


def actual_model():
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    config._attn_implementation = "sdpa"
    config.attn_mask_type = "causal"
    return Qwen3_5MoeForCausalLM(config).float()


@pytest.mark.parametrize("full_ac", [False, True])
def test_explicit_model_mask_and_valid_prefix_vjp(full_ac):
    torch.manual_seed(71)
    model = actual_model()
    batch = collated()
    records = []

    def capture(module, args, kwargs):
        mask = kwargs["attention_mask"]
        records.append(None if mask is None else mask.detach().clone())

    handle = model.model.layers[0].linear_attn.register_forward_pre_hook(
        capture,
        with_kwargs=True,
    )
    original_ids, original_positions = (
        batch["input"].clone(),
        batch["positions"].clone(),
    )
    parameters = tuple(model.parameters())

    def run(tokens, positions, padding=None):
        def forward():
            return model(tokens, positions=positions, padding_mask=padding).logits

        if not full_ac:
            return forward()
        # torch-npu changes this process-wide default even for CPU-only tests.
        # Scope the public setting to the forward and restore it immediately.
        previous = DefaultDeviceType.get_device_type()
        try:
            DefaultDeviceType.set_device_type("cpu")
            return checkpoint(forward, use_reentrant=False)
        finally:
            DefaultDeviceType.set_device_type(previous)

    try:
        padded = run(
            batch["input"],
            positions=batch["positions"],
            padding=batch["padding_mask"],
        )
        cotangent = torch.randn_like(padded[:93])
        padded_grads = torch.autograd.grad((padded[:93] * cotangent).sum(), parameters)
        changed = batch["input"].clone()
        changed[93:] = 17
        altered = run(
            changed, positions=batch["positions"], padding=batch["padding_mask"]
        )
        altered_grads = torch.autograd.grad(
            (altered[:93] * cotangent).sum(), parameters
        )
        # Separate no-padding forward also proves the per-call mask cannot leak.
        prefix = run(batch["input"][:93], positions=torch.arange(93))
        prefix_grads = torch.autograd.grad((prefix * cotangent).sum(), parameters)
        assert padded.shape == (128, model.config.vocab_size)
        assert bool(torch.isfinite(padded).all() and padded.count_nonzero())
        assert any(bool(gradient.count_nonzero()) for gradient in padded_grads)
        torch.testing.assert_close(padded[:93], altered[:93], rtol=0, atol=0)
        torch.testing.assert_close(padded[:93], prefix, rtol=1e-5, atol=1e-6)
        for a, b, c in zip(padded_grads, altered_grads, prefix_grads, strict=True):
            # Preserve the declared GDN diagnostic criterion. FP32 MoE
            # accumulation order may differ when padded token routes change.
            for other in (b, c):
                assert bool(torch.isfinite(a).all() and torch.isfinite(other).all())
                assert bool(((a - other).abs() <= 0.005 + 0.02 * a.abs()).all())
    finally:
        handle.remove()
    assert torch.equal(records[0], (~batch["padding_mask"])[None, :])
    # AC may replay each call; every retained mask remains call-specific.
    masks = [mask for mask in records if mask is not None]
    assert masks and all(torch.equal(mask, records[0]) for mask in masks)
    assert records[-1] is None
    assert torch.equal(batch["input"], original_ids)
    assert torch.equal(batch["positions"], original_positions)
