"""Real collator/wrapper/HF mask propagation; CPU math, never NPU fallback."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM

from llmtuner.datasets.collators import TextCollator
from llmtuner.datasets.contracts import TextSequence
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.qwen.qwen3_5.gdn_numerics import _validate_forward
from llmtuner.models.qwen.qwen3_5.linear_attention import validate_gdn_inputs
from llmtuner.models.qwen.qwen3_5.padding_metadata import validate_gdn_padding_metadata


def collated(valid=93):
    context = SimpleNamespace(num_tokens_per_batch=128, max_context_length=128)
    row = TextSequence(
        input_ids=(torch.arange(valid) % 29 + 1).numpy(),
        labels=(torch.arange(valid) % 29 + 1).numpy(),
    )
    return TextCollator(context=context)([row])


def actual_wrapper():
    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
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
    native = AutoModelForCausalLM.from_config(config).float()
    # Use the real wrapper forward with a real HF decoder and parameters.
    # Production NPU configuration/geometry remains separately strict; this
    # CPU test exercises mask semantics, without replacing any tensor math.
    wrapper = HFTransformerModel.__new__(HFTransformerModel)
    nn.Module.__init__(wrapper)
    wrapper.model = native
    wrapper.cp_mesh = None
    wrapper._cp_strategy = "kv_allgather"
    object.__setattr__(wrapper, "_decoder", native.model)
    wrapper._embed_name = "embed_tokens"
    native.config.gdn_allow_right_padding = True
    native.config.attn_mask_type = "causal"
    return wrapper


def test_real_collator_restart_only_allowed_in_proven_padding():
    batch = collated()
    positions = batch["positions"]
    assert positions[93:].tolist() == list(range(35))
    original = positions.clone()
    validate_gdn_padding_metadata(
        batch_size=1,
        sequence_length=128,
        positions=positions,
        padding_mask=batch["padding_mask"],
        allow_right_padding=True,
    )
    assert torch.equal(positions, original)
    for mask in (None, torch.zeros(128, dtype=torch.bool)):
        with pytest.raises(UnsupportedCombinationError, match="contiguous"):
            validate_gdn_padding_metadata(
                batch_size=1,
                sequence_length=128,
                positions=positions,
                padding_mask=mask,
                allow_right_padding=True,
            )
    positions[50] = 0
    with pytest.raises(UnsupportedCombinationError, match="valid prefix"):
        validate_gdn_padding_metadata(
            batch_size=1,
            sequence_length=128,
            positions=positions,
            padding_mask=batch["padding_mask"],
            allow_right_padding=True,
        )


@pytest.mark.parametrize("full_ac", [False, True])
def test_actual_wrapper_hf_mask_and_valid_prefix_vjp(full_ac):
    torch.manual_seed(71)
    wrapper = actual_wrapper()
    batch = collated()
    records = []

    def capture(module, args, kwargs):
        mask = kwargs["attention_mask"]
        records.append(None if mask is None else mask.detach().clone())

    handle = wrapper.model.model.layers[0].linear_attn.register_forward_pre_hook(
        capture,
        with_kwargs=True,
    )
    original_ids, original_positions = (
        batch["input"].clone(),
        batch["positions"].clone(),
    )
    parameters = tuple(wrapper.parameters())

    def run(tokens, positions, padding=None):
        def forward():
            return wrapper(tokens, positions=positions, padding_mask=padding)

        return checkpoint(forward, use_reentrant=False) if full_ac else forward()

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


@pytest.mark.parametrize("guard", [_validate_forward, validate_gdn_inputs])
def test_module_guard_retains_validity_and_strict_cache_positions(guard):
    batch = collated()
    module = SimpleNamespace(_llmtuner_gdn_allow_right_padding=True)
    kwargs = {
        "hidden_states": torch.empty(1, 128, 2048, dtype=torch.bfloat16),
        "position_ids": batch["positions"][None, :],
        "attention_mask": (~batch["padding_mask"])[None, :],
    }
    # CPU rejection remains deliberate; no NPU arithmetic is mocked.
    with pytest.raises(UnsupportedCombinationError, match="BF16 NPU"):
        guard(module, (), kwargs)
    kwargs["position_ids"][0, 50] = 0
    if guard is validate_gdn_inputs:
        with pytest.raises(UnsupportedCombinationError, match="valid prefix"):
            guard(module, (), kwargs)
    kwargs["position_ids"] = batch["positions"][None, :].clone()
    kwargs["cache_position"] = torch.cat((torch.arange(93), torch.arange(35)))
    if guard is validate_gdn_inputs:
        with pytest.raises(UnsupportedCombinationError, match="contiguous"):
            guard(module, (), kwargs)
