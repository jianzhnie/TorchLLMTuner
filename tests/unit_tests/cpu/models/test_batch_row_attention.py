"""Explicit Qwen3 row boundaries, independent of the removed HF wrapper."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceQwen3MoeForCausalLM,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM


def make_model(*, backend="sdpa", packed=False):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        mlp_only_layers=[0, 1],
    )
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    config.attn_mask_type = "block_causal" if packed else "causal"
    return Qwen3MoeForCausalLM(config).eval()


def test_causal_batch_rows_remain_independent_after_flattening():
    model = make_model()
    reference = ReferenceQwen3MoeForCausalLM(model.config).eval()
    reference.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    inputs, _, kwargs = model.prepare_inputs(
        {"input": tokens, "labels": tokens + 1}, parallel_dims=None
    )
    assert kwargs["positions"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert torch.isneginf(kwargs["attention_masks"][..., 4:, :4]).all()
    actual = model(inputs, **kwargs).logits
    expected = reference(input_ids=tokens, use_cache=False).logits.flatten(0, 1)
    assert actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()), probe)
    expected_gradients = torch.autograd.grad(
        expected, tuple(reference.parameters()), probe
    )
    assert actual_gradients[0].count_nonzero() > 0
    for observed, target in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, target, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("axis", ["cp", "tp"])
def test_multirow_parallel_rejects_before_sharding(packed, axis, monkeypatch):
    model = make_model(packed=packed)
    topology = SimpleNamespace(
        get_optional_mesh=lambda name: object() if name == axis else None
    )

    def unexpected_sharding(*args, **kwargs):
        pytest.fail("row-aware parallel rejection must precede token sharding")

    monkeypatch.setattr(
        "llmtuner.models.common.input_preparation.shard_batch_for_cp",
        unexpected_sharding,
    )
    monkeypatch.setattr(
        "llmtuner.models.common.input_preparation.shard_batch_for_tp",
        unexpected_sharding,
    )
    tokens = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(UnsupportedCombinationError, match="row-aware mask"):
        model.prepare_inputs(
            {"input": tokens, "labels": tokens}, parallel_dims=topology
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("position_start", [0, 5])
def test_packed_rows_with_continuation_positions_remain_independent(
    backend, position_start
):
    model = make_model(backend=backend, packed=True)
    tokens = torch.tensor([[1, 2, 3], [7, 11, 13]])
    positions = torch.arange(position_start, position_start + 6).reshape(2, 3)
    original_positions = positions.clone()
    inputs, _, kwargs = model.prepare_inputs(
        {"input": tokens, "labels": tokens, "positions": positions},
        parallel_dims=None,
    )
    assert torch.equal(positions, original_positions)
    assert torch.equal(kwargs["positions"], original_positions.flatten())
    mask = kwargs["attention_masks"]
    assert torch.isneginf(mask[..., 3:, :3]).all()
    assert mask[0, 0, 5, 3] == 0 and torch.isneginf(mask[0, 0, 3, 5])
    with torch.no_grad():
        packed = model(inputs, **kwargs).logits[3:]
        independent = model(tokens[1], positions=positions[1]).logits
    assert torch.isfinite(packed).all() and packed.count_nonzero() > 0
    torch.testing.assert_close(packed, independent, rtol=1e-5, atol=1e-7)
