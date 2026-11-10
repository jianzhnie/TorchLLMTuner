"""Portable Qwen adapters preserve sequence/channel layout and causal math."""

import pytest
import torch
from torch.nn import functional as F

from llmtuner.models.qwen.ops import (
    causal_conv1d,
    causal_conv1d_update,
    chunk_gdn,
    fused_recurrent_gdn,
    permute,
    unpermute,
)


@pytest.mark.parametrize("activation", [None, "silu", "swish"])
def test_causal_convolution_matches_explicit_windows_and_gradients(activation):
    torch.manual_seed(73)
    inputs = torch.randn(2, 7, 5, requires_grad=True)
    weight = torch.randn(3, 5, requires_grad=True)
    bias = torch.randn(5, requires_grad=True)
    output, state = causal_conv1d(inputs, weight, bias, activation)
    padded = F.pad(inputs, (0, 0, 2, 0))
    expected = torch.stack(
        [(padded[:, step : step + 3] * weight).sum(1) + bias for step in range(7)],
        dim=1,
    )
    if activation in ("silu", "swish"):
        expected = F.silu(expected)
    assert state is None
    assert output.shape == inputs.shape
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    actual_gradients = torch.autograd.grad(
        output.square().sum(), (inputs, weight, bias)
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (inputs, weight, bias)
    )
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        assert actual.count_nonzero() > 0
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-5)


def test_packed_convolution_rejects_unimplemented_document_boundaries():
    with pytest.raises(ValueError, match="document boundaries"):
        causal_conv1d(
            torch.ones(1, 7, 5),
            torch.ones(3, 5),
            cu_seqlens=torch.tensor([0, 3, 7]),
        )


def test_expert_combine_keeps_scores_aligned_with_sorted_rows():
    tokens = torch.tensor([[1.0, 3.0], [5.0, 7.0], [11.0, 13.0]], requires_grad=True)
    experts = torch.tensor([[2, 0], [1, 2], [0, 1]])
    scores = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]], requires_grad=True)
    routed, row_ids = permute(tokens, experts)
    gains = (experts.reshape(-1)[row_ids] + 1).unsqueeze(1)
    actual = unpermute(routed * gains, row_ids, probs=scores)
    expected = tokens * (scores * (experts + 1)).sum(1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (tokens, scores))
    expected_grads = torch.autograd.grad(expected.sum(), (tokens, scores))
    for value, reference in zip(actual_grads, expected_grads, strict=True):
        assert value.count_nonzero() == value.numel()
        torch.testing.assert_close(value, reference)


def test_unweighted_combine_uses_restore_geometry():
    tokens = torch.tensor([[1.0, 3.0], [5.0, 7.0]], requires_grad=True)
    experts = torch.tensor([[2, 0], [0, 1]])
    routed, rows = permute(tokens, experts)
    output = unpermute(routed, rows, restore_shape=tokens.shape)
    assert torch.equal(output, tokens * 2)
    output.sum().backward()
    assert torch.equal(tokens.grad, torch.full_like(tokens, 2))


def test_unweighted_empty_combine_preserves_shape():
    result = unpermute(
        torch.empty(0, 5), torch.empty(0, dtype=torch.long), restore_shape=(0, 5)
    )
    assert result.shape == (0, 5)


def test_convolution_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        causal_conv1d(torch.ones(1, 7, 5), torch.ones(3, 5), activation="gelu")


def test_incremental_convolution_matches_full_sequence_and_updates_state():
    torch.manual_seed(32)
    inputs = torch.randn(2, 7, 5)
    weight = torch.randn(3, 5)
    bias = torch.randn(5)
    expected, _ = causal_conv1d(inputs, weight, bias, activation="silu")
    state = torch.zeros(2, 5, 3)
    outputs = []
    for step in range(inputs.shape[1]):
        output, _ = causal_conv1d_update(
            inputs[:, step : step + 1], state, weight.T, bias, activation="silu"
        )
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    assert torch.equal(state, inputs[:, -3:].transpose(1, 2))


@pytest.mark.parametrize("operator", [chunk_gdn, fused_recurrent_gdn])
@pytest.mark.parametrize("backend", ["auto", "npu", "invalid"])
def test_gdn_rejects_unimplemented_backend_instead_of_silent_fallback(
    operator, backend
):
    with pytest.raises(NotImplementedError, match="eager.*only"):
        operator(implementation=backend)


@pytest.mark.parametrize("options", [{"activation": "gelu"}, {"cu_seqlens": [0, 1]}])
def test_invalid_incremental_convolution_does_not_modify_cache(options):
    state = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    original = state.clone()
    with pytest.raises(ValueError):
        causal_conv1d_update(torch.ones(2, 1, 5), state, torch.ones(5, 3), **options)
    assert torch.equal(state, original)


@pytest.mark.parametrize("capacity", [0, 3, 5])
def test_permutation_rejects_unsupported_capacity(capacity):
    with pytest.raises(ValueError, match="capacity"):
        permute(torch.ones(2, 3), torch.tensor([[1, 0], [0, 1]]), capacity)


def test_permutation_accepts_explicit_full_capacity():
    tokens = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    indices = torch.tensor([[1, 0], [0, 1]])
    expected, expected_order = permute(tokens, indices)
    actual, actual_order = permute(tokens, indices, num_out_tokens=4)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_order, expected_order)
