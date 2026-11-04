"""Attention geometry diagnostics preserve gradients and reject changed masks."""

import pytest
import torch
import torch.nn.functional as F

from tests.integration_tests.alignment_attention_control import (
    batched_attention_control,
)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_attention_geometry_control_matches_batched_forward_and_backward(mask_dtype):
    generator = torch.Generator().manual_seed(413)
    original = [
        torch.randn(
            2, heads, 5, 4, generator=generator, dtype=torch.float64
        ).requires_grad_()
        for heads in (4, 2, 2)
    ]
    packed = [
        tensor.detach()
        .transpose(1, 2)
        .reshape(1, 10, tensor.shape[1], 4)
        .transpose(1, 2)
        .requires_grad_()
        for tensor in original
    ]
    indices = torch.arange(10)
    mask = (indices[:, None] >= indices[None, :]) & (
        indices[:, None] // 5 == indices[None, :] // 5
    )
    if mask_dtype != torch.bool:
        mask = torch.zeros(mask.shape, dtype=mask_dtype).masked_fill(~mask, -torch.inf)
    actual, _ = batched_attention_control(*packed, mask[None, None], (2, 5))
    expected = (
        F.scaled_dot_product_attention(*original, is_causal=True, enable_gqa=True)
        .transpose(1, 2)
        .reshape_as(actual)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    actual.backward(gradient)
    expected.backward(gradient)
    for reference, candidate in zip(original, packed, strict=True):
        restored = candidate.grad.transpose(1, 2).reshape(2, 5, reference.shape[1], 4)
        torch.testing.assert_close(
            restored.transpose(1, 2), reference.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("mask", [None, torch.ones(10, 10), torch.ones(10, 10).bool()])
def test_attention_geometry_control_rejects_missing_or_different_mask(mask):
    value = torch.ones(1, 2, 10, 4)
    with pytest.raises(ValueError, match="mask"):
        batched_attention_control(value, value, value, mask, (2, 5))
