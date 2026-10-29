"""The explicit eager Flex entry must retain output and gradient semantics."""

from tests.caps import require_env

require_env("flex_attention")

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from llmtuner.models.common.attention.eager_flex import eager_flex_attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
def test_eager_flex_matches_dense_attention_and_gradients(dtype, packed):
    torch.manual_seed(73)
    length = 17
    query = torch.randn(2, 4, length, 8, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 2, length, 8, dtype=dtype, requires_grad=True)
    value = torch.randn(2, 2, length, 8, dtype=dtype, requires_grad=True)

    def mask_mod(batch, head, query_index, key_index):
        allowed = query_index >= key_index
        return allowed & (query_index // 5 == key_index // 5) if packed else allowed

    mask = create_block_mask(mask_mod, 2, None, length, length, device="cpu")
    actual = eager_flex_attention(query, key, value, mask, scale=0.25)
    reference_inputs = [
        tensor.detach().clone().requires_grad_() for tensor in (query, key, value)
    ]
    reference_query, reference_key, reference_value = reference_inputs
    indices = torch.arange(length)
    allowed = indices[:, None] >= indices[None, :]
    if packed:
        allowed &= indices[:, None] // 5 == indices[None, :] // 5
    scores = reference_query.float() @ reference_key.repeat_interleave(
        2, dim=1
    ).float().transpose(-2, -1)
    expected = (scores * 0.25).masked_fill(~allowed, -torch.inf).softmax(-1)
    expected = expected.to(dtype) @ reference_value.repeat_interleave(2, dim=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), upstream)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
    tolerance = 2e-6 if dtype == torch.float32 else 0.02
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )


def test_eager_flex_rejects_invalid_inputs():
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    mask = create_block_mask(
        lambda batch, head, query, key: query >= key, 1, None, 7, 7, device="cpu"
    )
    with pytest.raises(ValueError, match="dtypes"):
        eager_flex_attention(query, key.double(), key, mask)
    with pytest.raises(ValueError, match="four-dimensional"):
        eager_flex_attention(query[0], key, key, mask)
    with pytest.raises(TypeError, match="BlockMask"):
        eager_flex_attention(query, key, key, None)
    with pytest.raises(ValueError, match="lengths"):
        eager_flex_attention(query[:, :, :3], key, key, mask)
