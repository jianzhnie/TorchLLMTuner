"""Dense additive attention masks for causal and packed-document inputs."""

from __future__ import annotations

import torch

from llmtuner.models.common.attention import build_dense_attention_mask

# Three packed documents, lengths 4/3/4.
POSITIONS = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3])


def test_the_dense_mask_is_a_4d_additive_tensor() -> None:
    """DSA's modeling code calls ``.dim()`` on the mask and adds it to the
    scores, so this has to be a plain ``[1, 1, T, T]`` tensor of 0.0/-inf --
    a BlockMask is not an option there."""
    mask = build_dense_attention_mask(
        POSITIONS, dtype=torch.float32, block_causal=False
    )

    assert mask.shape == (1, 1, POSITIONS.numel(), POSITIONS.numel())
    assert mask.dtype is torch.float32
    assert set(mask.unique().tolist()) <= {0.0, float("-inf")}


def test_the_dense_mask_keeps_the_requested_dtype() -> None:
    """The mask is added to the attention scores, so building it in the model's
    dtype keeps that addition free."""
    mask = build_dense_attention_mask(
        POSITIONS, dtype=torch.bfloat16, block_causal=True
    )

    assert mask.dtype is torch.bfloat16


def test_dense_causal_mask_blocks_future_tokens() -> None:
    mask = build_dense_attention_mask(
        torch.arange(3), dtype=torch.float32, block_causal=False
    )
    expected = torch.tensor(
        [[0.0, -torch.inf, -torch.inf], [0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(mask[0, 0], expected, rtol=0, atol=0)


def test_dense_block_causal_mask_blocks_other_documents() -> None:
    mask = build_dense_attention_mask(
        torch.tensor([0, 1, 0, 1]), dtype=torch.float32, block_causal=True
    )
    expected = torch.tensor(
        [
            [0.0, -torch.inf, -torch.inf, -torch.inf],
            [0.0, 0.0, -torch.inf, -torch.inf],
            [-torch.inf, -torch.inf, 0.0, -torch.inf],
            [-torch.inf, -torch.inf, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(mask[0, 0], expected, rtol=0, atol=0)
