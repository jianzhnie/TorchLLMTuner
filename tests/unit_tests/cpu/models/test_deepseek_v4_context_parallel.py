"""DeepSeek V4 CP masks preserve global causal and document coordinates."""

import torch

from llmtuner.models.deepseek.deepseek_v4.context_parallel import (
    build_raw_attention_mask,
)


def test_raw_mask_preserves_packed_documents_for_query_shard() -> None:
    positions = torch.tensor([[0, 1, 0, 1, 2]])
    query_indices = torch.tensor([2, 3])

    mask = build_raw_attention_mask(
        None,
        positions,
        query_indices,
        batch_size=1,
        dtype=torch.float32,
        sliding_window=None,
    )

    allowed = mask == 0
    assert allowed.shape == (1, 1, 2, 5)
    assert not allowed[0, 0, :, :2].any()
    assert allowed[0, 0, 0, 2]
    assert allowed[0, 0, 1, 2:4].all()
    assert not allowed[0, 0, 0, 3:].any()
    assert not allowed[0, 0, 1, 4]


def test_raw_mask_combines_sliding_window_and_validity_mask() -> None:
    positions = torch.tensor([[0, 1, 2, 3]])
    validity = torch.tensor([[True, True, False, True]])

    mask = build_raw_attention_mask(
        validity,
        positions,
        torch.tensor([3]),
        batch_size=1,
        dtype=torch.bfloat16,
        sliding_window=2,
    )

    assert (mask == 0).tolist() == [[[[False, False, False, True]]]]
