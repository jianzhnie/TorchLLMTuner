from types import SimpleNamespace
from unittest.mock import patch

from tests.caps import require_env

require_env("spmd_types")

import pytest
import torch

from llmtuner.parallel.context_parallel.sdpa_attention import DualCPSdpaKernel


def _mesh(rank: int = 0):
    return SimpleNamespace(
        size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda: None
    )


@pytest.mark.parametrize(
    ("packed", "mask", "dropout", "message"),
    [
        (True, None, 0.0, "full document mask"),
        (False, torch.ones(4, 4, dtype=torch.bool), 0.0, "four-dimensional"),
        (False, torch.ones(1, 1, 4, 4), 0.0, "boolean mask"),
        (False, object(), 0.0, "boolean mask"),
        (False, None, 0.1, "dropout"),
    ],
)
def test_dual_sdpa_rejects_invalid_contract_before_collectives(
    packed, mask, dropout, message
):
    kernel = DualCPSdpaKernel(kv_mesh=_mesh(), ulysses_mesh=_mesh(), packed=packed)
    query = torch.zeros(1, 4, 4, 4)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply"
        ) as seq_to_head,
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply"
        ) as gather,
    ):
        with pytest.raises((ValueError, NotImplementedError), match=message):
            kernel(query, query, query, block_mask=mask, dropout=dropout)
    seq_to_head.assert_not_called()
    gather.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dropout": -0.1}, "dropout"),
        ({"softcap": 30.0}, "softcap"),
        ({"s_aux": torch.ones(1)}, "s_aux"),
    ],
)
def test_dual_sdpa_rejects_modifiers_before_collectives(kwargs, message) -> None:
    kernel = DualCPSdpaKernel(kv_mesh=_mesh(), ulysses_mesh=_mesh())
    query = torch.zeros(1, 4, 4, 4)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply"
        ) as seq_to_head,
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply"
        ) as gather,
    ):
        with pytest.raises(NotImplementedError, match=message):
            kernel(query, query, query, **kwargs)
    seq_to_head.assert_not_called()
    gather.assert_not_called()


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "4"])
def test_dual_sdpa_rejects_invalid_windows_before_collectives(window) -> None:
    kernel = DualCPSdpaKernel(kv_mesh=_mesh(), ulysses_mesh=_mesh())
    query = torch.zeros(1, 4, 4, 4)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply"
        ) as seq_to_head,
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply"
        ) as gather,
    ):
        with pytest.raises(ValueError, match="sliding_window.*positive integer"):
            kernel(query, query, query, sliding_window=window)
    seq_to_head.assert_not_called()
    gather.assert_not_called()


@pytest.mark.parametrize("packed", [False, True])
def test_dual_sliding_window_uses_global_kv_query_offsets(packed) -> None:
    generator = torch.Generator().manual_seed(91)
    query, key, value = [
        torch.randn(1, 2, 16, 4, generator=generator, dtype=torch.float64)
        for _ in range(3)
    ]
    mask = torch.ones(16, 16, dtype=torch.bool).tril()
    if packed:
        documents = torch.repeat_interleave(torch.arange(3), torch.tensor([5, 6, 5]))
        mask &= documents[:, None] == documents[None, :]
    mask = mask[None, None]
    indices = torch.arange(16)
    sliding = (indices[None, :] <= indices[:, None]) & (
        indices[:, None] - indices[None, :] < 4
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask & sliding[None, None]
    ).transpose(1, 2)
    unwindowed = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    ).transpose(1, 2)
    assert not torch.allclose(expected, unwindowed)

    kernel = DualCPSdpaKernel(
        kv_mesh=_mesh(rank=1), ulysses_mesh=_mesh(), packed=packed
    )
    kernel._cp_ranks = (0, 1)
    with (
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.SeqToHead.apply",
            side_effect=[query[:, :, 8:], key[:, :, 8:], value[:, :, 8:]],
        ),
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention._GatherSequence.apply",
            side_effect=[key, value],
        ),
        patch(
            "llmtuner.parallel.context_parallel.sdpa_attention.HeadToSeq.apply",
            side_effect=lambda output, group: output[:, :, 4:],
        ),
    ):
        actual = kernel(
            query[:, :, 12:],
            key[:, :, 12:],
            value[:, :, 12:],
            block_mask=mask if packed else None,
            sliding_window=4,
        )
    torch.testing.assert_close(actual, expected[:, 12:])
