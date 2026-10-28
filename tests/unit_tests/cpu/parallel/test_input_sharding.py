"""Token alignment through actual PyTorch CP and TP input partitions.

The fake process group supplies mesh coordinates; no collective or attention
kernel is validated by these local DTensor partitions.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from llmtuner.parallel.context_parallel import input_sharding
from llmtuner.parallel.context_parallel.input_sharding import (
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("load_balancer", [None, "headtail"])
def test_cp_then_tp_preserves_token_fields(rank, load_balancer):
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        cp_rank, tp_rank = divmod(rank, 2)
        tokens = torch.arange(16)
        labels = tokens + 100
        positions = tokens % 5
        padding = tokens % 3 == 1
        original = [x.clone() for x in (tokens, labels, positions, padding)]
        if load_balancer is None:
            cp_indices = torch.arange(cp_rank * 8, (cp_rank + 1) * 8)
        else:
            # Four chunks: each CP rank owns its head and opposite tail.
            cp_indices = torch.cat(
                [
                    torch.arange(cp_rank * 4, (cp_rank + 1) * 4),
                    torch.arange((3 - cp_rank) * 4, (4 - cp_rank) * 4),
                ]
            )
        tokens_cp, labels_cp, positions_cp = shard_batch_for_cp(
            tokens, labels, positions, mesh["cp"], load_balancer
        )
        padding_cp = shard_padding_mask_for_cp(padding, mesh["cp"], load_balancer)
        for actual, source in zip(
            (tokens_cp, labels_cp, positions_cp, padding_cp), original, strict=True
        ):
            torch.testing.assert_close(actual, source[cp_indices])
        tokens_tp, labels_tp = shard_batch_for_tp(tokens_cp, labels_cp, mesh["tp"])
        padding_tp = shard_padding_mask_for_tp(padding_cp, mesh["tp"])
        final_indices = cp_indices[tp_rank * 4 : (tp_rank + 1) * 4]
        for actual, source in zip(
            (tokens_tp, labels_tp, padding_tp),
            (tokens, labels, padding),
            strict=True,
        ):
            torch.testing.assert_close(actual, source[final_indices])
        # RoPE consumes the CP sequence after projection gathers the TP slices.
        torch.testing.assert_close(positions_cp, positions[cp_indices])
        for source, snapshot in zip(
            (tokens, labels, positions, padding), original, strict=True
        ):
            torch.testing.assert_close(source, snapshot)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("length", (1, 3, 15))
def test_cp_rejects_uneven_sequence_before_sharding(rank, length):
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("cp",))
        tokens = torch.arange(length)
        with pytest.raises(ValueError, match="not divisible by cp=2"):
            shard_batch_for_cp(tokens, tokens + 1, tokens, mesh, None)
        with pytest.raises(ValueError, match="not divisible by cp=2"):
            shard_padding_mask_for_cp(tokens == 0, mesh, None)
    finally:
        dist.destroy_process_group()


class _FakeMesh:
    """Minimal mesh: just the two queries the sharding paths make of it."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def test_missing_shard_helper_falls_back_to_public_chunk(monkeypatch):
    """torch < 2.12 has no private helper; the contiguous cut is reproduced.

    Every rank keeps ``T / mesh_size`` consecutive tokens, in mesh rank order,
    which is the order the fused TP collectives reassemble the sequence in.
    """
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", None)
    ids = torch.arange(8)
    for rank in (0, 1):
        ids_shard, labels_shard = input_sharding.shard_batch_for_tp(
            ids, ids + 10, _FakeMesh(2, rank)
        )
        expected = torch.arange(rank * 4, (rank + 1) * 4)
        torch.testing.assert_close(ids_shard, expected)
        torch.testing.assert_close(labels_shard, expected + 10)


def test_missing_shard_helper_rejects_load_balanced_partition(monkeypatch):
    """A load balancer rearranges the sequence and has no public equivalent."""
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", None)
    with pytest.raises(ImportError, match="TP/CP input sharding requires"):
        input_sharding._shard_buffers(
            mesh=_FakeMesh(2, 0),
            buffers=[torch.arange(8)],
            seq_dims=[0],
            load_balancer=object(),
        )


def test_contiguous_fallback_rejects_an_indivisible_sequence(monkeypatch):
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", None)
    with pytest.raises(ValueError, match="not divisible"):
        input_sharding.shard_batch_for_cp(
            torch.arange(7), torch.arange(7), torch.arange(7), _FakeMesh(2, 0)
        )
