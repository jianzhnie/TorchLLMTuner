"""Input partition contracts, independent of distributed backend availability."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.parallel.context_parallel import input_sharding


def test_contiguous_sharding_does_not_require_headtail_balancer(monkeypatch):
    monkeypatch.setattr(input_sharding, "_HeadTailLoadBalancer", None)
    monkeypatch.setattr(
        input_sharding, "_context_parallel_shard", lambda **kwargs: kwargs["buffers"]
    )
    mesh = SimpleNamespace(size=lambda: 2)
    assert input_sharding.resolve_load_balancer(None, 8, mesh) is None
    ids = torch.arange(8)
    result, _ = input_sharding.shard_batch_for_tp(ids, ids, mesh)
    assert result is ids
    with pytest.raises(ImportError, match="headtail load balancing requires"):
        input_sharding.resolve_load_balancer("headtail", 8, mesh)


def test_missing_shard_helper_reports_actual_dependency(monkeypatch):
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", None)
    with pytest.raises(ImportError, match="TP/CP input sharding requires"):
        input_sharding.shard_batch_for_tp(
            torch.arange(8), torch.arange(8), SimpleNamespace(size=lambda: 2)
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_headtail_batch_and_padding_use_identical_reordering(monkeypatch, rank):
    mesh = SimpleNamespace(size=lambda: 2, device_type="cpu")
    balancers = []

    class Balancer:
        def __init__(self, seq_len, cp_size, device):
            balancers.append((seq_len, cp_size, device))
            chunks = torch.arange(seq_len).chunk(2 * cp_size)
            self.indices = torch.cat([chunks[rank], chunks[-rank - 1]])

    def shard(*, mesh, buffers, seq_dims, load_balancer):
        return [
            buffer.index_select(dim, load_balancer.indices)
            for buffer, dim in zip(buffers, seq_dims, strict=True)
        ]

    monkeypatch.setattr(input_sharding, "_HeadTailLoadBalancer", Balancer)
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", shard)
    ids = torch.arange(8)
    labels = ids + 10
    positions = ids + 20
    padding = ids >= 6
    actual = input_sharding.shard_batch_for_cp(ids, labels, positions, mesh, "headtail")
    mask = input_sharding.shard_padding_mask_for_cp(padding, mesh, "headtail")
    expected_indices = torch.tensor([0, 1, 6, 7] if rank == 0 else [2, 3, 4, 5])
    for result, source in zip(
        (*actual, mask), (ids, labels, positions, padding), strict=True
    ):
        torch.testing.assert_close(result, source[expected_indices])
    assert balancers == [(8, 2, "cpu"), (8, 2, "cpu")]


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_tokens_labels_and_padding_share_contiguous_partition(monkeypatch, rank):
    mesh = SimpleNamespace(size=lambda: 2)
    calls = []

    def shard(*, mesh, buffers, seq_dims, load_balancer):
        calls.append((seq_dims, load_balancer))
        return [
            buffer.chunk(mesh.size(), dim=dim)[rank]
            for buffer, dim in zip(buffers, seq_dims, strict=True)
        ]

    monkeypatch.setattr(input_sharding, "require_torch_input_sharding", lambda: None)
    monkeypatch.setattr(input_sharding, "_context_parallel_shard", shard)
    ids = torch.arange(8)
    labels = ids + 10
    padding = ids >= 6
    ids_shard, labels_shard = input_sharding.shard_batch_for_tp(ids, labels, mesh)
    mask_shard = input_sharding.shard_padding_mask_for_tp(padding, mesh)
    expected = slice(rank * 4, (rank + 1) * 4)
    torch.testing.assert_close(ids_shard, ids[expected])
    torch.testing.assert_close(labels_shard, labels[expected])
    torch.testing.assert_close(mask_shard, padding[expected])
    assert calls == [([0, 0], None), ([0], None)]


@pytest.mark.parametrize("field", ["batch", "padding"])
def test_tp_nondivisible_sequence_fails_before_sharding(monkeypatch, field):
    monkeypatch.setattr(input_sharding, "require_torch_input_sharding", lambda: None)
    monkeypatch.setattr(
        input_sharding,
        "_context_parallel_shard",
        lambda **kwargs: pytest.fail("invalid length reached sharding"),
    )
    mesh = SimpleNamespace(size=lambda: 2)
    with pytest.raises(ValueError, match="not divisible"):
        if field == "batch":
            input_sharding.shard_batch_for_tp(torch.arange(7), torch.arange(7), mesh)
        else:
            input_sharding.shard_padding_mask_for_tp(
                torch.zeros(7, dtype=torch.bool), mesh
            )
