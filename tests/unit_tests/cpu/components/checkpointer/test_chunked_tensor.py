import pytest
import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.utils._python_dispatch import TorchDispatchMode

from llmtuner.components.checkpointer.chunked_tensor import ChunkedCheckpointTensor


def test_chunks_keep_noncontiguous_live_storage():
    local = torch.arange(24).reshape(2, 4, 3)
    state = ChunkedCheckpointTensor(
        (4, 8, 3), [((2, 0, 0), local[:, :2]), ((2, 4, 0), local[:, 2:])]
    )
    assert state.shape == (4, 8, 3)
    for offset, value in state.chunks:
        assert not value.is_contiguous()
        assert torch._C._is_alias_of(local, value)
        assert state.__get_tensor_shard__(MetadataIndex("weight", offset)) is value
    with pytest.raises(ValueError, match="Unknown"):
        state.__get_tensor_shard__(MetadataIndex("weight", (0, 0, 0)))


@pytest.mark.parametrize(
    "shape,chunks,message",
    [
        ((0, 4), [((0, 0), torch.ones(1, 1))], "positive"),
        ((4, 4), [((0,), torch.ones(1, 1))], "dimensions"),
        ((4, 4), [((-1, 0), torch.ones(1, 1))], "bounds"),
        ((4, 4), [((3, 0), torch.ones(2, 1))], "bounds"),
        ((4, 4), [((0, 0), torch.ones(2, 2)), ((1, 1), torch.ones(2, 2))], "overlap"),
        ((4, 4), [((0, 0), torch.empty(1, 1, device="meta"))], "materialized"),
    ],
)
def test_invalid_chunk_layout_is_rejected(shape, chunks, message):
    with pytest.raises(ValueError, match=message):
        ChunkedCheckpointTensor(shape, chunks)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"dtype": torch.float32},
        {"device": "cpu"},
    ],
)
def test_empty_chunks_require_explicit_dtype_and_device(metadata):
    with pytest.raises(ValueError, match="dtype and device"):
        ChunkedCheckpointTensor((4, 4), [], **metadata)


def test_empty_chunks_preserve_metadata_through_snapshots_and_conversions():
    source = ChunkedCheckpointTensor((4, 4), [], dtype=torch.float32, device="cpu")
    assert source.chunks == ()
    assert source.__create_chunk_list__() == []
    assert source.__create_write_items__("weight", source) == []

    for snapshot in (source.detach(), source.clone(), torch.zeros_like(source)):
        assert snapshot.shape == source.shape
        assert snapshot.dtype == torch.float32
        assert snapshot.device == torch.device("cpu")
        assert snapshot.chunks == ()

    converted = source.to(dtype=torch.float64)
    assert converted.dtype == torch.float64
    assert converted.device == torch.device("cpu")
    assert converted.chunks == ()
    allocated = source.new_empty(source.shape, dtype=torch.float64, device="cpu")
    assert allocated.dtype == torch.float64
    assert allocated.device == torch.device("cpu")
    assert allocated.chunks == ()
    assert allocated.copy_(converted) is allocated
    assert source.refresh_from(source.clone()) is source


def test_snapshot_copy_preserves_layout_and_isolates_mutations():
    local = torch.arange(12.0).reshape(2, 6)
    source = ChunkedCheckpointTensor(
        (4, 12), [((0, 0), local[:, :3]), ((0, 6), local[:, 3:])]
    )
    destination = torch.zeros_like(source)
    destination.copy_(source)
    local.fill_(-1)
    for (_, target), expected in zip(
        destination.chunks,
        (
            torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]),
            torch.tensor([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]),
        ),
        strict=True,
    ):
        torch.testing.assert_close(target, expected, atol=0, rtol=0)
        assert not torch._C._is_alias_of(local, target)
    with pytest.raises(NotImplementedError):
        source + 1


def test_copy_rejects_different_layout_before_modifying_destination():
    target_data = torch.full((2, 2), 7.0)
    target = ChunkedCheckpointTensor((4, 4), [((0, 0), target_data)])
    source = ChunkedCheckpointTensor((4, 4), [((2, 2), torch.ones(2, 2))])
    with pytest.raises(ValueError, match="identical"):
        target.copy_(source)
    assert torch.all(target_data == 7)


def test_staging_allocation_preserves_local_sizes_and_rejects_shape_changes():
    source = ChunkedCheckpointTensor((8, 8), [((4, 4), torch.ones(2, 2))])
    target = source.new_empty(source.shape, device="cpu")
    assert target.shape == source.shape
    assert target.chunks[0][0] == (4, 4)
    assert target.chunks[0][1].shape == (2, 2)
    assert not torch._C._is_alias_of(source.chunks[0][1], target.chunks[0][1])
    with pytest.raises(ValueError, match="same global shape"):
        source.new_empty((4, 4))


def test_cache_refresh_reuses_storage_for_fresh_hook_values():
    cached_data = torch.ones(2, 2)
    cached = ChunkedCheckpointTensor((4, 4), [((0, 0), cached_data)])
    for value in (3.0, 5.0):
        current = ChunkedCheckpointTensor((4, 4), [((0, 0), torch.full((2, 2), value))])
        assert cached.refresh_from(current) is cached
        assert torch.all(cached_data == value)


def test_cache_refresh_changed_live_view_detaches_without_mutation():
    live = torch.arange(4.0).reshape(2, 2)
    before = live.clone()
    cached = ChunkedCheckpointTensor((4, 4), [((0, 0), live)])
    current = ChunkedCheckpointTensor((4, 4), [((0, 0), live.t())])
    replacement = cached.refresh_from(current)
    assert replacement is not cached
    torch.testing.assert_close(live, before, atol=0, rtol=0)
    torch.testing.assert_close(replacement.chunks[0][1], before.t(), atol=0, rtol=0)
    assert not torch._C._is_alias_of(live, replacement.chunks[0][1])
    assert replacement.refresh_from(current) is replacement


def test_cache_refresh_unchanged_live_views_preserves_wrapper():
    local = torch.arange(12.0).reshape(2, 6)
    chunks = [((0, 0), local[:, :3]), ((0, 6), local[:, 3:])]
    cached = ChunkedCheckpointTensor((4, 12), chunks)
    current = ChunkedCheckpointTensor((4, 12), chunks)
    assert cached.refresh_from(current) is cached


def test_copy_handles_cross_chunk_storage_swap_without_overwriting_source():
    local = torch.arange(8.0).reshape(2, 4)
    before = local.clone()
    target = ChunkedCheckpointTensor(
        (2, 4), [((0, 0), local[:, :2]), ((0, 2), local[:, 2:])]
    )
    source = ChunkedCheckpointTensor(
        (2, 4), [((0, 0), local[:, 2:]), ((0, 2), local[:, :2])]
    )
    target.copy_(source)
    torch.testing.assert_close(local[:, :2], before[:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(local[:, 2:], before[:, :2], atol=0, rtol=0)


def test_cache_refresh_swapped_live_chunks_keeps_model_unchanged():
    local = torch.arange(8.0).reshape(2, 4)
    before = local.clone()
    cached = ChunkedCheckpointTensor(
        (2, 4), [((0, 0), local[:, :2]), ((0, 2), local[:, 2:])]
    )
    current = ChunkedCheckpointTensor(
        (2, 4), [((0, 0), local[:, 2:]), ((0, 2), local[:, :2])]
    )
    refreshed = cached.refresh_from(current)
    assert refreshed is not cached
    torch.testing.assert_close(local, before, atol=0, rtol=0)
    for (_, actual), expected in zip(
        refreshed.chunks, (before[:, 2:], before[:, :2]), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_copy_identical_live_views_does_not_allocate_or_write():
    class RejectCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.clone.default, torch.ops.aten.copy_.default):
                if not isinstance(args[0], ChunkedCheckpointTensor):
                    raise AssertionError(f"Redundant local operation: {func}")
            return func(*args, **(kwargs or {}))

    local = torch.arange(12.0).reshape(2, 6)
    chunks = [((0, 0), local[:, :3]), ((0, 6), local[:, 3:])]
    source = ChunkedCheckpointTensor((4, 12), chunks)
    destination = ChunkedCheckpointTensor((4, 12), chunks)
    with RejectCopies():
        assert destination.copy_(source) is destination
