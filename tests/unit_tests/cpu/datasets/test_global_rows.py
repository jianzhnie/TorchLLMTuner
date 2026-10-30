"""Global row ownership and independent iterator recovery contracts."""

from copy import deepcopy

import grain.python as grain
import pytest

from llmtuner.datasets.packing.global_rows import GlobalRowsIterDataset


def rows(rank, world, size=24):
    return GlobalRowsIterDataset(
        grain.MapDataset.source(list(range(size))).to_iter_dataset(),
        global_rows=8,
        dp_rank=rank,
        dp_world_size=world,
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_global_updates_have_identical_rows_across_dp_degrees(world):
    iterators = [iter(rows(rank, world)) for rank in range(world)]
    try:
        for update in range(3):
            actual = [
                next(iterator) for iterator in iterators for _ in range(8 // world)
            ]
            assert actual == list(range(update * 8, (update + 1) * 8))
    finally:
        for iterator in iterators:
            iterator.close()


def test_recovery_inside_local_batch_and_global_tail():
    original = iter(rows(1, 2, size=19))
    restored = iter(rows(1, 2, size=19))
    try:
        assert next(original) == 4
        restored.set_state(original.get_state())
        assert list(original) == list(restored) == [5, 6, 7, 12, 13, 14, 15]
    finally:
        original.close()
        restored.close()


def test_failed_replay_preserves_live_cursor():
    iterator = iter(rows(0, 2, size=19))
    try:
        assert next(iterator) == 0
        valid = iterator.get_state()
        exhausted = iter(rows(0, 2, size=19))
        try:
            list(exhausted)
            invalid = deepcopy(valid)
            invalid["parent"] = exhausted.get_state()["parent"]
        finally:
            exhausted.close()
        with pytest.raises(StopIteration):
            iterator.set_state(invalid)
        assert iterator.get_state() == valid
        assert next(iterator) == 1
    finally:
        iterator.close()
