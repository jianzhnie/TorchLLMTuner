"""Partition complete global batches of packed context rows across DP ranks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import grain.python as grain

from ..contracts import require_positive_integer, validate_dp_coordinates


class GlobalRowsIterDataset(grain.IterDataset):
    """Read the same global stream on each rank, assigning rows after packing.

    Incomplete global batches are dropped on every rank. The caller must use
    an identical, unsharded parent stream and group local rows per update.
    This trades redundant host preprocessing for DP-independent row ownership.
    """

    def __init__(self, parent, *, global_rows: int, dp_rank: int, dp_world_size: int):
        require_positive_integer("global_rows", global_rows)
        validate_dp_coordinates(dp_rank=dp_rank, dp_world_size=dp_world_size)
        if global_rows % dp_world_size:
            raise ValueError("global_rows must be divisible by dp_world_size")
        super().__init__(parent)
        self._geometry = (global_rows, dp_rank, dp_world_size)

    def __iter__(self):
        return _GlobalRowsIterator(iter(self._parent), self._geometry)


class _GlobalRowsIterator(grain.DatasetIterator):
    def __init__(self, parent, geometry):
        super().__init__(parent)
        self._geometry = geometry
        self._rows = []
        self._offset = 0
        self._batch_start = None

    def __next__(self):
        self._assert_not_closed()
        if self._offset == len(self._rows):
            self._rows = []
            self._offset = 0
            self._batch_start = deepcopy(self._parent.get_state())
            count, rank, world = self._geometry
            rows = []
            for _ in range(count):
                rows.append(next(self._parent))
            local = count // world
            self._rows = rows[rank * local : (rank + 1) * local]
        result = self._rows[self._offset]
        self._offset += 1
        return result

    def get_state(self) -> dict[str, Any]:
        pending = self._offset < len(self._rows)
        return {
            "geometry": self._geometry,
            "parent": deepcopy(
                self._batch_start if pending else self._parent.get_state()
            ),
            "offset": self._offset if pending else 0,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._assert_not_closed()
        if tuple(state["geometry"]) != self._geometry:
            raise ValueError("global row checkpoint geometry differs")
        offset = state["offset"]
        count, _, world = self._geometry
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not 0 <= offset < count // world
        ):
            raise ValueError("invalid global row checkpoint offset")
        previous_parent = deepcopy(self._parent.get_state())
        previous_rows, previous_offset, previous_start = (
            self._rows,
            self._offset,
            self._batch_start,
        )
        try:
            self._parent.set_state(deepcopy(state["parent"]))
            self._rows = []
            self._offset = 0
            self._batch_start = None
            for _ in range(offset):
                next(self)
        except Exception:
            self._parent.set_state(previous_parent)
            self._rows = previous_rows
            self._offset = previous_offset
            self._batch_start = previous_start
            raise
