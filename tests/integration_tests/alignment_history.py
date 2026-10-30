"""Disk-backed observations for multi-step numerical alignment.

Each item is a trusted local torch artifact. Legacy list captures remain readable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


class AlignmentHistory(Sequence):
    """Keep only one observation in memory when iterating a saved capture."""

    def __init__(self, root: Path, *, writable: bool = False):
        self.root = root
        self.writable = writable
        if writable:
            root.mkdir(parents=True, exist_ok=False)
            self.count = 0
        else:
            self.count = json.loads((root / "manifest.json").read_text())["count"]

    def __len__(self) -> int:
        return self.count

    def _index(self, index: int) -> int:
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.count))]
        index = self._index(index)
        return torch.load(
            self.root / f"step-{index + 1}.pt", map_location="cpu", weights_only=False
        )

    def __setitem__(self, index: int, value: Any) -> None:
        if not self.writable:
            raise TypeError("Alignment history is read-only")
        torch.save(value, self.root / f"step-{self._index(index) + 1}.pt")

    def append(self, value: Any) -> None:
        if not self.writable:
            raise TypeError("Alignment history is read-only")
        torch.save(value, self.root / f"step-{self.count + 1}.pt")
        self.count += 1
        (self.root / "manifest.json").write_text(
            json.dumps({"count": self.count}) + "\n"
        )


def load_alignment_history(root: Path, name: str):
    directory = root / name
    if directory.is_dir():
        return AlignmentHistory(directory)
    return torch.load(root / f"{name}.pt", map_location="cpu", weights_only=False)


def capture_artifact_path(shared_root: Path, rank_root: Path, name: str) -> Path:
    """Resolve shared artifacts while retaining compatibility with old captures."""
    shared_path = shared_root / name
    return shared_path if shared_path.is_file() else rank_root / name


def save_shared_snapshot(
    value: Any, path: Path, *, validate: Callable[[Any], None] | None = None
) -> None:
    """Publish on rank zero and propagate write/validation failures to every rank.

    Build ``value`` on every rank before calling: gathering DTensors may itself
    require collectives. This function only coordinates the shared file write.
    """
    distributed = dist.is_initialized()
    error = None
    if not distributed or dist.get_rank() == 0:
        try:
            if validate is not None:
                validate(value)
            temporary_path = path.with_name(f".{path.name}.tmp")
            torch.save(value, temporary_path)
            temporary_path.replace(path)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    if distributed:
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error is not None:
        raise RuntimeError(f"Failed to publish shared snapshot {path}: {error}")
