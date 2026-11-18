"""Pure path classification shared by configuration and storage."""

from __future__ import annotations

import os


def is_remote(path: str | os.PathLike[str]) -> bool:
    """Match the URI heuristic used by PyTorch DCP checkpoint paths."""
    return "://" in os.fspath(path)
