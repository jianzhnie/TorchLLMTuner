"""Checkpoint path-operation contract and filesystem implementation.

This module stays independent of checkpoint managers and distributed engines.
Filesystem operations use local paths or lazily resolved fsspec URIs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from . import filesystem

__all__ = ["CheckpointStorage", "FilesystemCheckpointStorage"]


@runtime_checkable
class CheckpointStorage(Protocol):
    """The path operations a checkpoint manager needs from its storage.

    Managers differ in how they read and write checkpoint bytes, but they ask
    the same handful of questions about paths: is this a checkpoint directory,
    did this metadata file land, which steps are on disk, delete this one. This
    protocol is the whole of that surface, so policies like retention and
    latest-step discovery can live on ``BaseCheckpointManager`` without knowing
    which backend answers them.

    Paths are ``str`` rather than ``Path`` because a checkpoint id may be a
    remote URI (``gs://...``) that ``Path`` would mangle -- it collapses the
    double slash. Carrying ``str`` keeps the vocabulary lossless; whether a
    given implementation can actually reach a remote URI is up to that
    implementation, which should reject what it cannot address rather than
    silently rewrite it.

    ``runtime_checkable`` so implementations can assert conformance in their
    tests. It only checks that the method names exist, which is enough to catch
    a rename that would otherwise surface as an ``AttributeError`` mid-save.
    """

    def isdir(self, path: str) -> bool:
        """Whether ``path`` is an existing directory."""
        ...

    def isfile(self, path: str) -> bool:
        """Whether ``path`` is an existing entry that is not a directory."""
        ...

    def listdir(self, path: str) -> list[str]:
        """The entry names directly under the directory ``path``."""
        ...

    def remove(self, path: str) -> None:
        """Recursively delete the directory ``path``."""
        ...


class FilesystemCheckpointStorage:
    """``CheckpointStorage`` backed by ``checkpointer/filesystem``.

    Local paths go through ``os``/``shutil`` and remote fsspec URIs through
    fsspec, so DCP keeps reading and writing remote checkpoint folders exactly as
    it does when the paths are local. The storage does not itself reject a remote
    URI it cannot address -- ``Config.__post_init__`` already refuses the
    combinations that would need to.
    """

    def isdir(self, path: str) -> bool:
        return filesystem.isdir(path)

    def isfile(self, path: str) -> bool:
        return filesystem.isfile(path)

    def listdir(self, path: str) -> list[str]:
        return filesystem.listdir(path)

    def remove(self, path: str) -> None:
        filesystem.rmtree(path)
