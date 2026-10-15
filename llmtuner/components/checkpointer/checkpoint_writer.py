"""DCP publication without overwriting a previously published shard generation.

The PyTorch writer uses stable shard names and removes metadata before rename.
Unique shard names plus metadata replacement let readers continue using the old
generation until the new one is published. Local metadata uses os.replace;
remote metadata requires the backend's atomic whole-object replacement.
Obsolete or failed generation files remain owned by the step directory and its
retention policy.
This does not provide concurrent-writer exclusion or power-loss durability.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import filesystem as dcp_filesystem
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import SavePlan
from torch.distributed.checkpoint.storage import WriteResult

from ...errors import UnsupportedCombinationError
from . import filesystem


class GenerationFileSystemWriter(FileSystemWriter):
    """Write fresh shards; publish local or fsspec metadata without pre-deletion."""

    def __init__(self, path: str | os.PathLike[str], **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        if filesystem.is_remote(path):
            # Keep fsspec optional on local saves, including worker imports.
            from torch.distributed.checkpoint._fsspec_filesystem import (
                FileSystem as FsspecFileSystem,
            )

            self.fs = FsspecFileSystem()
            self.path = self.fs.init_path(path)
            protocols = self.fs.fs.protocol
            protocols = {protocols} if isinstance(protocols, str) else set(protocols)
            if not protocols & {"file", "local", "memory", "s3", "s3a", "gs", "gcs"}:
                raise UnsupportedCombinationError(
                    "Native DCP generation publication requires local atomic "
                    "rename or atomic whole-object replacement; unsupported "
                    f"fsspec protocols: {sorted(protocols)}. Use a local path, "
                    "file://, memory://, s3:// or gs:// checkpoint folder."
                )

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        if not self.use_collectives:
            plan = replace(
                plan,
                storage_data=replace(
                    plan.storage_data,
                    prefix=f"{self.save_id}_{plan.storage_data.prefix}",
                ),
            )
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # The coordinator's generation is carried in each rank's SavePlan.
        # Independently generated per-rank save_ids need no extra collective.
        return [
            replace(
                plan,
                storage_data=replace(
                    plan.storage_data,
                    prefix=f"{self.save_id}_{plan.storage_data.prefix}",
                ),
            )
            for plan in super().prepare_global_plan(plans)
        ]

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        # Match PyTorch 2.14's metadata format. Build metadata before touching
        # either the local publication or the remote object.
        if hasattr(dcp_filesystem, "CURRENT_DCP_VERSION"):
            # Earlier supported PyTorch releases have no metadata version tag.
            metadata.version = dcp_filesystem.CURRENT_DCP_VERSION
        metadata.storage_data = {
            result.index: result.storage_data
            for rank_results in results
            for result in rank_results
        }
        metadata.storage_meta = self.storage_meta()
        metadata_path = (
            self._get_metadata_path(self.rank)
            if not self.use_collectives and self.rank is not None
            else self._get_metadata_path()
        )
        if filesystem.is_remote(self.path):
            remote_fs = self.fs.fs
            assert remote_fs is not None
            protocols = (
                {remote_fs.protocol}
                if isinstance(remote_fs.protocol, str)
                else set(remote_fs.protocol)
            )
            if protocols & {"file", "local"}:
                # LocalFileSystem.pipe_file truncates the destination directly;
                # file:// needs the same atomic replacement as a plain path.
                metadata_path = remote_fs._strip_protocol(metadata_path)
            else:
                # Serialization must succeed before the published object is
                # touched. Avoid rename, which may copy/delete on object stores.
                payload = pickle.dumps(metadata)
                remote_fs.pipe_file(os.fspath(metadata_path), payload)
                return
        metadata_path = Path(metadata_path)
        temporary_path = metadata_path.with_name(
            f"{metadata_path.name}.{self.save_id}.tmp"
        )
        with temporary_path.open("wb") as stream:
            pickle.dump(metadata, stream)
            if self.sync_files:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary_path, metadata_path)
