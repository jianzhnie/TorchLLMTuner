"""Indexed HF reads and generation-based local safetensors publication."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint import (
    _consolidate_hf_safetensors as hf_consolidation,
)
from torch.distributed.checkpoint import hf_storage as torch_hf_storage
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.utils import _DistWrapper

from . import filesystem
from .checkpoint_keys import SAFETENSORS_INDEX


class IndexedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Read indexed tensors; retain single-file HF loading without an index."""

    def __init__(self, path: str, *, thread_count: int = 2) -> None:
        if filesystem.is_remote(path):
            raise ValueError(
                f"HF safetensors reading requires a local path, got {path}"
            )
        super().__init__(path, thread_count=thread_count)

    def read_metadata(self) -> Metadata:
        index_path = Path(self.path) / SAFETENSORS_INDEX
        if not index_path.is_file():
            return super().read_metadata()
        from safetensors import safe_open
        from safetensors.torch import _getdtype

        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("HF checkpoint index requires a non-empty weight_map")
        files: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(filename, str)
                or not filename.endswith(".safetensors")
                or PurePosixPath(filename).is_absolute()
                or ".." in PurePosixPath(filename).parts
            ):
                raise ValueError(
                    f"Invalid HF checkpoint index entry: {name!r}: {filename!r}"
                )
            files.setdefault(filename, []).append(name)
        state_metadata = {}
        storage_data = {}
        for filename, names in files.items():
            shard_path = str(Path(self.path) / filename)
            with safe_open(shard_path, framework="pt") as tensors:
                extra_metadata = tensors.metadata() or {}
                sharding = json.loads(
                    extra_metadata.get(torch_hf_storage.CUSTOM_METADATA_KEY, "{}")
                )
                for name in names:
                    tensor = tensors.get_slice(name)
                    shape = torch.Size(tensor.get_shape())
                    dtype = _getdtype(tensor.get_dtype())
                    offsets = torch.Size(
                        sharding[name][torch_hf_storage.SAVED_OFFSETS_KEY]
                        if name in sharding
                        else [0] * len(shape)
                    )
                    state_metadata[name] = TensorStorageMetadata(
                        properties=TensorProperties(dtype=dtype),
                        size=torch.Size(
                            size + offset
                            for size, offset in zip(shape, offsets, strict=True)
                        ),
                        chunks=[ChunkStorageMetadata(offsets=offsets, sizes=shape)],
                    )
                    storage_data[MetadataIndex(name, offset=offsets)] = (
                        torch_hf_storage._HFStorageInfo(
                            relative_path=shard_path, shape=shape, dtype=dtype
                        )
                    )
        return Metadata(
            state_dict_metadata=state_metadata,
            storage_data=storage_data,
            storage_meta=StorageMeta(load_id=self.load_id),
        )


def save_hf_checkpoint(
    state_dict: dict[str, Any],
    checkpoint_id: str,
    fqn_to_index_mapping: dict[str, int] | None,
) -> None:
    """Consolidate across ranks, then atomically publish a local HF index.

    Each save owns an immutable generation directory. Failures leave the old
    index and its tensors readable; abandoned generations follow step retention.
    DCP's failure coordination also covers consolidation and index publication.
    """
    if filesystem.is_remote(checkpoint_id):
        raise ValueError(
            f"HF safetensors export requires a local path, got {checkpoint_id}"
        )
    distribution = _DistWrapper(None, dist.is_initialized(), 0)
    generation = distribution.broadcast_object(
        f".export-{uuid4().hex}" if distribution.is_coordinator else None
    )
    output_path = Path(checkpoint_id) / generation
    metadata = dcp.save(
        state_dict,
        storage_writer=HuggingFaceStorageWriter(
            path=output_path / "sharded",
            save_distributed=True,
            fqn_to_index_mapping=fqn_to_index_mapping,
            enable_consolidation=False,
        ),
    )
    mapping = (
        fqn_to_index_mapping
        if fqn_to_index_mapping is not None
        else dict.fromkeys(metadata.state_dict_metadata, 1)
    )

    def consolidate_local_shards() -> dict[str, hf_consolidation._OutputFileData]:
        highest_index = max(mapping.values())
        local_mapping = {
            name: hf_consolidation._gen_file_name(index, highest_index)
            for name, index in mapping.items()
            if index % distribution.get_world_size() == distribution.rank
        }
        if not local_mapping:
            return {}
        return hf_consolidation._consolidate_safetensors_files(
            input_dir=str(output_path / "sharded"),
            output_dir=str(output_path),
            fqn_to_file_mapping=local_mapping,
            num_threads=5,
        )

    def publish_index(
        results: list[dict[str, hf_consolidation._OutputFileData]],
    ) -> None:
        consolidated = {
            filename: value
            for rank_results in results
            for filename, value in rank_results.items()
        }
        hf_consolidation._write_overall_metadata_file(str(output_path), consolidated)
        with (output_path / SAFETENSORS_INDEX).open(encoding="utf-8") as stream:
            index = json.load(stream)
        index["weight_map"] = {
            name: f"{generation}/{filename}"
            for name, filename in index["weight_map"].items()
        }
        temporary_path = Path(checkpoint_id) / f"{SAFETENSORS_INDEX}.{generation}.tmp"
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(index, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, Path(checkpoint_id) / SAFETENSORS_INDEX)

    distribution.all_reduce(
        "consolidate and publish HF export", consolidate_local_shards, publish_index
    )
