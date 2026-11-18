"""Read individual HF experts into fused tensors without gathering model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint import hf_storage as torch_hf_storage
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)

from ....models.state import ExpertStorageLayout, ExpertTensorSlice
from .hf_storage import IndexedHuggingFaceStorageReader


@dataclass
class _ExpertStorageInfo(torch_hf_storage._HFStorageInfo):
    source_fqn: str


class GroupedExpertHuggingFaceStorageReader(IndexedHuggingFaceStorageReader):
    """Expose each expert projection as a chunk of a fused checkpoint tensor."""

    def __init__(
        self,
        path: str,
        *,
        layout: ExpertStorageLayout,
        thread_count: int = 2,
    ) -> None:
        super().__init__(path, thread_count=thread_count)
        if not isinstance(layout, ExpertStorageLayout):
            raise TypeError("HF expert reader requires an ExpertStorageLayout")
        self.layout = layout

    def read_metadata(self) -> Metadata:
        metadata = super().read_metadata()
        for key in metadata.state_dict_metadata:
            self.layout.validate_source_key(key)
        groups: dict[str, dict[tuple[int, int], tuple[str, Any]]] = {}
        projection_counts: dict[str, int] = {}
        for key, value in list(metadata.state_dict_metadata.items()):
            mapped = self.layout.map_source_key(key)
            if mapped is None:
                continue
            if not isinstance(mapped, ExpertTensorSlice):
                raise TypeError(
                    "Expert key rules must return ExpertTensorSlice or None"
                )
            target, expert, projection = (
                mapped.target,
                mapped.expert_index,
                mapped.projection_index,
            )
            if expert >= self.layout.num_experts:
                raise ValueError(f"HF expert index exceeds declared count: {key}")
            if (
                projection_counts.setdefault(target, mapped.projection_count)
                != mapped.projection_count
            ):
                raise ValueError(f"Conflicting projection slot counts for {target}")
            sources = groups.setdefault(target, {})
            if (expert, projection) in sources:
                raise ValueError(f"Duplicate individual HF expert index: {key}")
            sources[expert, projection] = (key, value)
        for target, sources in groups.items():
            if target in metadata.state_dict_metadata:
                raise ValueError(f"Mixed fused and individual HF experts: {target}")
            projection_count = projection_counts[target]
            # Indices are in range and unique, so the count proves coverage.
            if len(sources) != self.layout.num_experts * projection_count:
                raise ValueError(f"Incomplete individual HF experts for {target}")
            first = next(iter(sources.values()))[1]
            shape = first.size
            chunks = []
            for (expert, projection), (key, value) in sorted(sources.items()):
                if (
                    len(value.size) != 2
                    or value.size != shape
                    or value.properties.dtype != first.properties.dtype
                    or len(value.chunks) != 1
                    or any(value.chunks[0].offsets)
                    or value.chunks[0].sizes != shape
                ):
                    raise ValueError(f"Invalid individual HF expert tensor: {key}")
                old_index = MetadataIndex(key, offset=value.chunks[0].offsets)
                storage = metadata.storage_data.pop(old_index)
                offset = torch.Size((expert, projection * shape[0], 0))
                chunk_shape = torch.Size((1, *shape))
                chunks.append(ChunkStorageMetadata(offsets=offset, sizes=chunk_shape))
                metadata.storage_data[MetadataIndex(target, offset=offset)] = (
                    _ExpertStorageInfo(
                        relative_path=storage.relative_path,
                        shape=chunk_shape,
                        dtype=storage.dtype,
                        source_fqn=key,
                    )
                )
                del metadata.state_dict_metadata[key]
            metadata.state_dict_metadata[target] = TensorStorageMetadata(
                properties=first.properties,
                size=torch.Size(
                    (self.layout.num_experts, shape[0] * projection_count, shape[1])
                ),
                chunks=chunks,
            )
        return metadata

    def _process_read_request(self, tensors, request, planner) -> None:
        storage = self.storage_data[request.storage_index]
        if not isinstance(storage, _ExpertStorageInfo):
            return super()._process_read_request(tensors, request, planner)
        if request.storage_offsets[0] != 0 or request.lengths[0] != 1:
            raise ValueError("Individual expert reads must span exactly one expert")
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(
                request.storage_offsets[1:], request.lengths[1:], strict=True
            )
        )
        tensor = tensors.get_slice(storage.source_fqn)[slices].unsqueeze(0)
        destination = planner.resolve_tensor(request).detach()
        if destination.shape != tensor.shape:
            raise ValueError(f"HF expert read shape mismatch: {request.storage_index}")
        destination.copy_(tensor)
        planner.commit_tensor(request, destination)
