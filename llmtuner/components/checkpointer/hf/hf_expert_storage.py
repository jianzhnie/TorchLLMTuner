"""Read individual HF experts into fused tensors without gathering model weights."""

from __future__ import annotations

import re
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

from .hf_storage import IndexedHuggingFaceStorageReader


def grouped_expert_key(key: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(
        r"(.+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
    )
    if match is None:
        return None
    prefix, expert, projection = match.groups()
    target = "down_proj" if projection == "down_proj" else "gate_up_proj"
    return prefix + target, int(expert), projection


def native_ep_expert_key(key: str) -> str:
    """Map an individual HF projection to its native EP stack FQN."""
    mapped = grouped_expert_key(key)
    if mapped is None:
        return key
    projection = mapped[2]
    role = {"gate_proj": "w1_EFD", "up_proj": "w3_EFD", "down_proj": "w2_EDF"}[
        projection
    ]
    return key.rsplit("experts.", 1)[0] + "routed_experts.inner_experts." + role


@dataclass
class _ExpertStorageInfo(torch_hf_storage._HFStorageInfo):
    source_fqn: str


class GroupedExpertHuggingFaceStorageReader(IndexedHuggingFaceStorageReader):
    """Expose each expert projection as a chunk of a fused checkpoint tensor."""

    def __init__(
        self,
        path: str,
        *,
        num_experts: int,
        thread_count: int = 2,
        native_ep_layout: bool = False,
    ) -> None:
        super().__init__(path, thread_count=thread_count)
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.native_ep_layout = native_ep_layout

    def read_metadata(self) -> Metadata:
        metadata = super().read_metadata()
        if self.native_ep_layout:
            unsupported = [
                key
                for key in metadata.state_dict_metadata
                if key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
            ]
            if unsupported:
                raise ValueError(
                    "Native EP HF reader requires individual expert projections; "
                    f"fused expert tensors are unsupported: {unsupported[:4]}"
                )
        groups: dict[str, dict[tuple[int, str], tuple[str, Any]]] = {}
        for key, value in list(metadata.state_dict_metadata.items()):
            mapped = grouped_expert_key(key)
            if mapped is None:
                continue
            target, expert, projection = mapped
            if self.native_ep_layout:
                target = native_ep_expert_key(key)
            sources = groups.setdefault(target, {})
            if (expert, projection) in sources:
                raise ValueError(f"Duplicate individual HF expert index: {key}")
            sources[expert, projection] = (key, value)
        for target, sources in groups.items():
            if target in metadata.state_dict_metadata:
                raise ValueError(f"Mixed fused and individual HF experts: {target}")
            projections = (
                (
                    {"w1_EFD": "gate_proj", "w3_EFD": "up_proj", "w2_EDF": "down_proj"}[
                        target.rsplit(".", 1)[1]
                    ],
                )
                if self.native_ep_layout
                else ("down_proj",)
                if target.endswith(".down_proj")
                else ("gate_proj", "up_proj")
            )
            expected = {
                (expert, projection)
                for expert in range(self.num_experts)
                for projection in projections
            }
            if set(sources) != expected:
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
                offset = torch.Size(
                    (
                        expert,
                        shape[0]
                        if projection == "up_proj" and not self.native_ep_layout
                        else 0,
                        0,
                    )
                )
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
                    (self.num_experts, shape[0] * len(projections), shape[1])
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
