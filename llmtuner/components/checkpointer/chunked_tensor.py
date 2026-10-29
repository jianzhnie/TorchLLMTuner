"""Global checkpoint metadata backed by disjoint local tensor views."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


class ChunkedCheckpointTensor(torch.Tensor):
    """A DCP-only tensor; its global shape does not allocate global storage.

    Local chunks may be non-contiguous aliases. Only metadata, snapshot copying
    and device conversion are supported; this object must not enter training
    computations. Empty ownership carries explicit dtype and device metadata.
    """

    @staticmethod
    def __new__(
        cls,
        shape: Sequence[int],
        chunks: Sequence[tuple[Sequence[int], torch.Tensor]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
    ) -> ChunkedCheckpointTensor:
        shape = torch.Size(shape)
        chunks = tuple((torch.Size(offset), value) for offset, value in chunks)
        if not chunks and (dtype is None or device is None):
            raise ValueError(
                "Empty checkpoint chunks require explicit dtype and device"
            )
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("Checkpoint global dimensions must be positive")
        first = chunks[0][1] if chunks else None
        dtype = dtype if dtype is not None else first.dtype
        device = torch.device(device) if device is not None else first.device
        for offset, value in chunks:
            if isinstance(value, cls):
                raise TypeError("Checkpoint chunks must be ordinary tensors")
            if value.is_meta or value.layout != torch.strided:
                raise ValueError(
                    "Checkpoint chunks require materialized strided storage"
                )
            if value.dtype != dtype or value.device != device:
                raise ValueError("Checkpoint chunks must share dtype and device")
            if len(offset) != len(shape) or value.ndim != len(shape):
                raise ValueError("Checkpoint chunk dimensions differ from global shape")
            if any(
                start < 0 or size <= 0 or start + size > bound
                for start, size, bound in zip(offset, value.shape, shape, strict=True)
            ):
                raise ValueError("Checkpoint chunk is outside global bounds")
        for index, (offset, value) in enumerate(chunks):
            for other_offset, other in chunks[:index]:
                if all(
                    start < other_start + other_size and other_start < start + size
                    for start, size, other_start, other_size in zip(
                        offset, value.shape, other_offset, other.shape, strict=True
                    )
                ):
                    raise ValueError("Checkpoint local chunks overlap")
        result = torch.Tensor._make_wrapper_subclass(
            cls, shape, dtype=dtype, device=device, requires_grad=False
        )
        result.chunks = chunks
        return result

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        return [
            ChunkStorageMetadata(offsets=offset, sizes=value.size())
            for offset, value in self.chunks
        ]

    def __create_write_items__(self, fqn: str, obj: object) -> list[WriteItem]:
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunk.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=chunk,
                    properties=TensorProperties(dtype=self.dtype),
                    size=self.size(),
                ),
            )
            for chunk in self.__create_chunk_list__()
        ]

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        for offset, value in self.chunks:
            if index.offset == offset:
                return value
        raise ValueError(f"Unknown checkpoint chunk offset: {index.offset}")

    def refresh_from(self, current: ChunkedCheckpointTensor) -> ChunkedCheckpointTensor:
        """Keep cached storage, detaching once if exported live aliases change."""
        if (
            self.shape != current.shape
            or self.dtype != current.dtype
            or self.device != current.device
            or [(offset, value.shape) for offset, value in self.chunks]
            != [(offset, value.shape) for offset, value in current.chunks]
        ):
            raise ValueError("Checkpoint cache chunk layout changed")
        # Check every cross-chunk alias before writing anything: hooks may swap
        # the physical views while retaining the same global chunk metadata.
        for index, (_, cached) in enumerate(self.chunks):
            for other_index, (_, live) in enumerate(current.chunks):
                if not torch._C._is_alias_of(cached, live):
                    continue
                same_view = (
                    index == other_index
                    and cached.shape == live.shape
                    and cached.stride() == live.stride()
                    and cached.storage_offset() == live.storage_offset()
                )
                # Sharing a storage with disjoint sibling views is harmless.
                # Exact same-index views still need no copy; any changed view
                # conservatively detaches the whole local cache.
                if index == other_index and not same_view:
                    return current.detach().clone()
                if index != other_index and not torch._C._is_alias_of(
                    cached, current.chunks[index][1]
                ):
                    return current.detach().clone()
        for (_, cached), (_, live) in zip(self.chunks, current.chunks, strict=True):
            if not torch._C._is_alias_of(cached, live):
                cached.copy_(live)
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.new_empty.default:
            source, shape = args[:2]
            if torch.Size(shape) != source.shape:
                raise ValueError("Checkpoint allocation requires the same global shape")
            return cls(
                source.shape,
                [
                    (offset, value.new_empty(value.shape, **kwargs))
                    for offset, value in source.chunks
                ],
                dtype=kwargs.get("dtype", source.dtype),
                device=kwargs.get("device", source.device),
            )
        if func in (
            torch.ops.aten.zeros_like.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.detach.default,
            torch.ops.aten._to_copy.default,
        ):
            source = args[0]
            return cls(
                source.shape,
                [(offset, func(value, **kwargs)) for offset, value in source.chunks],
                dtype=kwargs.get("dtype", source.dtype),
                device=kwargs.get("device", source.device),
            )
        if func == torch.ops.aten.copy_.default:
            destination, source = args[:2]
            if not isinstance(source, cls):
                raise TypeError("Checkpoint copy requires a chunked source")
            target_layout = [
                (offset, value.shape) for offset, value in destination.chunks
            ]
            source_layout = [(offset, value.shape) for offset, value in source.chunks]
            if destination.shape != source.shape or target_layout != source_layout:
                raise ValueError("Checkpoint copy requires identical chunk layouts")
            if (
                destination.dtype == source.dtype
                and destination.device == source.device
            ):
                same_views = all(
                    torch._C._is_alias_of(target, origin)
                    and target.stride() == origin.stride()
                    and target.storage_offset() == origin.storage_offset()
                    for (_, target), (_, origin) in zip(
                        destination.chunks, source.chunks, strict=True
                    )
                )
                if same_views:
                    return destination
            non_blocking = (
                args[2] if len(args) > 2 else kwargs.get("non_blocking", False)
            )
            # Snapshot all sources before writes if any destination aliases a
            # source. Cross-chunk swaps otherwise overwrite unread values.
            source_chunks = source.chunks
            if any(
                torch._C._is_alias_of(target, origin)
                for _, target in destination.chunks
                for _, origin in source_chunks
            ):
                source_chunks = tuple(
                    (offset, value.clone()) for offset, value in source_chunks
                )
            for (_, target), (_, origin) in zip(
                destination.chunks, source_chunks, strict=True
            ):
                target.copy_(origin, non_blocking=non_blocking)
            return destination
        raise NotImplementedError(f"Checkpoint tensor does not support {func}")
