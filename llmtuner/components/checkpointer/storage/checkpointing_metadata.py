"""Instance-scoped metadata communication for non-CUDA checkpoint backends."""

from __future__ import annotations

import pickle
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ....accelerator.collectives import synchronize_local_errors
from ....accelerator.distributed import get_comm_device
from ....utils.logging import get_logger

try:
    from torch_checkpointing.metadata_manager import DefaultMetadataManager
except ImportError as error:
    DefaultMetadataManager = None
    BACKEND_IMPORT_ERROR = error

logger = get_logger(__name__)


def _gather_pickled_metadata(
    manager: Any, data: Any, process_group: ProcessGroup | None = None
) -> dict[int, Any]:
    # Keep communication outside local failure boundaries: a peer must not
    # enter the next collective after another rank failed serialization or IO.
    with synchronize_local_errors("Checkpoint metadata encoding", group=process_group):
        device = get_comm_device(process_group)
        serialized = pickle.dumps(data)
        payload = len(serialized).to_bytes(8, byteorder="little") + serialized
        maximum = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)
    with synchronize_local_errors("Checkpoint metadata buffers", group=process_group):
        padded = payload + b"\0" * (int(maximum.item()) - len(payload))
        local = torch.frombuffer(bytearray(padded), dtype=torch.uint8).to(device)
        gathered = [
            torch.empty_like(local) for _ in range(dist.get_world_size(process_group))
        ]
    dist.all_gather(gathered, local, group=process_group)
    with synchronize_local_errors("Checkpoint metadata decoding", group=process_group):
        result = {}
        for rank, tensor in enumerate(gathered):
            raw = tensor.cpu().numpy().tobytes()
            length = int.from_bytes(raw[:8], byteorder="little")
            if not 0 < length <= len(raw) - 8:
                raise ValueError(f"Invalid checkpoint metadata length from rank {rank}")
            result[rank] = pickle.loads(raw[8 : 8 + length])
    return result


def adapt_metadata_gather(manager: Any) -> bool:
    # Upstream hard-codes CUDA buffers even for CPU/Gloo checkpoints.
    if not dist.is_initialized() or get_comm_device().type == "cuda":
        return False
    if DefaultMetadataManager is None:
        raise ImportError(
            "torch_checkpointing metadata_manager API is unavailable"
        ) from (BACKEND_IMPORT_ERROR)

    metadata = getattr(manager, "_metadata_manager", None)
    if type(metadata) is not DefaultMetadataManager:
        return False
    metadata._all_gather_pickled_data = MethodType(_gather_pickled_metadata, metadata)
    logger.info("Applied instance-scoped device-aware metadata gather")
    return True


def configure_sync_metadata(manager: Any) -> None:
    """Publish every synchronous checkpoint without background serialization.

    The sync writer serializes compute_metadata's return value itself; a cache
    hit produces payload files without a metadata marker. Configure only this
    manager's default implementation, retaining async and custom policies.
    """
    metadata = getattr(manager, "_metadata_manager", None)
    if DefaultMetadataManager is not None and type(metadata) is DefaultMetadataManager:
        metadata._should_cache_metadata = False
        # SyncCheckpointSaver pickles the returned metadata itself. Its writer
        # never consumes the metadata manager's background serialized cache.
        metadata._enable_serialization = False
