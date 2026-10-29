"""Instance-scoped NPU metadata communication for torch_checkpointing 0.1.0."""

from __future__ import annotations

import pickle
from importlib.metadata import version
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ...accelerator.distributed import get_comm_device
from ...utils.logging import get_logger

logger = get_logger(__name__)


def _gather_pickled_metadata(
    manager: Any, data: Any, process_group: ProcessGroup | None = None
) -> dict[int, Any]:
    device = get_comm_device(process_group)
    serialized = pickle.dumps(data)
    payload = len(serialized).to_bytes(8, byteorder="little") + serialized
    maximum = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)
    padded = payload + b"\0" * (int(maximum.item()) - len(payload))
    local = torch.frombuffer(bytearray(padded), dtype=torch.uint8).to(device)
    gathered = [
        torch.empty_like(local) for _ in range(dist.get_world_size(process_group))
    ]
    dist.all_gather(gathered, local, group=process_group)
    result = {}
    for rank, tensor in enumerate(gathered):
        raw = tensor.cpu().numpy().tobytes()
        length = int.from_bytes(raw[:8], byteorder="little")
        result[rank] = pickle.loads(raw[8 : 8 + length])
    return result


def adapt_npu_metadata_gather(manager: Any) -> bool:
    if not dist.is_initialized() or get_comm_device().type != "npu":
        return False
    if version("torch_checkpointing") != "0.1.0":
        return False
    from torch_checkpointing.metadata_manager import DefaultMetadataManager

    metadata = getattr(manager, "_metadata_manager", None)
    if type(metadata) is not DefaultMetadataManager:
        return False
    metadata._all_gather_pickled_data = MethodType(_gather_pickled_metadata, metadata)
    logger.info("Applied instance-scoped torch_checkpointing 0.1.0 NPU metadata gather")
    return True
