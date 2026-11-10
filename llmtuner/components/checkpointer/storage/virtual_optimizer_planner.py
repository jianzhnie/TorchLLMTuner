"""Transfer virtual optimizer state through ordinary device storage for DCP."""

from copy import copy
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import DefaultSavePlanner
from torch.distributed.checkpoint.planner import ReadItem, WriteItem

from ....accelerator.collectives import synchronize_local_errors
from ..core.chunked_tensor import ChunkedCheckpointTensor
from .load_planner import CheckpointLoadPlanner


def snapshot_virtual_state(value: Any) -> Any:
    if isinstance(value, ChunkedCheckpointTensor):
        # Snapshot and release each device clone before allocating the next;
        # cloning the wrapper first would retain every local chunk on NPU.
        return ChunkedCheckpointTensor(
            value.shape,
            [(offset, snapshot_virtual_state(chunk)) for offset, chunk in value.chunks],
            dtype=value.dtype,
            device="cpu",
        )
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu() if value.device.type == "npu" else value
    if isinstance(value, dict):
        result = copy(value)
        for key, child in value.items():
            result[key] = snapshot_virtual_state(child)
        return result
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(snapshot_virtual_state(child) for child in value))
    if isinstance(value, list | tuple):
        return type(value)(snapshot_virtual_state(child) for child in value)
    return value


class VirtualOptimizerStager:
    """Snapshot virtual moments before the normal asynchronous CPU staging.

    DCP has materialized Stateful objects before calling stage. Read optimizer
    values on the caller's producer stream before training resumes, bounded by
    one local tensor/chunk of temporary NPU storage. Host snapshots remain live
    until the underlying stager finishes; disk upload remains asynchronous.
    """

    def __init__(self, stager: Any, group: dist.ProcessGroup | None = None) -> None:
        self.stager = stager
        self.group = group

    @property
    def should_synchronize_after_execute(self) -> bool:
        return self.stager.should_synchronize_after_execute

    def stage(self, state_dict: dict[str, Any]):
        with synchronize_local_errors("Virtual optimizer snapshot", group=self.group):
            snapshot = dict(state_dict)
            if "optimizer" in snapshot:
                snapshot["optimizer"] = snapshot_virtual_state(snapshot["optimizer"])
            staged = self.stager.stage(snapshot)
        return staged

    def synchronize_staging(self) -> None:
        self.stager.synchronize_staging()

    def close(self) -> None:
        self.stager.close()


class VirtualOptimizerSavePlanner(DefaultSavePlanner):
    """Snapshot one resolved local optimizer shard at a time.

    NPU Host-backed moments may expose stale host bytes through a direct CPU
    transfer, including when DCP resolves a chunk to an ordinary tensor view.
    A device clone reads the values used by Adam before copying them to CPU.
    Keeping this at resolve_data avoids cloning the entire optimizer on NPU.
    This planner is for synchronous saves, while training is stopped.
    """

    def resolve_data(self, write_item: WriteItem):
        value = super().resolve_data(write_item)
        if (
            write_item.index.fqn.startswith("optimizer.")
            and isinstance(value, torch.Tensor)
            and value.device.type == "npu"
        ):
            return value.detach().clone().cpu()
        return value


class VirtualOptimizerLoadPlanner(CheckpointLoadPlanner):
    """Stage each local read on NPU before updating Host-backed destinations.

    A direct CPU-to-virtual-view copy may leave the values read by NPU kernels
    unchanged. DCP fills an ordinary device tensor; commit copies its values to
    the exact local slice selected by the normal load planner. No global tensor
    or persistent per-parameter staging cache is allocated.
    """

    def resolve_tensor(self, read_item: ReadItem):
        destination = super().resolve_tensor(read_item)
        if (
            read_item.dest_index.fqn.startswith("optimizer.")
            and destination.device.type == "npu"
        ):
            return torch.empty_like(destination)
        return destination

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        if (
            read_item.dest_index.fqn.startswith("optimizer.")
            and tensor.device.type == "npu"
        ):
            super().resolve_tensor(read_item).copy_(tensor)
