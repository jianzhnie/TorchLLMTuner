"""Select the same distributed checkpoint layout for model and optimizer state."""

from enum import Enum
from typing import Any

from torch import nn
from torch.distributed import ProcessGroup

from ....parallel.expert_parallel.checkpointing import (
    gather_expert_state,
    load_expert_state,
    restore_sharded_expert_state,
    shard_expert_state,
)
from ....parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
    restore_disjoint_tp_ep_state,
    restore_sharded_tensor_parallel_state,
    shard_disjoint_tp_ep_state,
    shard_tensor_parallel_state,
)
from .chunked_tensor import ChunkedCheckpointTensor


class RestoreLayout(Enum):
    """Resolved storage format, independent of the destination state owner."""

    FULL = "full"
    TENSOR_SHARDED = "tensor_sharded"
    EXPERT_SHARDED = "expert_sharded"
    JOINT_SHARDED = "joint_sharded"

    @property
    def requires_local_state(self) -> bool:
        return self is not RestoreLayout.FULL


def resolve_restore_layout(
    state: dict[str, Any],
    *,
    sharded_tensor_parallel: bool,
    sharded_expert_parallel: bool,
) -> RestoreLayout:
    """Resolve locally before destination construction or layout collectives."""
    if sharded_expert_parallel:
        return (
            RestoreLayout.JOINT_SHARDED
            if sharded_tensor_parallel
            else RestoreLayout.EXPERT_SHARDED
        )
    chunked_keys = [
        key
        for key, value in state.items()
        if isinstance(value, ChunkedCheckpointTensor)
    ]
    if chunked_keys:
        if not sharded_tensor_parallel:
            raise ValueError(
                "Chunked checkpoint state requires sharded TP or EP restore: "
                f"{chunked_keys}"
            )
        return RestoreLayout.TENSOR_SHARDED
    return RestoreLayout.FULL


def export_parallel_state(
    state: dict[str, Any],
    *,
    model_parts: list[nn.Module],
    expert_shards: dict[str, ProcessGroup],
    sharded_tensor_parallel: bool,
    sharded_expert_parallel: bool,
) -> dict[str, Any]:
    """Export after local validation; tensor transforms may communicate.

    Joint TP/EP sharding uses disjoint ownership coordinates directly. Other
    formats transform expert ownership before the tensor-parallel layout.
    """
    if sharded_tensor_parallel and sharded_expert_parallel:
        return shard_disjoint_tp_ep_state(state, model_parts=model_parts)
    expert_transform = (
        shard_expert_state if sharded_expert_parallel else gather_expert_state
    )
    tensor_transform = (
        shard_tensor_parallel_state
        if sharded_tensor_parallel
        else gather_tensor_parallel_state
    )
    return tensor_transform(
        expert_transform(state, shard_map=expert_shards, model_parts=model_parts),
        model_parts=model_parts,
    )


def restore_parallel_state(
    state: dict[str, Any],
    *,
    local_state: dict[str, Any] | None,
    model_parts: list[nn.Module],
    expert_shards: dict[str, ProcessGroup],
    layout: RestoreLayout,
    coordinate_expert_errors: bool = False,
) -> dict[str, Any]:
    """Convert validated state, potentially copying into live local destinations.

    Callers prepare destinations and retire export caches before this operation.
    Conversion may communicate and partially write destination storage; failure
    is not transactional. Consumer load hooks run after conversion completes.
    """
    if not isinstance(layout, RestoreLayout):
        raise TypeError("Restore requires a resolved RestoreLayout")
    if layout.requires_local_state and local_state is None:
        raise ValueError("Sharded restore requires local state destinations")
    if layout is RestoreLayout.JOINT_SHARDED:
        return restore_disjoint_tp_ep_state(
            state, local_state=local_state, model_parts=model_parts
        )
    if layout is RestoreLayout.EXPERT_SHARDED:
        state = restore_sharded_expert_state(
            state,
            local_state=local_state,
            shard_map=expert_shards,
            model_parts=model_parts,
            coordinate_errors=coordinate_expert_errors,
        )
        return load_tensor_parallel_state(state, model_parts=model_parts)
    if layout is RestoreLayout.TENSOR_SHARDED:
        local_state = gather_expert_state(
            local_state, shard_map=expert_shards, model_parts=model_parts
        )
        state = restore_sharded_tensor_parallel_state(
            state, local_state=local_state, model_parts=model_parts
        )
    else:
        state = load_tensor_parallel_state(state, model_parts=model_parts)
    return load_expert_state(state, shard_map=expert_shards, model_parts=model_parts)
