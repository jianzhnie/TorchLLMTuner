"""Meshes used to count tokens and report train or validation losses.

The loader batch is replicated over CP and TP, so its valid-token count is
reduced over DP alone. The model's loss covers a sequence shard and needs the
DP/CP/TP loss mesh. Keep this choice shared by both passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from ...parallel.topology import ParallelTopology


@dataclass(frozen=True, slots=True)
class TrainingReductionMeshes:
    dp_mesh: DeviceMesh | None
    pp_mesh: DeviceMesh | None
    loss_mesh: DeviceMesh | None
    sequence_mesh: DeviceMesh | None


def resolve_loss_reduction_meshes(
    parallel_dims: ParallelTopology | None,
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """Return the DP token-count mesh and the loss-sum mesh."""
    if parallel_dims is None:
        return None, None
    dp_mesh = parallel_dims.get_optional_mesh("dp")
    loss_is_sharded = parallel_dims.dp_cp_enabled or parallel_dims.tp_enabled
    loss_mesh = parallel_dims.get_optional_mesh("loss") if loss_is_sharded else dp_mesh
    return dp_mesh, loss_mesh


def resolve_training_reduction_meshes(
    parallel_dims: ParallelTopology | None,
) -> TrainingReductionMeshes:
    """Resolve the four meshes used by one training step."""
    dp_mesh, loss_mesh = resolve_loss_reduction_meshes(parallel_dims)
    if parallel_dims is None:
        return TrainingReductionMeshes(None, None, None, None)
    sequence_axes = [
        axis
        for axis, enabled in (
            ("cp", parallel_dims.cp_enabled),
            ("tp", parallel_dims.tp_enabled),
        )
        if enabled
    ]
    sequence_mesh = (
        parallel_dims.get_optional_mesh(sequence_axes) if sequence_axes else None
    )
    if sequence_mesh is not None and len(sequence_axes) > 1:
        sequence_mesh = sequence_mesh._flatten("sequence_loss")
    return TrainingReductionMeshes(
        dp_mesh=dp_mesh,
        pp_mesh=parallel_dims.get_optional_mesh("pp"),
        loss_mesh=loss_mesh,
        sequence_mesh=sequence_mesh,
    )
