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
    from ..parallel.parallel_dims import ParallelDims


@dataclass(frozen=True, slots=True)
class TrainingReductionMeshes:
    dp: DeviceMesh | None
    pp: DeviceMesh | None
    loss: DeviceMesh | None
    sequence: DeviceMesh | None


def loss_reporting_meshes(
    parallel_dims: ParallelDims | None,
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """Return the DP token-count mesh and the loss-sum mesh."""
    if parallel_dims is None:
        return None, None
    dp_mesh = parallel_dims.get_optional_mesh("dp")
    loss_is_sharded = parallel_dims.dp_cp_enabled or parallel_dims.tp_enabled
    loss_mesh = (
        parallel_dims.get_optional_mesh("loss") if loss_is_sharded else dp_mesh
    )
    return dp_mesh, loss_mesh


def training_reduction_meshes(
    parallel_dims: ParallelDims | None,
) -> TrainingReductionMeshes:
    """Resolve the four meshes used by one training step."""
    dp_mesh, loss_mesh = loss_reporting_meshes(parallel_dims)
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
    return TrainingReductionMeshes(
        dp=dp_mesh,
        pp=parallel_dims.get_optional_mesh("pp"),
        loss=loss_mesh,
        sequence=(
            parallel_dims.get_optional_mesh(sequence_axes)
            if sequence_axes
            else None
        ),
    )
