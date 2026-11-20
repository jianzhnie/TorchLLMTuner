"""Meshes used to count tokens and report train or validation losses.

The loader batch is replicated over CP and TP, so its valid-token count is
reduced over DP alone. The model's loss covers a sequence shard and needs the
DP/CP/TP loss mesh. Keep this choice shared by both passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch.distributed.device_mesh import DeviceMesh

from llmtuner.trainer.runtime.reductions import (
    logical_tokens_per_rank as logical_tokens_per_rank,
)
from llmtuner.trainer.runtime.reductions import (
    resolve_loss_reduction_meshes as resolve_loss_reduction_meshes,
)
from llmtuner.trainer.runtime.reductions import (
    resolve_training_reduction_meshes as resolve_dense_reduction_meshes,
)

if TYPE_CHECKING:
    from experiments.pipeline_parallel.topology import ParallelTopology


@dataclass(frozen=True, slots=True)
class TrainingReductionMeshes:
    dp_mesh: DeviceMesh | None
    pp_mesh: DeviceMesh | None
    loss_mesh: DeviceMesh | None
    sequence_mesh: DeviceMesh | None


def resolve_training_reduction_meshes(
    parallel_dims: ParallelTopology | None,
) -> TrainingReductionMeshes:
    """Add pipeline stage reduction to the shared token/loss mesh selection."""
    meshes = resolve_dense_reduction_meshes(parallel_dims)
    return TrainingReductionMeshes(
        dp_mesh=meshes.dp_mesh,
        pp_mesh=(
            parallel_dims.get_optional_mesh("pp") if parallel_dims is not None else None
        ),
        loss_mesh=meshes.loss_mesh,
        sequence_mesh=meshes.sequence_mesh,
    )
