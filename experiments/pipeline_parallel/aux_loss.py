"""Experimental cross-stage auxiliary-loss metrics."""

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn

from experiments.pipeline_parallel.collectives import all_reduce
from experiments.pipeline_parallel.topology import ParallelTopology
from llmtuner.accelerator.device import device_type
from llmtuner.models.common.aux_loss import (
    AuxLoss,
    snapshot_aux_loss_metrics,
)


def collect_aux_loss_metrics(
    parallel_dims: ParallelTopology,
    *,
    model_parts: Sequence[nn.Module] | None = None,
) -> dict[str, float]:
    """Reduce the current step's ``group_acc`` registers for logging.

    Returns ``{metric_name}/mean`` per group, or ``{}`` when no auxiliary loss
    is configured. All ranks call this at log time. When model_parts are
    supplied, use only their snapshots and agree on metric keys across PP stages,
    including stages with no local loss. Standalone calls without parts retain
    the constructor registry; the trainer always supplies its current parts.
    """
    pp_mesh = parallel_dims.get_optional_mesh("pp")
    runtime_totals: dict[tuple[str, str], torch.Tensor] = {}
    if model_parts is not None:
        counts, runtime_totals = snapshot_aux_loss_metrics(model_parts)
        if pp_mesh is not None:
            # Stage-local groups may differ, including a stage with none.
            # Agree on keys before issuing the per-group tensor collectives.
            stage_keys = [None] * pp_mesh.size()
            dist.all_gather_object(
                stage_keys, sorted(counts), group=pp_mesh.get_group()
            )
            counts = {key: counts.get(key, 0) for keys in stage_keys for key in keys}
    else:
        counts = dict(AuxLoss._group_counts)

    totals = AuxLoss.group_acc if model_parts is None else runtime_totals
    metrics = {}
    for key, count in sorted(counts.items()):
        mesh_name, tag = key
        total = totals.get(key)
        if total is None:
            # A stage with no local instance still joins every collective.
            total = torch.zeros((), dtype=torch.float32, device=device_type)
        reduce_mesh = parallel_dims.get_optional_mesh(mesh_name)
        for mesh in (reduce_mesh, pp_mesh):
            if mesh is None:
                continue
            # Sum: each coordinate contributes its own data, and every layer
            # lives on exactly one pipeline stage, so summing over the reduce
            # mesh and the stages counts every layer once. Dividing by the
            # current runtime instance count below then
            # gives the mean over all layers.
            total = all_reduce(total, reduceOp="sum", group=mesh)
        if model_parts is not None:
            # Constructor counts include previously built models in the same
            # process. Count the current runtime's distinct local instances,
            # then combine pipeline stages (data peers replicate each layer).
            layer_count = torch.tensor(count, dtype=torch.int64, device=total.device)
            if pp_mesh is not None:
                layer_count = all_reduce(layer_count, reduceOp="sum", group=pp_mesh)
            count = int(layer_count.item())
            if count == 0:
                continue
        metrics[f"{tag}/mean"] = float(total.item()) / count
    return metrics
