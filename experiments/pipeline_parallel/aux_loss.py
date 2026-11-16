"""Experimental cross-stage auxiliary-loss metrics."""

from collections import Counter
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn

from experiment.pipeline_parallel.collectives import all_reduce
from experiment.pipeline_parallel.topology import ParallelTopology
from llmtuner.accelerator.device import device_type
from llmtuner.models.common.aux_loss import AuxLoss, rollup_aux_loss_metrics


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
    counts = dict(AuxLoss._group_counts)
    runtime_totals: dict[tuple[str, str], torch.Tensor] = {}
    if model_parts is not None:
        instances = {
            id(module): module
            for part in model_parts
            for module in part.modules()
            if isinstance(module, AuxLoss)
        }
        counts = Counter(
            (module.reduce_mesh, module.metric_name) for module in instances.values()
        )
        for module in instances.values():
            key = (module.reduce_mesh, module.metric_name)
            value = module._step_metric
            if value is None:
                continue
            if key not in runtime_totals:
                runtime_totals[key] = value.clone()
            else:
                runtime_totals[key].add_(value)
        if pp_mesh is not None:
            # Stage-local groups may differ, including a stage with none.
            # Agree on keys before issuing the per-group tensor collectives.
            stage_keys = [None] * pp_mesh.size()
            dist.all_gather_object(
                stage_keys, sorted(counts), group=pp_mesh.get_group()
            )
            counts = {key: counts.get(key, 0) for keys in stage_keys for key in keys}

    metrics = {}
    for key, count in sorted(counts.items()):
        mesh_name, tag = key
        total = (
            runtime_totals.get(key)
            if model_parts is not None
            else AuxLoss.group_acc.get(key)
        )
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


def register_aux_loss_metrics_hook(
    optimizer: torch.optim.Optimizer,
    *,
    model_parts: Sequence[nn.Module],
) -> None:
    """Register the step pre-hook that rolls ``instance_acc`` into ``group_acc``.

    This roll-up is local; distributed metric reduction happens later in
    ``collect_aux_loss_metrics``.
    """
    tracked_parts = tuple(model_parts)
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: rollup_aux_loss_metrics(tracked_parts)
    )
