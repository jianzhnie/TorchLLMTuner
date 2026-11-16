"""Rank-consistent metrics for one optimizer step."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from ...accelerator.collectives import all_reduce
from ...models.common.aux_loss import collect_aux_loss_metrics
from .reductions import TrainingReductionMeshes
from .results import AccumulationResult, UpdateResult

if TYPE_CHECKING:
    from ...parallel.topology import ParallelTopology
    from .training_observer import DispatchTimer


def collect_resource_metrics(
    *,
    activation_swap: Any,
    optimizer: Any,
    checkpointer: Any,
    dispatch_timers: Iterable[DispatchTimer] = (),
) -> dict[str, float]:
    """Rank-local resource counters, explicitly labelled rather than summed.

    Activation events are measured only when activation swap is configured.
    Byte counters are cumulative; device transfer seconds consume the log window.
    """
    result = {}
    result.update(getattr(activation_swap, "observation_metrics", {}).copy())
    result.update(getattr(optimizer, "observation_metrics", {}).copy())
    result.update(getattr(checkpointer, "observation_metrics", {}).copy())
    timers = tuple(dispatch_timers)
    if timers:
        result["moe/local_forward_dispatch_seconds_window"] = sum(
            timer.consume_seconds() for timer in timers
        )
    return result


def build_step_metrics(
    *,
    ntokens_seen: int,
    device: torch.device,
    accumulation: AccumulationResult,
    global_valid_tokens: torch.Tensor,
    local_valid_tokens: int,
    reduction_meshes: TrainingReductionMeshes,
    update: UpdateResult,
    lr_metrics: dict[str, float],
) -> dict[str, float]:
    """Reduce loss and token metrics without rank-local collective guards."""
    accumulated_loss = accumulation.loss_sum
    if accumulated_loss is None:
        raise ValueError("Step reporting requires an accumulation result with loss_sum")
    dp_mesh = reduction_meshes.dp_mesh
    loss_mesh = reduction_meshes.loss_mesh
    sequence_mesh = reduction_meshes.sequence_mesh
    # Division creates reporting tensors owned here; collectives can update
    # them directly without copying or mutating the accumulation result.
    loss_sum = accumulated_loss / global_valid_tokens

    if loss_mesh is not None:
        local_loss_sum = accumulated_loss.clone()
        if sequence_mesh is not None:
            all_reduce(local_loss_sum, group=sequence_mesh.get_group())
        local_max = (
            local_loss_sum / local_valid_tokens
            if local_valid_tokens > 0
            else torch.zeros_like(accumulated_loss)
        )
        all_reduce(loss_sum, group=loss_mesh.get_group())
        if dp_mesh is not None:
            all_reduce(local_max, op="max", group=dp_mesh.get_group())
        ntokens_seen_tensor = torch.tensor(
            ntokens_seen, dtype=torch.int64, device=device
        )
        all_reduce(ntokens_seen_tensor, group=loss_mesh.get_group())
        global_ntokens_seen = float(ntokens_seen_tensor)
    else:
        local_max = loss_sum.clone()
        global_ntokens_seen = float(ntokens_seen)
    metrics = {
        "loss": float(loss_sum),
        "max_loss": float(local_max),
        "n_tokens_seen": global_ntokens_seen,
    }
    if update.grad_norm is not None:
        metrics["grad_norm"] = float(update.grad_norm)
    metrics.update(lr_metrics)
    metrics.update(update.auxiliary_metrics)
    return metrics


def collect_update_metrics(
    result: UpdateResult,
    *,
    parallel_dims: ParallelTopology | None,
    model_parts: list[torch.nn.Module],
) -> UpdateResult:
    """Attach post-hook auxiliary snapshots at the existing logging boundary.

    Every reporting rank participates in the same reductions, including PP
    stages without local losses. Non-reporting steps do not call this function.
    """
    metrics = dict(result.auxiliary_metrics)
    if parallel_dims is not None:
        metrics.update(collect_aux_loss_metrics(parallel_dims, model_parts=model_parts))
    return replace(result, auxiliary_metrics=metrics)
