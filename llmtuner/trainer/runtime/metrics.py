"""Rank-consistent metrics for one optimizer step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...accelerator.collectives import all_reduce
from ...models.common.aux_loss import collect_aux_loss_metrics
from .reductions import TrainingReductionMeshes

if TYPE_CHECKING:
    from ..trainer import Trainer


def collect_resource_metrics(trainer: Trainer) -> dict[str, float]:
    """Rank-local resource counters, explicitly labelled rather than summed.

    Activation events are measured only when activation swap is configured.
    Byte counters are cumulative; device transfer seconds consume the log window.
    """
    from ...components.optimizer.adamw_swap import AdamWSwap

    result = {}
    if getattr(trainer.cfg, "activation_swap", False):
        from ...memory.activation_swap import activation_swap_metrics

        result.update(
            {
                f"swap/activation_local/{key}": float(value)
                for key, value in activation_swap_metrics().items()
            }
        )
    optimizers = getattr(trainer.optimizer, "optimizers", ())
    swaps = [optimizer for optimizer in optimizers if isinstance(optimizer, AdamWSwap)]
    if swaps:
        for key in (
            "h2d_bytes",
            "d2h_bytes",
            "host_state_bytes",
            "staging_peak_payload_bytes",
            "copy_wait_seconds",
        ):
            result[f"swap/optimizer_local/{key}"] = float(
                sum(getattr(optimizer, key) for optimizer in swaps)
            )
    result.update(getattr(trainer.checkpointer, "observation_metrics", {}).copy())
    timers = getattr(trainer, "dispatch_timers", ())
    if timers:
        result["moe/local_forward_dispatch_seconds_window"] = sum(
            timer.consume_seconds() for timer in timers
        )
    return result


def build_step_metrics(
    trainer: Trainer,
    *,
    accumulated_loss: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    local_valid_tokens: int,
    local_valid_tokens_tensor: torch.Tensor,
    reduction_meshes: TrainingReductionMeshes,
    grad_norm: torch.Tensor,
    lr_metrics: dict[str, float],
) -> dict[str, float]:
    """Reduce loss and token metrics without rank-local collective guards."""
    dp_mesh = reduction_meshes.dp_mesh
    pp_mesh = reduction_meshes.pp_mesh
    loss_mesh = reduction_meshes.loss_mesh
    sequence_mesh = reduction_meshes.sequence_mesh
    if pp_mesh is not None and not trainer.pp_has_last_stage:
        accumulated_loss = torch.zeros_like(accumulated_loss)
    loss = accumulated_loss / global_valid_tokens

    if loss_mesh is not None:
        local_loss_sum = accumulated_loss.clone()
        if sequence_mesh is not None:
            all_reduce(local_loss_sum, group=sequence_mesh.get_group())
        local_avg = (
            local_loss_sum / local_valid_tokens_tensor
            if local_valid_tokens > 0
            else torch.zeros_like(accumulated_loss)
        )
        loss_sum = loss.clone()
        local_max = local_avg.clone()
        all_reduce(loss_sum, group=loss_mesh.get_group())
        if dp_mesh is not None:
            all_reduce(local_max, op="max", group=dp_mesh.get_group())
        ntokens_seen_tensor = torch.tensor(
            trainer.ntokens_seen, dtype=torch.int64, device=trainer.device
        )
        all_reduce(ntokens_seen_tensor, group=loss_mesh.get_group())
        global_ntokens_seen = float(ntokens_seen_tensor)
    else:
        loss_sum = loss
        local_max = loss.clone()
        global_ntokens_seen = float(trainer.ntokens_seen)
    if pp_mesh is not None:
        all_reduce(loss_sum, group=pp_mesh.get_group())
        all_reduce(local_max, group=pp_mesh.get_group())
    metrics = {
        "loss": float(loss_sum),
        "max_loss": float(local_max),
        "grad_norm": float(grad_norm),
        "n_tokens_seen": global_ntokens_seen,
    }
    metrics.update(lr_metrics)
    if trainer.parallel_dims is not None:
        metrics.update(
            collect_aux_loss_metrics(
                trainer.parallel_dims, model_parts=trainer.model_parts
            )
        )
    return metrics
