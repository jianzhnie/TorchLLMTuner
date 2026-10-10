"""Pipeline schedule input splitting and one forward/backward step.

This module owns the PP-specific runtime path. The Trainer keeps its public
methods and delegates here so existing callers can use the same interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..accelerator.capabilities import has as capability
from ..accelerator.spmd_context import spmd_context
from ..datasets.loader import TrainerBatch
from ..datasets.types import Batch
from ..parallel import matrix

if TYPE_CHECKING:
    from .trainer import Trainer


def split_pipeline_microbatches(
    trainer: Trainer, batch: Batch | TrainerBatch
) -> list[Batch | TrainerBatch]:
    """Split whole rows for PP without cutting packed document boundaries."""
    labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
    num_microbatches = trainer.cfg.parallel.num_pp_microbatches
    if isinstance(batch, dict):
        if labels.ndim == 1 and num_microbatches > 1:
            matrix.pp_packed_microbatch_split(num_microbatches)
        rows_per_microbatch = labels.shape[0] // num_microbatches
        microbatches = []
        for index in range(num_microbatches):
            rows = slice(
                index * rows_per_microbatch,
                (index + 1) * rows_per_microbatch,
            )
            microbatches.append(
                {
                    key: value[rows]
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                    for key, value in batch.items()
                }
            )
        return microbatches

    input_chunks = batch.input_ids.chunk(num_microbatches, dim=0)
    label_chunks = batch.labels.chunk(num_microbatches, dim=0)
    return [
        Batch(input_ids=inputs, labels=targets)
        for inputs, targets in zip(input_chunks, label_chunks, strict=True)
    ]


def forward_backward_pipeline(
    trainer: Trainer,
    batch: Batch | TrainerBatch,
    *,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Drive one PP schedule step and return its unnormalized local loss sum."""
    arg_mbs: list[tuple[torch.Tensor, ...]] = []
    kwarg_mbs: list[dict[str, Any]] = []
    target_mbs: list[torch.Tensor] | None = [] if trainer.pp_has_last_stage else None
    for microbatch in trainer.pp_microbatches(batch):
        inputs, labels, extra_kwargs = trainer.preprocess({"batch": microbatch})
        if trainer.pp_has_first_stage:
            arg_mbs.append((inputs,))
        kwarg_mbs.append(extra_kwargs)
        if target_mbs is not None:
            target_mbs.append(labels)

    losses: list[torch.Tensor] | None = [] if trainer.pp_has_last_stage else None
    with trainer.param_context(), spmd_context(trainer.parallel_dims):
        # The private driver accepts already-split microbatches. Its setup is
        # normally done by step(), so stage backward flags and runtime state
        # must be initialized explicitly before calling it.
        trainer.pp_schedule._llmtuner_global_valid_tokens = global_valid_tokens
        if capability("pipelining_microbatch_drivers"):
            stages = getattr(trainer.pp_schedule, "_stages", None)
            if stages is None:
                stages = [trainer.pp_schedule._stage]
            for stage in stages:
                stage.has_backward = trainer.pp_schedule._has_backward
                stage.clear_runtime_states()
            trainer.pp_schedule._step_microbatches(
                arg_mbs if trainer.pp_has_first_stage else None,
                kwarg_mbs,
                target_mbs,
                losses,
                return_outputs=False,
            )
        else:
            trainer.pp_schedule.step(
                arg_mbs=arg_mbs if trainer.pp_has_first_stage else None,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
                losses=losses,
                loss_kwargs={"global_valid_tokens": global_valid_tokens},
                return_outputs=False,
            )

    if trainer.pp_has_last_stage:
        assert losses is not None
        # The schedule backpropagated normalized losses. Convert their
        # detached sum to the raw token sum expected by Trainer.train_step.
        detached_losses = [loss.detach() for loss in losses]
        losses.clear()
        return torch.sum(torch.stack(detached_losses)) * global_valid_tokens
    return trainer._pp_loss_sentinel
