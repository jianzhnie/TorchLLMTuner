"""Pipeline schedule input splitting and one forward/backward step.

This module owns the PP-specific runtime path. The Trainer keeps its public
methods and delegates here so existing callers can use the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ...accelerator.capabilities import has as capability
from ...accelerator.spmd import spmd_context
from ...components.loss import IGNORE_INDEX
from ...datasets.contracts import Batch, TrainerBatch, require_positive_integer
from ...parallel import compatibility

if TYPE_CHECKING:
    from ..trainer import Trainer


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineInputs:
    """Prepared microbatches for a rank's pipeline stage ownership."""

    args: list[tuple[torch.Tensor, ...]]
    kwargs: list[dict[str, Any]]
    targets: list[torch.Tensor] | None


def split_pipeline_microbatches(
    batch: Batch | TrainerBatch, *, num_microbatches: int
) -> list[Batch | TrainerBatch]:
    """Split whole rows for PP without cutting packed document boundaries.

    A single microbatch keeps every tensor intact: packed media patches and
    coordinate axes do not share the labels' leading dimension. Multiple
    microbatches require every non-scalar tensor to use the same leading row
    dimension; layouts needing media-aware slicing must be prepared separately.
    """
    require_positive_integer("num_pp_microbatches", num_microbatches)
    labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
    expected_ranks = (2,) if isinstance(batch, Batch) else (1, 2)
    if labels.ndim not in expected_ranks:
        raise ValueError(
            f"Pipeline batch labels must have rank in {expected_ranks}, "
            f"got shape {tuple(labels.shape)}"
        )
    if isinstance(batch, dict) and labels.ndim == 1 and num_microbatches > 1:
        compatibility.pp_packed_microbatch_split(num_microbatches)
    if labels.shape[0] % num_microbatches != 0:
        raise ValueError(
            "Pipeline batch rows must be divisible by num_pp_microbatches: "
            f"got {labels.shape[0]} rows and {num_microbatches} microbatches"
        )
    if labels.shape[0] == 0:
        raise ValueError("Pipeline batches must contain at least one row")
    if num_microbatches > 1:
        fields = (
            {"input_ids": batch.input_ids, "labels": labels}
            if isinstance(batch, Batch)
            else batch
        )
        for name, value in fields.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] != labels.shape[0]
            ):
                raise ValueError(
                    f"Pipeline batch field {name!r} has {value.shape[0]} leading "
                    f"entries, but labels have {labels.shape[0]} rows; multiple "
                    "microbatches require row-aligned tensors. Use one "
                    "microbatch for packed media or coordinate layouts."
                )
    if isinstance(batch, dict):
        rows_per_microbatch = labels.shape[0] // num_microbatches
        microbatches = []
        for index in range(num_microbatches):
            rows = slice(
                index * rows_per_microbatch,
                (index + 1) * rows_per_microbatch,
            )
            microbatch = {
                key: value[rows]
                if num_microbatches > 1
                and isinstance(value, torch.Tensor)
                and value.ndim > 0
                else value
                for key, value in batch.items()
            }
            if "num_valid_tokens" in batch:
                microbatch["num_valid_tokens"] = int(
                    (microbatch["labels"] != IGNORE_INDEX).sum()
                )
            microbatches.append(microbatch)
        return microbatches

    input_chunks = batch.input_ids.chunk(num_microbatches, dim=0)
    label_chunks = batch.labels.chunk(num_microbatches, dim=0)
    return [
        Batch(input_ids=inputs, labels=targets)
        for inputs, targets in zip(input_chunks, label_chunks, strict=True)
    ]


def prepare_pipeline_inputs(
    trainer: Trainer, batch: Batch | TrainerBatch
) -> PipelineInputs:
    """Prepare schedule inputs for the local first/last-stage ownership."""
    arg_mbs: list[tuple[torch.Tensor, ...]] = []
    kwarg_mbs: list[dict[str, Any]] = []
    target_mbs: list[torch.Tensor] | None = [] if trainer.pp_has_last_stage else None
    for microbatch in trainer.pp_microbatches(batch):
        inputs, labels, extra_kwargs = trainer.preprocess(microbatch)
        if trainer.pp_has_first_stage:
            arg_mbs.append((inputs,))
        kwarg_mbs.append(extra_kwargs)
        if target_mbs is not None:
            target_mbs.append(labels)

    return PipelineInputs(args=arg_mbs, kwargs=kwarg_mbs, targets=target_mbs)


def run_pipeline_step(
    trainer: Trainer,
    batch: Batch | TrainerBatch,
    *,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Drive one PP schedule step and return its unnormalized local loss sum."""
    inputs = prepare_pipeline_inputs(trainer, batch)

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
                inputs.args if trainer.pp_has_first_stage else None,
                inputs.kwargs,
                inputs.targets,
                losses,
                return_outputs=False,
            )
        else:
            trainer.pp_schedule.step(
                arg_mbs=inputs.args if trainer.pp_has_first_stage else None,
                kwarg_mbs=inputs.kwargs,
                target_mbs=inputs.targets,
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
