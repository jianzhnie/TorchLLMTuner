"""Pipeline schedule input splitting and one forward/backward step.

The execution engine supplies stage ownership, preprocessing and schedule
dependencies. Trainer does not implement a second pipeline execution interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils._pytree import tree_leaves, tree_map_only

from experiments.pipeline_parallel import compatibility
from experiments.pipeline_parallel.collectives import synchronize_local_errors
from experiments.pipeline_parallel.microbatch_drivers import (
    run_eval_microbatches,
    run_microbatches,
)
from llmtuner.accelerator.distributed import get_world_size
from llmtuner.accelerator.spmd import spmd_context
from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.components.loss_contract import (
    validate_loss_sum,
    validate_normalization_tokens,
)
from llmtuner.datasets.contracts import Batch, TrainerBatch, require_positive_integer


class PipelineRuntime(Protocol):
    pp_schedule: Any
    device: torch.device
    error_group: ProcessGroup | None
    pp_has_first_stage: bool
    pp_has_last_stage: bool
    parallel_dims: Any
    pp_loss_sentinel: torch.Tensor | None

    def pp_microbatches(
        self, batch: Batch | TrainerBatch
    ) -> list[Batch | TrainerBatch]: ...

    def preprocess(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def param_context(self) -> Any: ...


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
    microbatches require every nested non-scalar tensor to use the same leading row
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
            for leaf in tree_leaves(value):
                if (
                    isinstance(leaf, torch.Tensor)
                    and leaf.ndim > 0
                    and leaf.shape[0] != labels.shape[0]
                ):
                    raise ValueError(
                        f"Pipeline batch field {name!r} has {leaf.shape[0]} leading "
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

            def slice_rows(value: torch.Tensor, selected: slice = rows) -> torch.Tensor:
                return value[selected] if value.ndim > 0 else value

            microbatch = (
                tree_map_only(torch.Tensor, slice_rows, batch)
                if num_microbatches > 1
                else dict(batch)
            )
            # A single microbatch retains the original labels and count. Only
            # sliced rows need a fresh count; do not synchronize device labels
            # just to reconstruct metadata already supplied by the collator.
            if num_microbatches > 1 and "num_valid_tokens" in batch:
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
    runtime: PipelineRuntime, batch: Batch | TrainerBatch
) -> PipelineInputs:
    """Prepare schedule inputs for the local first/last-stage ownership."""
    arg_mbs: list[tuple[torch.Tensor, ...]] = []
    kwarg_mbs: list[dict[str, Any]] = []
    target_mbs: list[torch.Tensor] | None = [] if runtime.pp_has_last_stage else None
    # Splitting must succeed everywhere before preprocessing enters its
    # coordinated boundary once for each resulting microbatch.
    with synchronize_local_errors(
        "Pipeline input splitting", group=runtime.error_group
    ):
        microbatches = runtime.pp_microbatches(batch)
        require_positive_integer("pipeline microbatch count", len(microbatches))
    if get_world_size(runtime.error_group) > 1:
        counts = [None] * get_world_size(runtime.error_group)
        dist.all_gather_object(counts, len(microbatches), group=runtime.error_group)
        if any(count != len(microbatches) for count in counts):
            raise ValueError("Pipeline ranks disagree on microbatch count")
    geometry = []
    for microbatch in microbatches:
        inputs, labels, extra_kwargs = runtime.preprocess(microbatch)
        geometry.append(
            (tuple(inputs.shape), inputs.dtype, tuple(labels.shape), labels.dtype)
        )
        if runtime.pp_has_first_stage:
            arg_mbs.append((inputs,))
        kwarg_mbs.append(extra_kwargs)
        if target_mbs is not None:
            target_mbs.append(labels)

    if get_world_size(runtime.error_group) > 1:
        # PP peers share a batch and a CP/TP coordinate. DP replicas may use
        # different lengths, so compare geometry only within each pipeline.
        with synchronize_local_errors(
            "Pipeline input geometry setup", group=runtime.error_group
        ):
            pp_mesh = runtime.parallel_dims.get_mesh("pp")
            peers = [None] * pp_mesh.size()
            pp_group = pp_mesh.get_group()
        dist.all_gather_object(peers, geometry, group=pp_group)
        with synchronize_local_errors(
            "Pipeline input geometry", group=runtime.error_group
        ):
            if any(peer != geometry for peer in peers):
                raise ValueError(
                    "Pipeline ranks disagree on prepared token/target shapes or dtypes"
                )
    return PipelineInputs(args=arg_mbs, kwargs=kwarg_mbs, targets=target_mbs)


def _validate_pipeline_losses(
    losses: list[torch.Tensor] | None, targets: list[torch.Tensor] | None
) -> list[torch.Tensor]:
    """Check schedule results locally after all pipeline work has finished."""
    if losses is None or targets is None or not targets:
        raise ValueError("Final pipeline stage must provide microbatch losses/targets")
    if len(losses) != len(targets):
        raise ValueError("Pipeline schedule must return one loss per target microbatch")
    device = targets[0].device
    for loss, labels in zip(losses, targets, strict=True):
        if labels.device != device:
            raise ValueError("Pipeline target microbatches must share one device")
        validate_loss_sum(loss, device=device)
    return losses


def run_pipeline_step(
    runtime: PipelineRuntime,
    batch: Batch | TrainerBatch,
    *,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Drive one PP schedule step and return its unnormalized local loss sum."""
    with synchronize_local_errors(
        "Pipeline loss normalization validation", group=runtime.error_group
    ):
        validate_normalization_tokens(global_valid_tokens)
    inputs = prepare_pipeline_inputs(runtime, batch)

    losses: list[torch.Tensor] | None = [] if runtime.pp_has_last_stage else None
    with runtime.param_context(), spmd_context(runtime.parallel_dims):
        run_microbatches(
            runtime.pp_schedule,
            args=inputs.args if runtime.pp_has_first_stage else None,
            kwargs=inputs.kwargs,
            targets=inputs.targets,
            losses=losses,
            normalization_tokens=global_valid_tokens,
        )

    # Only final stages own losses, but every update participant enters the
    # boundary after schedule communication, before reporting/reductions begin.
    with synchronize_local_errors(
        "Pipeline training loss result validation", group=runtime.error_group
    ):
        if runtime.pp_has_last_stage:
            losses = _validate_pipeline_losses(losses, inputs.targets)
            for loss in losses:
                if loss.device != global_valid_tokens.device:
                    raise ValueError("Pipeline loss and normalization devices differ")
        else:
            validate_loss_sum(
                runtime.pp_loss_sentinel,
                device=global_valid_tokens.device,
                requires_grad=False,
            )
    if runtime.pp_has_last_stage:
        assert losses is not None  # Validated on the owning rank above.
        # The schedule backpropagated normalized losses. Convert their
        # detached sum to the raw token sum expected by Trainer.train_step.
        detached_losses = [loss.detach() for loss in losses]
        losses.clear()
        return torch.sum(torch.stack(detached_losses)) * global_valid_tokens
    return runtime.pp_loss_sentinel


def run_pipeline_eval(
    runtime: PipelineRuntime, batch: Batch | TrainerBatch
) -> torch.Tensor | None:
    """Run forward-only microbatches; only the final PP stage returns a loss sum.

    Use a denominator of one so losses are summed cross-entropies. The driver
    restores the previous denominator on exit; the common validation loop
    normalizes once at the end.
    """
    inputs = prepare_pipeline_inputs(runtime, batch)
    losses: list[torch.Tensor] | None = [] if runtime.pp_has_last_stage else None
    with runtime.param_context(), spmd_context(runtime.parallel_dims):
        run_eval_microbatches(
            runtime.pp_schedule,
            args=inputs.args if runtime.pp_has_first_stage else None,
            kwargs=inputs.kwargs,
            targets=inputs.targets,
            losses=losses,
            normalization_tokens=torch.ones(
                (), dtype=torch.float32, device=runtime.device
            ),
        )
    with synchronize_local_errors(
        "Pipeline validation loss result validation", group=runtime.error_group
    ):
        if runtime.pp_has_last_stage:
            losses = _validate_pipeline_losses(losses, inputs.targets)
    if losses is None:
        return None
    result = torch.stack([loss.detach() for loss in losses]).sum()
    losses.clear()
    return result
