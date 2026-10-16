"""Validation passes, token/loss reductions, and temporary loader lifecycle.

Trainer delegates evaluation here. Pipeline input preparation is shared with
training; this module owns validation scheduling, reporting, and mode restore.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from llmtuner.accelerator.capabilities import require
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.parallel import compatibility
from llmtuner.utils.logging import get_logger
from llmtuner.utils.seeding import preserve_rng_state

from ...accelerator.collectives import all_reduce, synchronize_local_errors
from ...accelerator.spmd import spmd_context
from ...datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
)
from ...datasets.loader_builder import build_dataloader
from .batch import fetch_batch
from .pipeline import prepare_pipeline_inputs
from .reductions import resolve_loss_reduction_meshes

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..trainer import Trainer


def close_validation_dataloader(loader: BaseDataLoader, *, failed: bool) -> None:
    """Close temporary data resources without replacing an active pass error."""
    try:
        loader.close()
    except BaseException:
        if not failed:
            raise
        get_logger(__name__).exception(
            "Validation dataloader cleanup failed after validation failed"
        )


def create_validation_dataloader(
    self: Trainer, validation: ValidationConfig
) -> BaseDataLoader:
    """The pass's temporary loader: fresh per pass, closed when the pass ends.

    ``steps=-1`` reads it to exhaustion (built with repeat=False); a positive
    ``steps`` bounds the pass (built repeating, so the bound is reachable).
    """
    dp_rank, dp_world_size = self.dp_rank_world_size()
    batch_size_per_rank = self.batch_size_per_rank(dp_world_size)
    return build_dataloader(
        self.cfg,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        num_tokens_per_batch=batch_size_per_rank * self.cfg.max_seq_len,
        repeat=validation.steps != -1,
        dataset=validation.dataset,
    )


def count_validation_batch_tokens(
    self: Trainer, batch: Batch | TrainerBatch, *, dp_mesh: DeviceMesh | None
) -> torch.Tensor:
    """Throughput accounting plus the dp-reduced global valid-token count.

    The throughput number counts every label the loader produced, whether or
    not it is predictable; ``ntokens_seen`` is deliberately not touched -- it
    is the checkpointed *training* counter. The valid-token count is taken
    from the unsharded batch, exactly as in training, so the dp-axis reduction
    counts the whole batch once even when CP later slices the sequence.
    """
    # Counting and metric updates are local and may fail on only one rank.
    # Agree on success before any peer enters the DP token collective.
    with synchronize_local_errors("Validation token accounting"):
        labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
        self.metrics.add_tokens(labels.numel())
        local_valid_tokens = self.count_valid_tokens(batch)
        global_valid_tokens = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
    if dp_mesh is not None:
        all_reduce(global_valid_tokens, group=dp_mesh.get_group())
    return global_valid_tokens


def finalize_validation(
    self: Trainer,
    *,
    num_steps: int,
    accumulated_loss: torch.Tensor | None,
    total_global_valid_tokens: torch.Tensor,
    loss_mesh: DeviceMesh | None,
    step: int,
) -> None:
    """Reduce the pass's sums and report the average, with loud empty-pass errors.

    Two outcomes are loud errors rather than a silently skipped report: a pass
    that read zero batches (the source is empty or processing and filtering
    leave no usable samples on this rank),
    and a pass over zero valid tokens (every label masked), which has no
    average to report. Under PP only the last stage holds losses; other stages
    run the identical loop and coordinate reporting errors without emitting metrics.
    """
    # Preflight and reporting are local phases. Loss reduction remains outside
    # those boundaries, since a collective failure cannot be coordinated here.
    with synchronize_local_errors("Validation result preflight"):
        if num_steps == 0:
            raise ValueError(
                "Validation ran zero batches on this rank. Check that the "
                "validation source is nonempty and that processing and "
                "filtering leave usable samples. Finite text packing retains "
                "and pads partial final batches."
            )
        num_global_valid_tokens = int(total_global_valid_tokens.item())
        if num_global_valid_tokens == 0:
            raise ValueError(
                "Validation ran on zero valid tokens; cannot compute an "
                "average validation loss. Ensure the validation batches "
                "contain unmasked labels."
            )
        global_loss_sum = (
            accumulated_loss.clone() if accumulated_loss is not None else None
        )
    if global_loss_sum is not None and loss_mesh is not None:
        all_reduce(global_loss_sum, group=loss_mesh.get_group())
    # Non-final PP stages must participate even though only the final stage
    # checks and logs the loss; otherwise a sink failure strands their next step.
    with synchronize_local_errors("Validation result reporting"):
        if global_loss_sum is not None:
            global_avg_loss = float(global_loss_sum) / num_global_valid_tokens
            if not math.isfinite(global_avg_loss):
                raise FloatingPointError(
                    f"Validation loss is not finite at step {step}: {global_avg_loss}"
                )
            self.metrics.log_validation(loss=global_avg_loss, step=step)


def check_validation_feasibility(
    validation: ValidationConfig,
    *,
    dp_world_size: int,
    training_dataset: str,
    chunked_loss_num_chunks: int = 1,
) -> None:
    """Reject infeasible validation before models and loaders are built.

    Chunked training loss cannot use this full-logits evaluation path. A finite
    pass (``steps=-1``) requires DP=1 to avoid unequal shard exhaustion, and a
    real corpus because the synthetic random source never ends.
    """
    if chunked_loss_num_chunks > 1:
        compatibility.chunked_loss_validation(chunked_loss_num_chunks)
    if validation.steps != -1:
        return
    if dp_world_size > 1:
        compatibility.validation_once_requires_dp1(dp_world_size)
    dataset = training_dataset if validation.dataset is None else validation.dataset
    if dataset == "random":
        compatibility.validation_once_requires_finite_corpus()


def should_validate(self: Trainer, step: int) -> bool:
    """Whether a validation pass runs at the end of ``step``.

    Step 1 always validates (a run sees its first eval number immediately,
    which is the cheap sanity check that the eval path works at all);
    after that, every ``validation.freq`` steps.
    """
    validation = self.cfg.validation
    return validation is not None and (step == 1 or step % validation.freq == 0)


@torch.no_grad()
def validate(self: Trainer, step: int) -> None:
    """Evaluate without gradients, restoring model modes and mask policy.

    Report summed next-token cross-entropy divided by the global valid-token
    count, using the same DP token and loss-mesh reductions as training. Only
    the last PP stage reports. Optimizer, scheduler, and training token counters
    are unchanged; the temporary loader is closed after each pass.

    ``steps=-1`` reads a finite corpus once; positive steps use a repeating
    loader. Empty passes and passes without valid tokens raise explicit errors.
    """
    validation = self.cfg.validation
    assert validation is not None, "validate() is gated by should_validate"

    # Validation follows save() in the training loop. Eval forward can still
    # mutate state (for example embedding max_norm); finish the snapshot first.
    if self.checkpointer is not None:
        self.checkpointer.maybe_wait_for_staging()

    training_modes = {
        module: module.training
        for part in self.model_parts
        for module in part.modules()
    }
    mask_configs = {
        id(config): config
        for part in self.model_parts
        if (config := getattr(getattr(part, "model", None), "config", None)) is not None
    }
    original_masks = {
        key: (
            hasattr(config, "attn_mask_type"),
            getattr(config, "attn_mask_type", None),
        )
        for key, config in mask_configs.items()
    }
    kernel_modes = {
        kernel: kernel.packed
        for module in training_modes
        if (kernel := getattr(module, "_titan_flex_kernel", None)) is not None
        and hasattr(kernel, "packed")
    }
    with preserve_rng_state(self.device):
        try:
            if validation.dataset is not None:
                mask_type = self.cfg.dataloader.attention_mask_type(
                    dataset=validation.dataset
                )
                if (
                    mask_type == "block_causal"
                    and self.cfg.parallel.cp_size > 1
                    and self.cfg.parallel.cp_config.context_parallel_strategy == "kcp"
                ):
                    raise ValueError("KCP does not support packed document validation")
                for config in mask_configs.values():
                    config.attn_mask_type = mask_type
                for kernel in kernel_modes:
                    kernel.packed = mask_type == "block_causal"
            for part in self.model_parts:
                part.eval()
            run_validation_loop(self, validation, step)
        finally:
            for kernel, packed in kernel_modes.items():
                kernel.packed = packed
            for key, config in mask_configs.items():
                had_mask, original_mask = original_masks[key]
                if not had_mask:
                    if hasattr(config, "attn_mask_type"):
                        delattr(config, "attn_mask_type")
                else:
                    config.attn_mask_type = original_mask
            for module, training in training_modes.items():
                module.training = training


def run_validation_loop(self: Trainer, validation: ValidationConfig, step: int) -> None:
    """Read, score, and finalize one validation pass for dense or PP models."""
    parallel_dims = self.parallel_dims
    pipeline = parallel_dims is not None and parallel_dims.pp_enabled
    if pipeline:
        require(
            "pipelining_microbatch_drivers",
            feature="validation with pipeline parallelism",
        )
    dp_mesh, loss_mesh = resolve_loss_reduction_meshes(parallel_dims)
    validation_dataloader: BaseDataLoader | None = None
    try:
        # These operations are local. Coordinate before fetch_batch enters its
        # world collective, and retain successful peer loaders for cleanup.
        with synchronize_local_errors("Validation input initialization"):
            validation_dataloader = create_validation_dataloader(self, validation)
            total_global_valid_tokens = torch.zeros(
                (), dtype=torch.int64, device=self.device
            )
            data_iterator = iter(validation_dataloader)
        accumulated_loss: torch.Tensor | None = None
        num_steps = 0
        while validation.steps == -1 or num_steps < validation.steps:
            try:
                batch = fetch_batch(data_iterator)
            except (DataloaderExhaustedError, StopIteration):
                break
            global_valid_tokens = count_validation_batch_tokens(
                self, batch, dp_mesh=dp_mesh
            )
            if pipeline:
                loss_sum = evaluate_pipeline_batch(self, batch)
            else:
                inputs, labels, extra_kwargs = self.preprocess(batch)
                with self.param_context(), spmd_context(parallel_dims):
                    logits = self.example_model(inputs, **extra_kwargs)
                    loss_sum = self.loss_sum(logits, labels, **self.loss_vocab_kwargs())
            if loss_sum is not None:
                if accumulated_loss is None:
                    accumulated_loss = loss_sum.detach().clone()
                else:
                    accumulated_loss.add_(loss_sum.detach())
            total_global_valid_tokens.add_(global_valid_tokens)
            num_steps += 1
    finally:
        failed = sys.exc_info()[0] is not None
        # A successful loop proceeds to loss collectives. Agree on local close
        # success first; during failure unwinding preserve the original error
        # without introducing communication that peers may no longer reach.
        with (
            nullcontext()
            if failed
            else synchronize_local_errors("Validation input cleanup")
        ):
            if validation_dataloader is not None:
                close_validation_dataloader(validation_dataloader, failed=failed)

    finalize_validation(
        self,
        num_steps=num_steps,
        accumulated_loss=accumulated_loss,
        total_global_valid_tokens=total_global_valid_tokens,
        loss_mesh=loss_mesh,
        step=step,
    )


def evaluate_pipeline_batch(
    self: Trainer, batch: Batch | TrainerBatch
) -> torch.Tensor | None:
    """Run forward-only microbatches; only the final PP stage returns a loss sum.

    Pin the schedule denominator to one so its losses are summed cross-entropies.
    The common validation loop normalizes once at the end; training republishes
    its denominator before the next optimizer step.
    """
    inputs = prepare_pipeline_inputs(self, batch)
    losses: list[torch.Tensor] | None = [] if self.pp_has_last_stage else None
    with self.param_context(), spmd_context(self.parallel_dims):
        self.pp_schedule._llmtuner_global_valid_tokens = torch.ones(
            (), dtype=torch.float32
        )
        self.pp_schedule.eval(
            arg_mbs=inputs.args if self.pp_has_first_stage else None,
            kwarg_mbs=inputs.kwargs,
            target_mbs=inputs.targets,
            losses=losses,
        )
    if losses is None:
        return None
    return torch.stack([loss.detach() for loss in losses]).sum()
