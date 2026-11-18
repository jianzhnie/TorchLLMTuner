"""Validation passes, token/loss reductions, and temporary loader lifecycle.

Trainer delegates evaluation here. Model input preparation is shared with
training; this module owns validation scheduling, reporting, and mode restore.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from typing import TYPE_CHECKING

import torch

from llmtuner.config.validation_config import ValidationConfig
from llmtuner.parallel.compatibility import (
    chunked_loss_validation,
    validation_once_requires_dp1,
    validation_once_requires_finite_corpus,
)
from llmtuner.utils.logging import get_logger
from llmtuner.utils.seeding import preserve_rng_state

from ...accelerator.collectives import (
    all_reduce,
    synchronize_after_wait,
    synchronize_local_errors,
)
from ...datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
)
from .batch import (
    build_rank_dataloader,
    fetch_batch,
    validated_batch_token_counts,
)
from .reductions import resolve_loss_reduction_meshes

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

    from ...components.metrics import MetricsProcessor
    from ...config.llmtuner_config import LLMTunerConfig
    from ...parallel.topology import ParallelTopology
    from .engine import TrainingEngine


class ValidationDataLoaderResources:
    """Own temporary loaders until their close succeeds, including pass failure.

    The caller retains this owner across passes and closes it during shutdown.
    A new pass drains pending owners before constructing another loader.
    """

    def __init__(self) -> None:
        self._pending: list[BaseDataLoader] = []

    def acquire(self, loader: BaseDataLoader) -> None:
        if any(owned is loader for owned in self._pending):
            raise ValueError("Validation loader is already owned")
        self._pending.append(loader)

    def _release(self, loader: BaseDataLoader) -> None:
        loader.close()
        self._pending = [owned for owned in self._pending if owned is not loader]

    def close_dataloader(self, loader: BaseDataLoader, *, failed: bool) -> None:
        """Keep failed releases without replacing an active validation error."""
        try:
            self._release(loader)
        except BaseException:
            if not failed:
                raise
            try:
                get_logger(__name__).exception(
                    "Validation dataloader cleanup failed after validation failed"
                )
            except BaseException:
                pass

    def close(self) -> None:
        first_error: BaseException | None = None
        for loader in reversed(tuple(self._pending)):
            try:
                self._release(loader)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def create_validation_dataloader(
    cfg: LLMTunerConfig,
    parallel_dims: ParallelTopology | None,
    validation: ValidationConfig,
) -> BaseDataLoader:
    """The pass's temporary loader: fresh per pass, closed when the pass ends.

    ``steps=-1`` reads it to exhaustion (built with repeat=False); a positive
    ``steps`` bounds the pass (built repeating, so the bound is reachable).
    """
    return build_rank_dataloader(
        cfg,
        parallel_dims=parallel_dims,
        repeat=validation.steps != -1,
        dataset=validation.dataset,
    )


def count_validation_batch_tokens(
    batch: Batch | TrainerBatch,
    *,
    metrics: MetricsProcessor,
    token_counter: Callable[[Batch | TrainerBatch], int],
    device: torch.device,
    dp_mesh: DeviceMesh | None,
    error_group: ProcessGroup | None = None,
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
    with synchronize_local_errors("Validation token accounting", group=error_group):
        num_tokens, local_valid_tokens = validated_batch_token_counts(
            batch, token_counter=token_counter
        )
        metrics.add_tokens(num_tokens)
        global_valid_tokens = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=device
        )
    if dp_mesh is not None:
        all_reduce(global_valid_tokens, group=dp_mesh.get_group())
    return global_valid_tokens


def finalize_validation(
    *,
    metrics: MetricsProcessor,
    error_group: ProcessGroup | None = None,
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
    average to report.
    """
    # Preflight and reporting are local phases. Loss reduction remains outside
    # those boundaries, since a collective failure cannot be coordinated here.
    with synchronize_local_errors("Validation result preflight", group=error_group):
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
        if accumulated_loss is None:
            raise ValueError("Validation batches did not produce a loss sum")
        global_loss_sum = accumulated_loss.clone()
    if loss_mesh is not None:
        all_reduce(global_loss_sum, group=loss_mesh.get_group())
    # Coordinate reporting errors before any rank proceeds to the next step.
    with synchronize_local_errors("Validation result reporting", group=error_group):
        global_avg_loss = float(global_loss_sum) / num_global_valid_tokens
        if not math.isfinite(global_avg_loss):
            raise FloatingPointError(
                f"Validation loss is not finite at step {step}: {global_avg_loss}"
            )
        metrics.log_validation(loss=global_avg_loss, step=step)


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
        chunked_loss_validation(chunked_loss_num_chunks)
    if validation.steps != -1:
        return
    if dp_world_size > 1:
        validation_once_requires_dp1(dp_world_size)
    dataset = training_dataset if validation.dataset is None else validation.dataset
    if dataset == "random":
        validation_once_requires_finite_corpus()


def should_validate(validation: ValidationConfig | None, step: int) -> bool:
    """Whether a validation pass runs at the end of ``step``.

    Step 1 always validates (a run sees its first eval number immediately,
    which is the cheap sanity check that the eval path works at all);
    after that, every ``validation.freq`` steps.
    """
    return validation is not None and (step == 1 or step % validation.freq == 0)


@torch.no_grad()
def validate(
    engine: TrainingEngine,
    *,
    cfg: LLMTunerConfig,
    metrics: MetricsProcessor,
    token_counter: Callable[[Batch | TrainerBatch], int],
    loader_resources: ValidationDataLoaderResources,
    step: int,
    wait_for_staging: Callable[[], None] | None = None,
) -> None:
    """Evaluate without gradients, restoring modes, gradients and mask policy.

    Report summed next-token cross-entropy divided by the global valid-token
    count, using the same DP token and loss-mesh reductions as training.
    Optimizer, scheduler, and training token counters
    are unchanged; the temporary loader is closed after each pass.

    ``steps=-1`` reads a finite corpus once; positive steps use a repeating
    loader. Empty passes and passes without valid tokens raise explicit errors.
    """
    validation = cfg.validation
    assert validation is not None, "validate() is gated by should_validate"

    # Validation follows save() in the training loop. Eval forward can still
    # mutate state (for example embedding max_norm); finish the snapshot first.
    synchronize_after_wait(
        wait_for_staging, "Validation checkpoint staging", group=engine.error_group
    )

    model_contexts = ExitStack()
    with ExitStack() as resources:
        # State capture is local and may fail before mode setup. Peers must not
        # begin validation forward until every rank owns its restoration state.
        with synchronize_local_errors("Validation RNG setup", group=engine.error_group):
            training_modes = {
                module: module.training
                for part in engine.model_parts
                for module in part.modules()
            }
            parameter_gradients = {
                parameter: parameter.grad
                for part in engine.model_parts
                for parameter in part.parameters()
            }
            resources.enter_context(preserve_rng_state(engine.device))
        try:
            with synchronize_local_errors(
                "Validation mode setup", group=engine.error_group
            ):
                mask_type = None
                if validation.dataset is not None:
                    mask_type = cfg.dataloader.attention_mask_type(
                        dataset=validation.dataset
                    )
                    if (
                        mask_type == "block_causal"
                        and cfg.parallel.cp_size > 1
                        and cfg.parallel.cp_config.context_parallel_strategy == "kcp"
                    ):
                        raise ValueError(
                            "KCP does not support packed document validation"
                        )
                for part in dict.fromkeys(engine.model_parts):
                    context = getattr(part, "validation_context", None)
                    if callable(context):
                        model_contexts.enter_context(context(mask_type=mask_type))
                    elif mask_type is not None:
                        raise TypeError(
                            f"{type(part).__name__} must implement "
                            "validation_context() to change validation datasets"
                        )
                for part in engine.model_parts:
                    part.eval()
            run_validation_loop(
                engine,
                cfg=cfg,
                metrics=metrics,
                token_counter=token_counter,
                loader_resources=loader_resources,
                validation=validation,
                step=step,
            )
        finally:
            try:
                for parameter, gradient in parameter_gradients.items():
                    parameter.grad = gradient
            finally:
                try:
                    model_contexts.close()
                finally:
                    for module, training in training_modes.items():
                        module.training = training
        # Only a successful pass reaches this boundary. On pass failure the
        # stack unwinds normally, preserving the original error without new
        # communication that peers may no longer reach.
        rng_failure: BaseException | None = None
        try:
            resources.close()
        except BaseException as error:
            rng_failure = error
        with synchronize_local_errors(
            "Validation RNG cleanup", group=engine.error_group
        ):
            if rng_failure is not None:
                raise rng_failure


def run_validation_loop(
    engine: TrainingEngine,
    *,
    cfg: LLMTunerConfig,
    metrics: MetricsProcessor,
    token_counter: Callable[[Batch | TrainerBatch], int],
    validation: ValidationConfig,
    loader_resources: ValidationDataLoaderResources,
    step: int,
) -> None:
    """Read, score, and finalize one validation pass for the local model."""
    parallel_dims = engine.parallel_dims
    dp_mesh, loss_mesh = resolve_loss_reduction_meshes(parallel_dims)
    validation_dataloader: BaseDataLoader | None = None
    try:
        # These operations are local. Coordinate before fetch_batch enters its
        # world collective, and retain successful peer loaders for cleanup.
        with synchronize_local_errors(
            "Validation input initialization", group=engine.error_group
        ):
            loader_resources.close()
            validation_dataloader = create_validation_dataloader(
                cfg, engine.parallel_dims, validation
            )
            loader_resources.acquire(validation_dataloader)
            total_global_valid_tokens = torch.zeros(
                (), dtype=torch.int64, device=engine.device
            )
            data_iterator = iter(validation_dataloader)
        accumulated_loss: torch.Tensor | None = None
        num_steps = 0
        while validation.steps == -1 or num_steps < validation.steps:
            try:
                batch = fetch_batch(data_iterator, group=engine.error_group)
            except (DataloaderExhaustedError, StopIteration):
                break
            global_valid_tokens = count_validation_batch_tokens(
                batch,
                metrics=metrics,
                token_counter=token_counter,
                device=engine.device,
                dp_mesh=dp_mesh,
                error_group=engine.error_group,
            )
            loss_sum = engine.evaluate(batch).detach()
            if accumulated_loss is None:
                accumulated_loss = loss_sum.clone()
            else:
                accumulated_loss.add_(loss_sum)
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
            else synchronize_local_errors(
                "Validation input cleanup", group=engine.error_group
            )
        ):
            if validation_dataloader is not None:
                loader_resources.close_dataloader(validation_dataloader, failed=failed)

    finalize_validation(
        metrics=metrics,
        error_group=engine.error_group,
        num_steps=num_steps,
        accumulated_loss=accumulated_loss,
        total_global_valid_tokens=total_global_valid_tokens,
        loss_mesh=loss_mesh,
        step=step,
    )
