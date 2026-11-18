"""Training loop and stable Trainer interface.

``runtime.assembly`` builds explicit services, ``runtime.engine`` executes
model and update steps, and ``runtime.validation`` evaluates. This module owns
token accounting, loss reporting, checkpoint counters, and the outer loop.

Step order matters: read every accumulation batch, reduce the whole-batch
valid-token count over DP, run forward/backward with that denominator, clip
and check gradients, update the optimizer and scheduler, then report metrics.
CP and TP partition tokens within a DP batch, so their loss sums are reduced
across the sequence axes before reporting. The process group belongs to the
entry point, not ``Trainer.close``.
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.llmtuner_config import LLMTunerConfig

from ..accelerator.collectives import (
    all_reduce,
    set_pg_timeouts,
    synchronize_after_wait,
    synchronize_local_errors,
)
from ..components.checkpointer.core.base import BaseCheckpointManager
from ..components.metrics import MetricsProcessor
from ..components.optimizer.ema import EMA
from ..components.optimizer.lr_scheduler import LRSchedulersContainer
from ..components.optimizer.optimizer import OptimizersContainer
from ..datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
)
from ..memory.activation_swap import (
    ActivationSwapResources,
)
from ..parallel.fully_shard_parallel.resources import FSDPCommunicationResources
from ..parallel.model_compile import rollback_compile_settings_on_error
from ..parallel.topology import ParallelTopology
from ..utils.garbage_collection import GarbageCollection
from ..utils.logging import get_logger, set_log_ranks
from ..utils.seeding import rollback_execution_settings_on_error
from .runtime import assembly, engine
from .runtime import batch as batch_runtime
from .runtime import metrics as metrics_runtime
from .runtime import validation as validation_runtime
from .runtime.builders import ProfilerOwner, TrainingBuilders
from .runtime.environment import build_runtime_environment
from .runtime.gradients import check_step_condition
from .runtime.reductions import (
    logical_tokens_per_rank,
    resolve_training_reduction_meshes,
)
from .runtime.results import StepResult, TrainingProgress
from .runtime.training_observer import (
    DispatchTimerResources,
    TrainingObserver,
    install_dispatch_timers,
)

# Rank-aware: the helper attaches a handler whose filter passes below-ERROR
# lines only on the log ranks (rank 0 by default), so a torchrun run logs one
# line per step instead of one per rank.
logger = get_logger(__name__)

__all__ = ["Trainer"]


class Trainer:
    # Resource defaults support cleanup after partial assembly. Acquired owners
    # are assigned to the instance before dependent construction begins.
    cfg: LLMTunerConfig
    _builders: TrainingBuilders | None
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    parallel_dims: ParallelTopology | None
    mesh: DeviceMesh | None
    model: torch.nn.Module | None
    model_parts: list[torch.nn.Module]
    training_engine: engine.TrainingEngine
    error_group: dist.ProcessGroup | None = None
    optimizer: OptimizersContainer | None = None
    # None denotes incomplete assembly; data_iterator rejects it explicitly.
    # Defaults also let close() safely handle partial initialization failures.
    dataloader: BaseDataLoader | None = None
    lr_scheduler: LRSchedulersContainer | None = None
    ema: EMA | None = None
    checkpointer: BaseCheckpointManager | None = None
    metrics: MetricsProcessor | None = None
    gc_handler: GarbageCollection | None = None
    activation_swap_resources: ActivationSwapResources | None = None
    fsdp_communication_resources: FSDPCommunicationResources | None = None
    dispatch_timer_resources: DispatchTimerResources | None = None
    validation_resources: validation_runtime.ValidationDataLoaderResources | None = None
    profiler_resource: ProfilerOwner | None = None
    training_observer: TrainingObserver | None = None

    # Created lazily for callers that construct a Trainer with __new__.
    # Each instance owns its counters; checkpoints do not retain the Trainer.
    _progress: TrainingProgress | None = None

    @property
    def progress(self) -> TrainingProgress:
        if self._progress is None:
            self._progress = TrainingProgress()
        return self._progress

    @property
    def step(self) -> int:
        return self.progress.step

    @step.setter
    def step(self, value: int) -> None:
        self.progress.step = value

    @property
    def ntokens_seen(self) -> int:
        return self.progress.ntokens_seen

    @ntokens_seen.setter
    def ntokens_seen(self, value: int) -> None:
        self.progress.ntokens_seen = value

    def __init__(
        self, cfg: LLMTunerConfig, *, builders: TrainingBuilders | None = None
    ):
        """Build runtime services in lifecycle order, cleaning up partial failures."""
        self._builders = builders
        compile_settings = nullcontext()
        if getattr(getattr(cfg, "training", None), "compile", False):
            compile_settings = rollback_compile_settings_on_error()
        with rollback_execution_settings_on_error(), compile_settings:
            try:
                self._assemble_runtime(cfg)
            except BaseException:
                # Failed construction never returns an object to the caller, so
                # release resources already acquired by partial assembly here.
                try:
                    self.close()
                except BaseException:
                    try:
                        logger.exception(
                            "Resource cleanup failed during trainer construction"
                        )
                    except BaseException:
                        pass  # Construction must retain the original assembly error.
                raise

    def _retain_model_resources(self, model: torch.nn.Module) -> None:
        """Keep partially transformed models alive for construction cleanup."""
        if self.fsdp_communication_resources is None:
            self.fsdp_communication_resources = FSDPCommunicationResources(())
        if self.activation_swap_resources is None:
            self.activation_swap_resources = ActivationSwapResources(())
        for owner in (
            self.fsdp_communication_resources,
            self.activation_swap_resources,
        ):
            if all(part is not model for part in owner.model_parts):
                owner.model_parts += (model,)

    def _assemble_runtime(self, cfg: LLMTunerConfig) -> None:
        """Acquire services in order and retain each owner before the next build.

        Builders return explicit values. Assigning lifecycle resources here lets
        close() release partial construction without passing this Trainer into
        a builder or introducing a second resource owner.
        """
        self.cfg = cfg
        environment = build_runtime_environment(cfg)
        self.rank = environment.rank
        self.world_size = environment.world_size
        self.local_rank = environment.local_rank
        self.parallel_dims = environment.parallel_dims
        self.device = environment.device
        self.mesh = environment.mesh
        error_group = environment.error_group
        self.error_group = error_group

        factories = self._builders or TrainingBuilders()
        loss_functions = assembly.build_loss_functions(
            cfg, factory=factories.loss, error_group=error_group
        )
        prepared_model = assembly.prepare_parallel_model(
            cfg,
            environment=environment,
            on_model_acquired=self._retain_model_resources,
        )
        model_assembly = prepared_model.assembly
        self.model = model_assembly.model
        self.model_parts = model_assembly.model_parts
        # Successful handoff retains only final local model parts.
        self.fsdp_communication_resources = FSDPCommunicationResources(
            tuple(self.model_parts)
        )
        self.activation_swap_resources = ActivationSwapResources(
            tuple(self.model_parts)
        )
        optimizer_stack = assembly.build_optimizer_stack(
            cfg,
            model_parts=model_assembly.model_parts,
            layouts=model_assembly.layouts,
            builders=factories,
            parallel_dims=environment.parallel_dims,
            error_group=error_group,
            on_service_acquired=lambda name, service: setattr(self, name, service),
        )
        self.training_engine = assembly.create_training_engine(
            cfg,
            model_assembly=model_assembly,
            optimizer_stack=optimizer_stack,
            parallel_dims=environment.parallel_dims,
            device=environment.device,
            param_context=self.param_context,
            loss_functions=loss_functions,
            error_group=error_group,
        )

        # The checkpointer captures the loader cursor alongside model state.
        with synchronize_local_errors("Dataloader construction", group=error_group):
            self.dataloader = batch_runtime.build_rank_dataloader(
                cfg, parallel_dims=environment.parallel_dims
            )

        # The checkpointer and public counters share this progress owner.
        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = assembly.build_checkpointer(
            cfg,
            prepared_model.config,
            model_assembly=model_assembly,
            optimizer_stack=optimizer_stack,
            dataloader=self.dataloader,
            train_state=self.progress,
            device=environment.device,
            error_group=error_group,
            builders=factories,
        )

        assembly.build_metrics_processor(
            cfg,
            prepared_model.config,
            parallel_dims=environment.parallel_dims,
            num_parameters=prepared_model.num_parameters,
            num_trainable_parameters=prepared_model.num_trainable_parameters,
            error_group=error_group,
            builders=factories,
            on_service_acquired=lambda name, service: setattr(self, name, service),
        )
        self.training_observer = TrainingObserver(
            self.metrics, environment.device, group=error_group
        )
        if cfg.metrics.measure_moe_dispatch:
            with synchronize_local_errors(
                "Dispatch observer installation", group=error_group
            ):
                self.dispatch_timer_resources = install_dispatch_timers(
                    model_assembly.model_parts
                )
            if not self.dispatch_timer_resources.timers:
                logger.warning(
                    "Dispatch timing requested, but no model dispatch boundary "
                    "was found; dispatch timing is unavailable for this model."
                )
        if cfg.activation_swap:
            self.activation_swap_resources.reset_metrics()
        set_log_ranks(cfg.metrics.log_ranks)
        if environment.device.type == "npu":
            logger.info(
                "NPU step guards use explicit synchronous scalar reads; "
                "loss/gradient and valid-token checks remain enabled"
            )

    # -- batch handling (bodies live in batch.py) -------------------------------

    def data_iterator(self) -> Iterator[Batch | TrainerBatch]:
        """Iterate the assembled data source without an extra forwarding layer."""
        if self.dataloader is None:
            raise RuntimeError("Trainer dataloader has not been initialized")
        return iter(self.dataloader)

    def batch_generator(
        self, data_iterable: Iterable[Batch | TrainerBatch]
    ) -> Iterator[Batch | TrainerBatch]:
        """Wrap the source with per-fetch accounting. Body in ``batch.py``."""
        return batch_runtime.batch_generator(
            data_iterable, metrics=self.metrics, group=self.error_group
        )

    @staticmethod
    def count_valid_tokens(batch: Batch | TrainerBatch) -> int:
        """Count loss-bearing labels before sequence sharding."""
        return batch_runtime.count_valid_tokens(batch)

    # -- the step, one function per level --------------------------------------

    def microbatch(
        self, batch: Batch | TrainerBatch
    ) -> batch_runtime.AccumulationBatch:
        """Everything one accumulation group's forward/backward needs.

        Body in ``batch.py``.
        """
        prepared, num_tokens = batch_runtime.prepare_microbatch(
            batch,
            valid_token_counter=self.count_valid_tokens,
            error_group=self.error_group,
        )
        # The loader batch is replicated across CP/TP. Commit this rank's
        # share only after every local counter has passed group validation.
        self.ntokens_seen += logical_tokens_per_rank(num_tokens, self.parallel_dims)
        return prepared

    def forward_backward_step(
        self,
        microbatch: batch_runtime.AccumulationBatch,
        *,
        global_valid_tokens: torch.Tensor,
    ) -> StepResult:
        """Execute one accumulation group through the assembled engine."""
        return self.training_engine.forward_backward(
            microbatch["batch"],
            global_valid_tokens=global_valid_tokens,
            valid_tokens=microbatch["num_valid_tokens"],
        )

    def before_gradient_clipping(self) -> None:
        """Run diagnostics after gradient reduction and before clipping."""

    def param_context(self) -> AbstractContextManager[None]:
        """Context shared by forward-backward and validation.

        The default is empty. Diagnostic integrations wrap this hook to control
        output capture across all execution paths. Activation checkpointing is
        installed by the parallel layer, outside this context.
        """
        return nullcontext()

    def train_step(
        self, data_iterator: Iterator[Batch | TrainerBatch]
    ) -> dict[str, float] | None:
        """One optimizer step. Returns the metrics to log, or ``None`` if not logging.

        The ordering mirrors torchtitan's and it is the whole content of this
        function: zero the gradients, snapshot the learning rate, read *every*
        micro-batch the step will consume, reduce the token count those batches
        imply, then run the forward/backward groups, clip, check finiteness,
        step the optimizer, and finally normalize and reduce the loss for
        reporting. Reads that must agree across every micro-batch of a step --
        the denominator and the lr snapshot -- are taken up front, before any
        of them is consumed.

        With ``gradient_accumulation_steps > 1`` the loop below runs the whole
        forward/backward once per group, and the optimizer advances once at the
        end. Each group's loss is still divided by the *step's* global token
        count, not its own, so the accumulated gradient is the step's summed
        loss over the step's token total -- the same quantity a single group
        would produce if the batch had not been split.
        """
        # ``set_to_none=True`` is what the reference uses whenever CUDA graphs
        # are off, and llmtuner runs no graph path: freeing the gradient buffers
        # outright rather than zeroing them in place is both cheaper and what
        # makes the accumulated-gradient bookkeeping below trivially correct.
        self.training_engine.begin_step()

        # Snapshot the lr *before* the schedule advances below. The value
        # reported for a step must be the one the optimizer applied during it;
        # reading after ``lr_scheduler.step()`` would report the next step's
        # value and, on the last step, one past the end of the schedule.
        with synchronize_local_errors(
            "Training step reporting setup", group=self.error_group
        ):
            lr_metrics = self.lr_scheduler.get_metrics()
            should_log = self.should_log()

        # Count replicated loader tokens over DP only; sum sequence-sharded
        # losses over DP/CP/TP. runtime/reductions.py owns these mesh choices.
        parallel_dims = self.parallel_dims
        reduction_meshes = resolve_training_reduction_meshes(parallel_dims)
        dp_mesh = reduction_meshes.dp_mesh

        # Read the entire accumulation window on the host before the first
        # forward: every group's gradient uses the same global token denominator.
        # Counting must precede CP/TP sequence sharding.
        microbatches: deque[batch_runtime.AccumulationBatch] = deque()
        local_valid_tokens = 0
        local_routing_tokens = 0
        for _ in range(self.cfg.gradient_accumulation_steps):
            # Account against the original loader batch; transfer at consumption.
            microbatch = self.microbatch(next(data_iterator))
            self.metrics.add_valid_tokens(microbatch["num_valid_tokens"])
            microbatches.append(microbatch)
            local_valid_tokens += microbatch["num_valid_tokens"]
            local_routing_tokens += microbatch["num_routing_tokens"]
            # The queue owns pending batches; release the loop's extra reference
            # so popping a consumed batch also releases its host tensors.
            del microbatch

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        global_valid_tokens = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        if dp_mesh is not None:
            all_reduce(global_valid_tokens, group=dp_mesh.get_group())

        # A fully masked accumulation window has no average loss. Assert the
        # DP-reduced count on every rank before entering forward/backward, so
        # no stage updates on an undefined (zero-divided) gradient. The device
        # assertion stays asynchronous where supported; NPU uses an explicit
        # scalar read because torch-npu 2.12 lacks the native assertion kernel.
        check_step_condition(
            global_valid_tokens > 0,
            "Training step has zero valid tokens; cannot normalize the loss.",
        )

        self.training_engine.prepare_auxiliary_loss(
            local_routing_tokens,
            dp_group=None if dp_mesh is None else dp_mesh.get_group(),
        )

        # Forward can mutate checkpointed state too: running buffers and
        # embedding max_norm weights are examples. Finish the previous snapshot
        # before model execution; background upload may still overlap training.
        synchronize_after_wait(
            self.checkpointer.maybe_wait_for_staging,
            "Training checkpoint staging",
            group=self.error_group,
        )

        def executed_groups() -> Iterator[StepResult]:
            while microbatches:
                microbatch = microbatches.popleft()
                self.training_engine.prepare_accumulation_group(
                    is_last=not microbatches
                )
                result = self.forward_backward_step(
                    microbatch, global_valid_tokens=global_valid_tokens
                )
                del microbatch
                yield result

        accumulation = self.training_engine.accumulate_results(
            executed_groups(), retain_loss=should_log
        )

        update_result = self.training_engine.finish_step(
            self.step,
            accumulation=accumulation,
            collect_metrics=should_log,
            before_gradient_clipping=self.before_gradient_clipping,
        )

        if not should_log:
            return None

        # All accumulation groups have contributed their raw token loss here.
        # Dividing by the step's global valid-token count reports the same
        # average that normalized the gradients, independent of DP rank and
        # accumulation boundaries.
        update_result = metrics_runtime.collect_update_metrics(
            update_result,
            parallel_dims=self.parallel_dims,
            model_parts=self.model_parts,
        )
        result = metrics_runtime.build_step_metrics(
            ntokens_seen=self.ntokens_seen,
            device=self.device,
            accumulation=accumulation,
            global_valid_tokens=global_valid_tokens,
            local_valid_tokens=local_valid_tokens,
            reduction_meshes=reduction_meshes,
            update=update_result,
            lr_metrics=lr_metrics,
        )
        with synchronize_local_errors(
            "Training resource observation", group=self.error_group
        ):
            result.update(
                metrics_runtime.collect_resource_metrics(
                    activation_swap=(
                        self.activation_swap_resources
                        if self.cfg.activation_swap
                        else None
                    ),
                    optimizer=self.optimizer,
                    checkpointer=self.checkpointer,
                    dispatch_timers=(
                        self.dispatch_timer_resources.timers
                        if self.dispatch_timer_resources is not None
                        else ()
                    ),
                )
            )
        return result

    # -- validation (bodies live in runtime/validation.py) ----------------------

    def should_validate(self, step: int) -> bool:
        """Whether a validation pass runs at the end of ``step``.

        The body lives in ``runtime/validation.py``; see there for the gating rule.
        """
        return validation_runtime.should_validate(self.cfg.validation, step)

    def validate(self, step: int) -> None:
        """Run one eval-mode, gradient-free pass and log its loss.

        The body lives in ``runtime/validation.py`` (gradient-free via its own
        ``torch.no_grad``); see there for the reporting contract.
        """
        if self.validation_resources is None:
            self.validation_resources = (
                validation_runtime.ValidationDataLoaderResources()
            )
        validation_runtime.validate(
            self.training_engine,
            loader_resources=self.validation_resources,
            cfg=self.cfg,
            metrics=self.metrics,
            token_counter=self.count_valid_tokens,
            step=step,
            wait_for_staging=(
                self.checkpointer.maybe_wait_for_staging
                if self.checkpointer is not None
                else None
            ),
        )

    # -- step cadence ------------------------------------------------------------

    def should_log(self) -> bool:
        # Delegated rather than reimplemented: the metrics processor also
        # guarantees the first step logs, and two copies of that rule would
        # drift the moment one of them changed.
        return self.metrics.should_log(self.step)

    def should_continue_training(self) -> bool:
        return self.step < self.cfg.steps

    # -- the loop ---------------------------------------------------------------

    def train(self) -> None:
        try:
            loaded = self.checkpointer.load(self.cfg.checkpoint.load_step)
            with synchronize_local_errors(
                "Training resume reporting", group=self.error_group
            ):
                if loaded:
                    logger.info(f"Resuming from step {self.step}")

            self._run_training()
        finally:
            # Teardown lives in ``close`` rather than inline so a caller that
            # drives the trainer programmatically -- rather than through
            # ``train`` -- gets the same cleanup from the same place. It is in
            # a ``finally`` so a run that dies mid-loop still finishes the
            # checkpoint it had already started writing.
            training_error = sys.exc_info()[1]
            try:
                self.close()
            except BaseException:
                if training_error is None:
                    raise
                try:
                    logger.exception("Resource cleanup failed after a training error")
                except BaseException:
                    # Reporting must not replace the active training failure.
                    pass

    def _run_training(self) -> None:
        """Initialize loop-scoped services and run optimizer steps."""

        # The step this process's *first* train step lands on. Startup work --
        # process groups, the model build, the first collective, compile --
        # runs under the long process-group timeout those groups were created
        # with, because that is what genuinely takes minutes. Once one step
        # has completed, that work is behind this process, so the timeout can
        # be lowered to the one a real stall should be measured against (see
        # ``set_pg_timeouts``). Relative to the loaded step rather than
        # absolute ``1`` so a resumed run lowers it too.
        first_step_of_this_process = self.step + 1

        # ``batch_generator`` wraps the bare source so every fetch carries
        # its token and loading-time accounting; the loop below sees only
        # batches. One exception type crosses that boundary.
        with synchronize_local_errors(
            "Training input initialization", group=self.error_group
        ):
            data_iterator = self.batch_generator(self.data_iterator())
        # Entered around the loop rather than around a single step: the
        # torch profiler's schedule counts iterations across the whole run
        # and only dumps a trace at the end of a cycle, so a per-step
        # context would never reach one. Left open when profiling is off --
        # the Profiler holds no handles in that case.
        with ExitStack() as stack:
            # Local tracing setup can fail after peers acquired handles.
            # ExitStack closes those handles if the shared boundary raises.
            with synchronize_local_errors(
                "Training profiler initialization", group=self.error_group
            ):
                builders = getattr(self, "_builders", None)
                profiler_context = assembly.build_profiler(
                    self.cfg.profiler,
                    global_step=self.step,
                    base_folder=self.cfg.dump_folder,
                    factory=None if builders is None else builders.profiler,
                )
                # Own cleanup before entry, including partially started traces.
                self.profiler_resource = profiler_context
                profiler = stack.enter_context(profiler_context)
                if not callable(getattr(profiler, "step", None)):
                    raise TypeError("Entered profiler must provide step()")
            # Takes the cyclic collector over from CPython for the duration
            # of training: the default schedule fires at unpredictable
            # times, often inside a forward, and walking a multi-gigabyte
            # object graph there is pure stall. Built here rather than in
            # ``__init__`` because that is when the model and optimizer
            # exist, so the first collection already covers the real graph.
            with synchronize_local_errors(
                "Training GC initialization", group=self.error_group
            ):
                self.gc_handler = GarbageCollection(gc_freq=self.cfg.gc_freq)
            while self.should_continue_training():
                self.step += 1
                with synchronize_local_errors(
                    "Training GC step", group=self.error_group
                ):
                    self.gc_handler.run(self.step)

                try:
                    observer = self.training_observer
                    if observer is not None:
                        observer.begin()
                    step_metrics = self.train_step(data_iterator)
                    if observer is not None:
                        observations = observer.end()
                        if step_metrics is not None:
                            step_metrics.update(observations)
                except DataloaderExhaustedError:
                    self._cancel_incomplete_step()
                    break

                self._report_completed_step(step_metrics, observer=observer)

                # Advances the schedule. After the save, so the profiler's
                # active iteration covers an ordinary step rather than one
                # that also wrote a checkpoint.
                with synchronize_local_errors(
                    "Training profiler step", group=self.error_group
                ):
                    profiler.step()

                if self.step == first_step_of_this_process:
                    # Startup is finished on this process; from here a long
                    # wait is a stall, not a slow launch. Skipped entirely on
                    # a single process: it has no group to time out, and its
                    # barrier would be the only collective in the program.
                    if self.parallel_dims is not None:
                        set_pg_timeouts(
                            timedelta(seconds=self.cfg.parallel.train_timeout_seconds),
                            self.parallel_dims,
                            device=self.device,
                        )

    def _cancel_incomplete_step(self) -> None:
        """Restore the completed-step counter and save its latest state."""
        # An accumulation window is atomic: none of an incomplete step's
        # batches may contribute to an optimizer update or checkpoint step.
        self.step -= 1
        with synchronize_local_errors(
            "Training exhaustion reporting", group=self.error_group
        ):
            logger.warning("Ran out of data; the last step was canceled.")
        saved = self.step > 0 and self.checkpointer.save(self.step, last_step=True)
        with synchronize_local_errors(
            "Training exhausted checkpoint reporting", group=self.error_group
        ):
            if saved:
                logger.info(f"Saved checkpoint for step {self.step}")

    def _report_completed_step(
        self,
        step_metrics: dict[str, float] | None,
        *,
        observer: TrainingObserver | None,
    ) -> None:
        """Log, checkpoint and validate one completed update in that order."""
        # Agree on local logging success before peers enter checkpoint collectives.
        with synchronize_local_errors(
            "Training metric reporting", group=self.error_group
        ):
            if step_metrics is not None:
                self.metrics.log(
                    self.step,
                    global_avg_loss=step_metrics["loss"],
                    global_max_loss=step_metrics["max_loss"],
                    grad_norm=step_metrics["grad_norm"],
                    extra_metrics={
                        key: value
                        for key, value in step_metrics.items()
                        if key not in ("loss", "max_loss", "grad_norm")
                    },
                )

        saved = self.checkpointer.save(self.step, last_step=self.step == self.cfg.steps)
        with synchronize_local_errors(
            "Training checkpoint reporting", group=self.error_group
        ):
            if saved:
                logger.info(f"Saved checkpoint for step {self.step}")
                if observer is not None:
                    self.metrics.log_resources(
                        self.step, self.checkpointer.observation_metrics.copy()
                    )

        if self.should_validate(self.step):
            self.validate(self.step)

    def close(self) -> None:
        """Release trainer resources in dependency order.

        Drain checkpoint writes while the loader they may reference is alive,
        then close reporting and data resources. Restore process-wide GC state
        last. Successful releases are not repeated; failed owners remain for
        retry, including during partial initialization. Release independent
        resources even after failure, but keep the loader alive while an
        interrupted checkpoint drain may still reference it.
        """
        first_error: BaseException | None = None
        # A retained checkpointer may still read source tensors and loader state.
        # Retained FSDP communication may still reduce/offload optimizer tensors.
        # Release independent resources; retry dependents after their owners drain.
        for name, dependencies in (
            ("profiler_resource", ()),
            ("checkpointer", ()),
            ("dispatch_timer_resources", ()),
            ("fsdp_communication_resources", ("checkpointer",)),
            (
                "activation_swap_resources",
                ("checkpointer", "fsdp_communication_resources"),
            ),
            ("validation_resources", ()),
            ("metrics", ()),
            ("dataloader", ("checkpointer",)),
            ("ema", ("checkpointer", "fsdp_communication_resources")),
            ("lr_scheduler", ("checkpointer",)),
            (
                "optimizer",
                ("checkpointer", "fsdp_communication_resources", "lr_scheduler"),
            ),
            ("gc_handler", ()),
        ):
            if any(getattr(self, owner, None) is not None for owner in dependencies):
                continue
            resource = getattr(self, name, None)
            if resource is None:
                continue
            try:
                close = getattr(resource, "close", None)
                if name in ("ema", "lr_scheduler", "optimizer") and not callable(close):
                    # Tensor-only services need no shutdown, but their ownership
                    # must end so dependent resources can be released below.
                    setattr(self, name, None)
                    continue
                setattr(self, name, None)
                close()
            except BaseException as error:
                # A failed close leaves ownership indeterminate. Keep every
                # resource owner for retry, including injected checkpoint
                # managers that do not expose a retry capability flag.
                setattr(self, name, resource)
                if first_error is None:
                    first_error = error
                try:
                    logger.exception("Failed to close trainer resource %s", name)
                except BaseException:
                    # Continue releasing resources even when the sink is broken.
                    pass
        if first_error is not None:
            raise first_error
