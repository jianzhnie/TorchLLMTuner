"""Training loop and stable Trainer interface.

``runtime.assembly`` assembles runtime state, ``runtime.batch`` prepares
inputs, ``runtime.pipeline`` drives PP schedules, and ``runtime.validation``
evaluates. This module owns optimizer steps, gradient checks, loss reductions,
checkpoint counters, and the outer training loop.

Step order matters: read every accumulation batch, reduce the whole-batch
valid-token count over DP, run forward/backward with that denominator, clip
and check gradients, update the optimizer and scheduler, then report metrics.
CP and TP partition tokens within a DP batch, so their loss sums are reduced
across the sequence axes before reporting. PP stage losses are synchronized to
all stages. The process group belongs to the entry point, not ``Trainer.close``.
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.validation_config import ValidationConfig

from ..accelerator.collectives import (
    all_reduce,
    set_pg_timeouts,
    synchronize_local_errors,
)
from ..accelerator.spmd import spmd_context
from ..components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from ..components.loss import (
    chunked_lm_head_cross_entropy,
    cross_entropy_loss,
)
from ..components.metrics import MetricsProcessor
from ..components.optimizer.ema import EMA
from ..components.optimizer.lr_scheduler import LRSchedulersContainer
from ..components.profiler import Profiler
from ..datasets.contracts import (
    BaseDataLoader,
    Batch,
    DataloaderExhaustedError,
    TrainerBatch,
)
from ..models.common.aux_loss import AuxLoss
from ..models.common.moe.bias_update import (
    collect_routing_load_metrics,
    reset_routing_statistics,
)
from ..parallel.model_compile import rollback_compile_settings_on_error
from ..parallel.topology import ParallelTopology
from ..utils.garbage_collection import GarbageCollection
from ..utils.logging import get_logger
from ..utils.seeding import rollback_execution_settings_on_error, seed_everything
from .runtime import assembly, gradients, pipeline
from .runtime import batch as batch_runtime
from .runtime import metrics as metrics_runtime
from .runtime import validation as validation_runtime
from .runtime.reductions import resolve_training_reduction_meshes

# Rank-aware: the helper attaches a handler whose filter passes below-ERROR
# lines only on the log ranks (rank 0 by default), so a torchrun run logs one
# line per step instead of one per rank.
logger = get_logger(__name__)

__all__ = ["Trainer"]


class Trainer:
    # The state a built trainer carries, declared here so the shape of the
    # object is readable without reading ``__init__`` -- the reference does the
    # same. Anything built later is annotated with ``| None`` so a ``Trainer``
    # made with ``__new__`` (how the tests exercise the pure helpers without a
    # process group) sees the same "not built yet" state an attribute would
    # give, rather than an AttributeError.
    cfg: LLMTunerConfig
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    parallel_dims: ParallelTopology | None
    mesh: DeviceMesh | None
    model: torch.nn.Module | None
    model_parts: list[torch.nn.Module]
    pp_schedule: Any | None
    pp_has_first_stage: bool
    pp_has_last_stage: bool
    _pp_loss_sentinel: torch.Tensor | None
    _chunked_loss_num_chunks: int
    optimizer: torch.optim.Optimizer
    # None denotes incomplete assembly; data_iterator rejects it explicitly.
    # Defaults also let close() safely handle partial initialization failures.
    dataloader: BaseDataLoader | None = None
    lr_scheduler: LRSchedulersContainer | None
    ema: EMA | None
    checkpointer: DistributedCheckpointManager | None
    metrics: MetricsProcessor | None
    gc_handler: GarbageCollection | None

    # Additional training state, saved in the checkpoint.
    step: int
    ntokens_seen: int

    def __init__(self, cfg: LLMTunerConfig):
        """Assemble the trainer. The whole body lives in ``runtime/assembly.py`` --
        the assembly order there is a contract (see its docstring)."""
        compile_settings = nullcontext()
        if getattr(getattr(cfg, "training", None), "compile", False):
            compile_settings = rollback_compile_settings_on_error()
        with rollback_execution_settings_on_error(), compile_settings:
            try:
                assembly.assemble_trainer_runtime(self, cfg)
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

    # -- setup helpers ---------------------------------------------------------

    @staticmethod
    def seed_everything(
        seed: int, *, deterministic: bool, detect_anomaly: bool = False
    ) -> None:
        """Seed host and device generators through the shared utility."""
        seed_everything(
            seed, deterministic=deterministic, detect_anomaly=detect_anomaly
        )

    # -- batch handling (bodies live in batch.py) -------------------------------

    def dp_rank_world_size(self) -> tuple[int, int]:
        """This rank's position and extent along the dataloading (DP) axis."""
        return batch_runtime.dp_rank_world_size(self)

    def batch_size_per_rank(self, dp_world_size: int) -> int:
        """This rank's share of the global batch. Body in ``batch.py``."""
        return batch_runtime.batch_size_per_rank(self, dp_world_size)

    def build_dataloader(self) -> BaseDataLoader:
        """Build the micro-batch source the config names. Body in ``batch.py``."""
        return batch_runtime.build_dataloader(self)

    def data_iterator(self) -> Iterator[Batch | TrainerBatch]:
        """The raw micro-batch source. Body in ``batch.py``."""
        return batch_runtime.data_iterator(self)

    def batch_generator(
        self, data_iterable: Iterable[Batch | TrainerBatch]
    ) -> Iterator[Batch | TrainerBatch]:
        """Wrap the source with per-fetch accounting. Body in ``batch.py``."""
        return batch_runtime.batch_generator(self, data_iterable)

    @staticmethod
    def count_valid_tokens(batch: Batch | TrainerBatch) -> int:
        """The number of labels that contribute to the loss, pre-shard.

        Body in ``batch.py``.
        """
        return batch_runtime.count_valid_tokens(batch)

    # -- the step, one function per level --------------------------------------

    @property
    def example_model(self):
        """The model a batch is normalized against, present on every PP stage.

        Unlike ``self.model`` (``None`` under PP, where this rank holds several
        chunks and the schedule drives them), every rank keeps a module that can
        run ``preprocess_inputs``: the first stage owns the embedding chunk. The
        PP path already assumes as much -- that is how it builds the schedule.
        """
        return self.model if self.model is not None else self.model_parts[0]

    def microbatch(self, batch: Batch | TrainerBatch) -> dict[str, Any]:
        """Everything one accumulation group's forward/backward needs.

        Body in ``batch.py``.
        """
        return batch_runtime.microbatch(self, batch)

    def to_device(self, batch: Batch | TrainerBatch) -> Batch | TrainerBatch:
        """Move one consumption group's tensors to the training device.

        Body in ``batch.py``.
        """
        return batch_runtime.to_device(self, batch)

    def preprocess(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Ask the model to turn its batch into forward inputs.

        Body in ``batch.py``.
        """
        return batch_runtime.preprocess(self, batch)

    def forward_backward_step(
        self,
        microbatch: dict[str, Any],
        *,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run one accumulation group's forward and backward; return its loss sum.

        A *group*, not a micro-batch: with pipeline parallelism one group is one
        schedule step, which internally drives several micro-batches.

        The returned tensor is the **un-normalized** token sum. The division by
        ``global_valid_tokens`` happens inside this call's graph, so the
        gradients are normalized already; ``train_step`` applies it again when
        it reports. torchtitan normalizes inside its ``loss_fn`` and returns the
        normalized value instead -- the numbers agree either way, but the tensor
        that comes back from here is *not* an average.

        Two bodies, matching torchtitan's split. The PP one takes the raw batch
        and calls ``preprocess_inputs`` itself, once per schedule micro-batch;
        the non-PP one preprocesses here, because it has exactly one.

        ``global_valid_tokens`` is the step's denominator, reduced across the DP
        axis. It is passed in rather than computed here because it must be the
        *same* number for every group of the step -- under gradient
        accumulation the count only exists once all of them have been read, so
        the caller reduces it first and hands it down.
        """
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return self.pp_forward_backward_body(
                microbatch["batch"], global_valid_tokens=global_valid_tokens
            )
        inputs, labels, extra_kwargs = self.preprocess(microbatch["batch"])
        return self._forward_backward_body(
            inputs,
            labels,
            extra_kwargs=extra_kwargs,
            global_valid_tokens=global_valid_tokens,
        )

    def _forward_backward_body(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        extra_kwargs: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # ``spmd_context`` is what makes a process group answerable *by name*
        # (``spmd_mesh_group("tp")`` and friends) for the duration of the body.
        # It is entered here, around the forward/backward only, because that is
        # the region whose components read the ambient mesh -- the optimizer and
        # the checkpointers take their groups as arguments. On a single process
        # it is a no-op, so the same code runs from one device to a full mesh.
        with self.param_context(), spmd_context(self.parallel_dims):
            if self._chunked_loss_num_chunks > 1:
                # Chunked loss: the forward skips lm_head and returns hidden
                # states; lm_head + cross-entropy then run per sequence chunk,
                # so the peak logits memory is 1/num_chunks of the full T*V
                # tensor. The call runs the backward itself (per chunk, scaled
                # by 1/global_valid_tokens -- the same normalization the
                # non-chunked path applies inside the graph) and returns the
                # detached sum.
                hidden_states = self.model(inputs, **extra_kwargs, skip_lm_head=True)
                return chunked_lm_head_cross_entropy(
                    self.model.lm_head,
                    hidden_states,
                    labels,
                    num_chunks=self._chunked_loss_num_chunks,
                    grad_scale=1.0 / global_valid_tokens,
                    **self.loss_vocab_kwargs(),
                )
            logits = self.model(inputs, **extra_kwargs)
            loss_sum = self.loss_sum(logits, labels, **self.loss_vocab_kwargs())
            del logits
            # Normalize BEFORE backward, while the sum is still differentiable.
            # Dividing after backwarding the raw sum would work for a single
            # group but not under accumulation: gradients add, so N groups each
            # having backpropped an un-normalized sum would have to be rescaled
            # afterwards, and the intermediate buffers (`clip_grad_norm_`'s
            # norm) would already have been computed from the wrong values.
            # Dividing here, inside the graph, is also what keeps the backward
            # numerically identical to `loss.backward()` on a pre-divided loss
            # -- it is the division that is linear, not an extra op.
            (loss_sum / global_valid_tokens).backward()
        return loss_sum.detach()

    def loss_vocab_kwargs(self) -> dict[str, Any]:
        """The vocab-parallel arguments for the loss seam, when TP is on.

        Both are handed to :func:`cross_entropy_loss`, which selects the
        vocab-sharded path when the local logits width differs from the global
        vocabulary size. A replicated lm_head takes the plain loss path; a
        TP-sharded head uses the TP group for distributed cross entropy.

        ``global_vocab_size`` is read from the model rather than from
        ``cfg.model.vocab_size``: the head is built against the HF config's
        number (which wins for a hub id and for a local checkpoint directory,
        and which the config field may never have been filled from), and that
        is the only value that keeps a replicated head on the plain path.

        Empty when there is no model in hand (the ``pp > 1`` body drives the
        schedule's own loss, which carries its own copy of these) or no TP axis
        to reduce across.
        """
        tp_mesh = (
            None
            if self.parallel_dims is None or self.model is None
            else self.parallel_dims.get_optional_mesh("tp")
        )
        # ``getattr`` rather than an attribute read: a test double for the
        # model is often a plain module, and the missing vocabulary must fall
        # back to the plain loss rather than raise.
        vocab_size = (
            None if self.model is None else getattr(self.model, "vocab_size", None)
        )
        if tp_mesh is None or vocab_size is None:
            return {}
        return {"tp_group": tp_mesh.get_group(), "global_vocab_size": vocab_size}

    @staticmethod
    def loss_sum(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        tp_group: dist.ProcessGroup | None = None,
        global_vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Summed next-token cross-entropy over the predictable labels.

        ``labels`` arrives already aligned with ``logits`` -- ``logits[t]``
        predicts ``labels[t]``, both sources having done their shift upstream
        (see ``HFTransformerModel.preprocess_inputs``). No shift happens here,
        which is what lets the two sources share one loss: the synthetic path
        slots its rows together and the packed path arrives already shifted, and
        both mark the positions that must not be predicted with ``IGNORE_INDEX``
        rather than dropping them. Those positions are the row ends of the
        synthetic path and the document boundaries and packing padding of the
        packed one.

        Not normalized, and deliberately not told the token count. The
        denominator is a *global* count reduced across DP, which the caller
        owns; the per-rank count that pairs with it is taken upstream from the
        unsharded batch (``count_valid_tokens``) precisely so a loss that has
        since been sliced by context parallelism cannot be recounted. Passing
        the count in here would suggest this function has a use for it, and a
        recount would silently undercount by a factor of ``cp``.

        ``tp_group`` / ``global_vocab_size`` select the vocab-parallel form when
        the logits are a vocab shard, exactly as in
        :func:`cross_entropy_loss`; both default to ``None``, which is the plain
        path a replicated head takes.
        """
        return cross_entropy_loss(
            logits,
            labels,
            tp_group=tp_group,
            global_vocab_size=global_vocab_size,
        )

    # -- pipeline-parallel steps ----------------------------------------------

    def pp_microbatches(
        self, batch: Batch | TrainerBatch
    ) -> list[Batch | TrainerBatch]:
        """Split this rank's batch into whole-row pipeline microbatches."""
        return pipeline.split_pipeline_microbatches(
            batch, num_microbatches=self.cfg.parallel.pp_config.num_pp_microbatches
        )

    def pp_forward_backward_body(
        self,
        batch: Batch | TrainerBatch,
        *,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Drive the pipeline schedule for one accumulation group."""
        return pipeline.run_pipeline_step(
            self, batch, global_valid_tokens=global_valid_tokens
        )

    def before_gradient_clipping(self) -> None:
        """Run diagnostics after gradient reduction and before clipping."""

    def param_context(self):
        """Context shared by dense/PP forward-backward and validation.

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
        self.optimizer.zero_grad(set_to_none=True)
        reset_routing_statistics(self.model_parts)

        # Snapshot the lr *before* the schedule advances below. The value
        # reported for a step must be the one the optimizer applied during it;
        # reading after ``lr_scheduler.step()`` would report the next step's
        # value and, on the last step, one past the end of the schedule.
        with synchronize_local_errors("Training step reporting setup"):
            lr_metrics = self.lr_scheduler.get_metrics()
            should_log = self.should_log()

        # Count replicated loader tokens over DP only; sum sequence-sharded
        # losses over DP/CP/TP. runtime/reductions.py owns these mesh choices.
        # PP stages report independently; only the final stage's loss is logged.
        parallel_dims = self.parallel_dims
        reduction_meshes = resolve_training_reduction_meshes(parallel_dims)
        dp_mesh = reduction_meshes.dp_mesh

        # Read the entire accumulation window on the host before the first
        # forward: every group's gradient uses the same global token denominator.
        # Counting must precede CP/TP sequence sharding and PP microbatch splits.
        microbatches: deque[dict[str, Any]] = deque()
        local_valid_tokens = 0
        local_routing_tokens = 0
        for _ in range(self.cfg.gradient_accumulation_steps):
            # Account against the original loader batch; transfer at consumption.
            microbatch = self.microbatch(next(data_iterator))
            self.metrics.add_valid_tokens(microbatch["num_valid_tokens"])
            microbatches.append(microbatch)
            local_valid_tokens += microbatch["num_valid_tokens"]
            local_routing_tokens += microbatch["num_routing_tokens"]

        # Keep the count on device so normalizing the loss adds no device sync
        # to the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens, dtype=torch.int64, device=self.device
        )
        global_valid_tokens = local_valid_tokens_tensor
        if dp_mesh is not None:
            # Clone before the in-place collective: the local count is read
            # again below for this rank's own per-rank average.
            global_valid_tokens = global_valid_tokens.clone()
            all_reduce(global_valid_tokens, group=dp_mesh.get_group())

        # A fully masked accumulation window has no average loss. Assert the
        # DP-reduced count on every rank before entering forward/backward, so
        # no stage updates on an undefined (zero-divided) gradient. The device
        # assertion stays asynchronous where supported; NPU uses an explicit
        # scalar read because torch-npu 2.12 lacks the native assertion kernel.
        gradients.check_step_condition(
            global_valid_tokens > 0,
            "Training step has zero valid tokens; cannot normalize the loss.",
        )

        # Router losses cover every non-padding input token, including SFT
        # prompts whose labels are masked from the main cross-entropy loss.
        if AuxLoss.has_pending_counts():
            global_routing_tokens = torch.tensor(
                local_routing_tokens, dtype=torch.int64, device=self.device
            )
            if dp_mesh is not None:
                all_reduce(global_routing_tokens, group=dp_mesh.get_group())
            AuxLoss.set_step_denominator(
                global_routing_tokens, model_parts=self.model_parts
            )

        # Forward can mutate checkpointed state too: running buffers and
        # embedding max_norm weights are examples. Finish the previous snapshot
        # before model execution; background upload may still overlap training.
        staging_failure: BaseException | None = None
        try:
            self.checkpointer.maybe_wait_for_staging()
        except BaseException as error:
            staging_failure = error
        # Waiting can involve background communication; agree only after the
        # wait returns, before any rank starts model execution.
        with synchronize_local_errors("Training checkpoint staging"):
            if staging_failure is not None:
                raise staging_failure

        # Process each group, then free it. Loss values are retained only on a
        # logging step: backward has already consumed them, and non-logging
        # steps need only the on-device finiteness verdict. On logging steps,
        # take ownership of the first detached value and accumulate later
        # groups in place, avoiding a list plus a final stack proportional to
        # ``gradient_accumulation_steps``.
        accumulated_loss: torch.Tensor | None = None
        # int32 is supported by NCCL reductions, unlike bool.
        loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        while microbatches:
            microbatch = microbatches.popleft()
            # HSDP accumulates locally until the final group, then all-reduces
            # the accumulated gradient once for this optimizer step.
            if parallel_dims is not None and parallel_dims.dp_replicate_enabled:
                is_last = not microbatches
                for part in self.model_parts:
                    part.set_requires_all_reduce(is_last)
            detached_loss = self.forward_backward_step(
                microbatch, global_valid_tokens=global_valid_tokens
            )
            del microbatch
            local_loss = (
                detached_loss.to_local()
                if isinstance(detached_loss, DTensor)
                else detached_loss
            )
            loss_is_finite.logical_and_(torch.isfinite(local_loss).all())
            if should_log:
                if accumulated_loss is None:
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        # FSDP exclusions retain ordinary local gradients after accumulation.
        gradients.allreduce_fsdp_excluded_grads(self)
        # After the last backward, before clipping: replicated parameters under
        # TP hold token-partial gradients that nothing else reduces.
        gradients.allreduce_replicated_tp_grads(self)

        grad_norm = gradients.clip_and_check_finite(
            self,
            reduction_meshes=reduction_meshes,
            loss_is_finite=loss_is_finite,
        )

        routing_metrics = (
            collect_routing_load_metrics(
                self.model_parts, self.parallel_dims, self.device
            )
            if should_log
            else {}
        )
        self.optimizer.step()
        # Advance the scheduler after the optimizer uses the current rate.
        # Agree on failure before EMA or reporting collectives. The optimizer
        # update is already committed and cannot be rolled back here.
        with synchronize_local_errors("Training scheduler update"):
            self.lr_scheduler.step()
        with synchronize_local_errors("Training EMA update"):
            if self.ema is not None:
                # EMA validates fixed layouts and updates only local storage.
                # All ranks enter, including PP stages with no tracked tensors.
                self.ema.step(self.step)

        if not should_log:
            return None

        assert accumulated_loss is not None

        # All accumulation groups have contributed their raw token loss here.
        # Dividing by the step's global valid-token count reports the same
        # average that normalized the gradients, independent of DP rank and
        # accumulation boundaries.
        result = metrics_runtime.build_step_metrics(
            self,
            accumulated_loss=accumulated_loss,
            global_valid_tokens=global_valid_tokens,
            local_valid_tokens=local_valid_tokens,
            local_valid_tokens_tensor=local_valid_tokens_tensor,
            reduction_meshes=reduction_meshes,
            grad_norm=grad_norm,
            lr_metrics=lr_metrics,
        )
        result.update(routing_metrics)
        with synchronize_local_errors("Training resource observation"):
            result.update(metrics_runtime.collect_resource_metrics(self))
        return result

    # -- validation (bodies live in runtime/validation.py) ----------------------

    @staticmethod
    def check_validation_feasibility(
        validation: ValidationConfig,
        *,
        dp_world_size: int,
        training_dataset: str,
        chunked_loss_num_chunks: int = 1,
    ) -> None:
        """Reject the validation configurations that cannot terminate cleanly.

        The body lives in ``runtime/validation.py``; see there for the rationale.
        """
        validation_runtime.check_validation_feasibility(
            validation,
            dp_world_size=dp_world_size,
            training_dataset=training_dataset,
            chunked_loss_num_chunks=chunked_loss_num_chunks,
        )

    def should_validate(self, step: int) -> bool:
        """Whether a validation pass runs at the end of ``step``.

        The body lives in ``runtime/validation.py``; see there for the gating rule.
        """
        return validation_runtime.should_validate(self, step)

    def validate(self, step: int) -> None:
        """Run one eval-mode, gradient-free pass and log its loss.

        The body lives in ``runtime/validation.py`` (gradient-free via its own
        ``torch.no_grad``); see there for the reporting contract.
        """
        validation_runtime.validate(self, step)

    # -- step cadence ------------------------------------------------------------

    def should_log(self) -> bool:
        # Delegated rather than reimplemented: the metrics processor also
        # guarantees the first step logs, and two copies of that rule would
        # drift the moment one of them changed.
        return self.metrics.should_log(self.step)

    def should_continue_training(self) -> bool:
        return self.step < self.cfg.steps

    # -- checkpoint state -------------------------------------------------------
    # The manager serializes ``states[TRAIN_STATE]`` (this object) alongside the
    # model and optimizer. These two counters are the whole of that state, and
    # they are here rather than in the checkpoint dict because the running
    # trainer is what has to be mutated back into a resumed step.

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate checkpoint counters without changing the current run."""
        step = state_dict["step"]
        ntokens_seen = state_dict["ntokens_seen"]
        for name, value in (("step", step), ("ntokens_seen", ntokens_seen)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"trainer {name} must be a nonnegative integer")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    # -- the loop ---------------------------------------------------------------

    def train(self) -> None:
        try:
            loaded = self.checkpointer.load(self.cfg.checkpoint.load_step)
            with synchronize_local_errors("Training resume reporting"):
                if loaded:
                    logger.info(f"Resuming from step {self.step}")

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
            with synchronize_local_errors("Training input initialization"):
                data_iterator = self.batch_generator(self.data_iterator())
            # Entered around the loop rather than around a single step: the
            # torch profiler's schedule counts iterations across the whole run
            # and only dumps a trace at the end of a cycle, so a per-step
            # context would never reach one. Left open when profiling is off --
            # the Profiler holds no handles in that case.
            with ExitStack() as stack:
                # Local tracing setup can fail after peers acquired handles.
                # ExitStack closes those handles if the shared boundary raises.
                with synchronize_local_errors("Training profiler initialization"):
                    profiler = stack.enter_context(
                        Profiler(
                            self.cfg.profiler,
                            global_step=self.step,
                            base_folder=self.cfg.dump_folder,
                        )
                    )
                # Takes the cyclic collector over from CPython for the duration
                # of training: the default schedule fires at unpredictable
                # times, often inside a forward, and walking a multi-gigabyte
                # object graph there is pure stall. Built here rather than in
                # ``__init__`` because that is when the model and optimizer
                # exist, so the first collection already covers the real graph.
                with synchronize_local_errors("Training GC initialization"):
                    self.gc_handler = GarbageCollection(gc_freq=self.cfg.gc_freq)
                while self.should_continue_training():
                    self.step += 1
                    with synchronize_local_errors("Training GC step"):
                        self.gc_handler.run(self.step)

                    try:
                        observer = getattr(self, "training_observer", None)
                        if observer is not None:
                            observer.begin()
                        step_metrics = self.train_step(data_iterator)
                        if observer is not None:
                            observations = observer.end()
                            if step_metrics is not None:
                                step_metrics.update(observations)
                    except DataloaderExhaustedError:
                        # The step is abandoned rather than trained on a
                        # partial batch: a batch's worth of tokens either all
                        # contribute to a gradient or none of them do.
                        # ``self.step`` counts *completed* optimizer steps, so
                        # the abandoned one is given back. torchtitan's
                        # ``num_completed_steps`` only advances at the end of a
                        # successful update, and a counter that kept this step
                        # would make ``state_dict`` resume past a step whose
                        # weights were never updated.
                        self.step -= 1
                        with synchronize_local_errors("Training exhaustion reporting"):
                            logger.warning(
                                "Ran out of data; the last step was canceled."
                            )
                        # Finite data can end before cfg.steps and off the
                        # checkpoint interval. Preserve the last completed
                        # update using the configured final-save policy.
                        saved = self.step > 0 and self.checkpointer.save(
                            self.step, last_step=True
                        )
                        with synchronize_local_errors(
                            "Training exhausted checkpoint reporting"
                        ):
                            if saved:
                                logger.info(f"Saved checkpoint for step {self.step}")
                        break

                    # Reporting is local, but a sink can fail on just one
                    # rank. Agree before any peer starts checkpoint collectives.
                    with synchronize_local_errors("Training metric reporting"):
                        if step_metrics is not None:
                            self.metrics.log(
                                self.step,
                                global_avg_loss=step_metrics["loss"],
                                global_max_loss=step_metrics["max_loss"],
                                grad_norm=step_metrics["grad_norm"],
                                extra_metrics={
                                    k: v
                                    for k, v in step_metrics.items()
                                    if k not in ("loss", "max_loss", "grad_norm")
                                },
                            )

                    # The manager owns the interval policy: ``save`` decides for
                    # itself whether this step is a checkpointing step. The final
                    # step is forced so a run that ends off-interval still leaves
                    # a resumable artifact rather than only a mid-run one.
                    last_step = self.step == self.cfg.steps
                    saved = self.checkpointer.save(self.step, last_step=last_step)
                    with synchronize_local_errors("Training checkpoint reporting"):
                        if saved:
                            logger.info(f"Saved checkpoint for step {self.step}")
                            if observer is not None:
                                self.metrics.log_resources(
                                    self.step,
                                    self.checkpointer.observation_metrics.copy(),
                                )

                    # Validation, after the checkpoint save and before the
                    # profiler advances -- the reference's order. The pass is
                    # eval-only and leaves no state behind, so its position in
                    # the step affects only which step's weights it scores.
                    if self.should_validate(self.step):
                        self.validate(self.step)

                    # Advances the schedule. After the save, so the profiler's
                    # active iteration covers an ordinary step rather than one
                    # that also wrote a checkpoint.
                    with synchronize_local_errors("Training profiler step"):
                        profiler.step()

                    if self.step == first_step_of_this_process:
                        # Startup is finished on this process; from here a long
                        # wait is a stall, not a slow launch. Skipped entirely on
                        # a single process: it has no group to time out, and its
                        # barrier would be the only collective in the program.
                        if self.parallel_dims is not None:
                            set_pg_timeouts(
                                timedelta(
                                    seconds=self.cfg.parallel.train_timeout_seconds
                                ),
                                self.parallel_dims,
                                device=self.device,
                            )
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

    def close(self) -> None:
        """Release trainer resources in dependency order.

        Drain checkpoint writes while the loader they may reference is alive,
        then close reporting and data resources. Restore process-wide GC state
        last. Successful releases are not repeated; failed owners remain for
        retry, including during partial initialization. Attempt every release
        before propagating the first cleanup error.
        """
        first_error: BaseException | None = None
        for name in ("checkpointer", "metrics", "dataloader", "gc_handler"):
            resource = getattr(self, name, None)
            if resource is None:
                continue
            setattr(self, name, None)
            try:
                resource.close()
            except BaseException as error:
                if name in ("metrics", "dataloader") or (
                    name == "checkpointer" and getattr(resource, "_initialized", False)
                ):
                    # Failed sinks, loader workers, and interrupted checkpoint
                    # waits may retain resources. Keep their owner for a retry.
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
