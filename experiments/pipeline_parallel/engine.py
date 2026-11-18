"""Model execution independent of the Trainer lifecycle.

The engine owns forward/backward dispatch, parallel gradient synchronization,
routing state, gradient clipping and optimizer/scheduler/EMA updates. Data
iteration, reporting cadence and checkpointing remain Trainer responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from torch.distributed import ProcessGroup

from experiments.pipeline_parallel import gradients
from experiments.pipeline_parallel import runtime as pipeline
from experiments.pipeline_parallel.execution import (
    ExecutionLayout,
    validate_local_pipeline_layouts,
)
from experiments.pipeline_parallel.reductions import TrainingReductionMeshes
from experiments.pipeline_parallel.topology import ParallelTopology
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.spmd import spmd_context
from llmtuner.components.loss import chunked_lm_head_cross_entropy, cross_entropy_loss
from llmtuner.components.loss_contract import (
    ChunkedLossBackward,
    LossSum,
    validate_loss_sum,
    validate_normalization_tokens,
)
from llmtuner.datasets.contracts import Batch, TrainerBatch
from llmtuner.models.common.moe.bias_update import collect_routing_load_metrics
from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.trainer.runtime.batch import prepare_model_inputs
from llmtuner.trainer.runtime.engine import _TrainingStepOperations
from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from llmtuner.trainer.runtime.results import (
    AccumulationResult,
    StepResult,
    UpdateResult,
)

ParamContext = Callable[[], AbstractContextManager[Any]]


def loss_vocab_kwargs(
    parallel_dims: ParallelTopology | None,
    *,
    layout: ExecutionLayout | None = None,
) -> dict[str, Any]:
    """Bind loss reduction to declared output ownership, never module attributes."""
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
    if layout is not None:
        return layout.loss_kwargs(None if tp_mesh is None else tp_mesh.get_group())
    if tp_mesh is not None:
        raise ValueError("TP loss requires an explicit ExecutionLayout")
    return {}


@dataclass(slots=True)
class TrainingEngine(_TrainingStepOperations):
    """Execute forward/backward with fixed model and parallel dependencies."""

    model_parts: list[torch.nn.Module]
    parallel_dims: ParallelTopology | None
    device: torch.device
    chunked_loss_num_chunks: int
    param_context: ParamContext
    execution_layouts: tuple[ExecutionLayout, ...] = ()
    loss_sum: LossSum = cross_entropy_loss
    chunked_loss_backward: ChunkedLossBackward | None = chunked_lm_head_cross_entropy
    optimizer: Any = None
    lr_scheduler: Any = None
    ema: Any = None
    pp_schedule: Any | None = None
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False
    pp_loss_sentinel: torch.Tensor | None = None
    pp_num_microbatches: int = 1
    # All update participants, including PP stages without tracked EMA tensors.
    # None preserves the default-group behavior for standalone callers.
    error_group: ProcessGroup | None = None
    max_norm: float = 1.0
    grad_norm_precision: str = "float32"
    expert_reduce_layout: str = "storage"

    _run_forward_backward: Callable[..., torch.Tensor] = field(
        init=False, repr=False, compare=False
    )
    _run_evaluation: Callable[[Batch | TrainerBatch], torch.Tensor | None] = field(
        init=False, repr=False, compare=False
    )
    _dense_backward: Callable[..., torch.Tensor] = field(
        init=False, repr=False, compare=False
    )
    _output_capabilities: ModelExecutionCapabilities = field(
        init=False, repr=False, compare=False
    )
    _chunked_projection: torch.nn.Module | None = field(
        init=False, default=None, repr=False, compare=False
    )

    _loss_kwargs: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.chunked_loss_num_chunks > 1 and self.chunked_loss_backward is None:
            raise ValueError("Configured loss does not support chunked backward")
        # Execution policy belongs to this assembly. Rebuild the engine after
        # changing topology or loss mode rather than selecting it on each batch.
        pipeline_enabled = (
            self.parallel_dims is not None and self.parallel_dims.pp_enabled
        )
        if not self.model_parts:
            raise ValueError("Training execution requires at least one model part")
        if not pipeline_enabled and len(self.model_parts) != 1:
            raise ValueError("Dense execution requires exactly one model part")
        if pipeline_enabled != (self.pp_schedule is not None):
            raise ValueError("Pipeline topology and schedule must be enabled together")
        if self.chunked_loss_num_chunks > 1:
            if pipeline_enabled:
                raise ValueError(
                    "Chunked loss cannot execute with pipeline parallelism"
                )
            structure = getattr(self.model, "decoder_structure", None)
            if not isinstance(structure, DecoderStructure):
                raise TypeError("Chunked loss requires a model-owned DecoderStructure")
            self._chunked_projection = self.model.get_submodule(structure.head)
            if isinstance(self._chunked_projection, torch.nn.Identity):
                raise ValueError("Chunked loss requires an owned output projection")
        if self.execution_layouts:
            if len(self.execution_layouts) != len(self.model_parts):
                raise ValueError("Execution layouts must match local model parts")
            expected_axes = tuple(
                axis
                for axis in ("cp", "tp")
                if self.parallel_dims is not None
                and getattr(self.parallel_dims, f"{axis}_enabled")
            )
            for part, layout in zip(
                self.model_parts, self.execution_layouts, strict=True
            ):
                layout.validate_model(part)
                if layout.sequence_shard_axes != expected_axes:
                    raise ValueError("Engine topology disagrees with sequence layout")
                if layout.chunked_loss != (self.chunked_loss_num_chunks > 1):
                    raise ValueError(
                        "Engine loss strategy disagrees with assembled output layout"
                    )
                if not pipeline_enabled and layout.output != "logits":
                    raise ValueError(
                        "Dense engine requires a logits-producing model part"
                    )
            if pipeline_enabled:
                validate_local_pipeline_layouts(
                    self.execution_layouts,
                    has_first_stage=self.pp_has_first_stage,
                    has_last_stage=self.pp_has_last_stage,
                )
        capabilities = (
            tuple(layout.capabilities for layout in self.execution_layouts)
            if self.execution_layouts
            else tuple(
                getattr(part, "execution_capabilities", None)
                for part in self.model_parts
            )
        )
        if not capabilities or any(
            not isinstance(value, ModelExecutionCapabilities) for value in capabilities
        ):
            raise TypeError(
                "Every model part must declare valid execution capabilities"
            )
        self._output_capabilities = capabilities[0]
        if not pipeline_enabled and not self._output_capabilities.logits:
            raise ValueError("Dense execution requires model-declared logits output")
        if (
            self.chunked_loss_num_chunks > 1
            and not self._output_capabilities.hidden_states
        ):
            raise ValueError("Chunked loss requires model-declared hidden output")
        # PP schedules own their loss binding; intermediate parts return hidden
        # activations and must never be bound to the dense LM loss.
        self._loss_kwargs = (
            {}
            if pipeline_enabled
            else loss_vocab_kwargs(
                self.parallel_dims,
                layout=self.execution_layouts[0] if self.execution_layouts else None,
            )
        )
        self._run_forward_backward = (
            partial(pipeline.run_pipeline_step, self)
            if pipeline_enabled
            else self._forward_backward_batch
        )
        self._run_evaluation = (
            partial(pipeline.run_pipeline_eval, self)
            if pipeline_enabled
            else self._evaluate_dense
        )
        self._dense_backward = (
            self._backward_chunked
            if self.chunked_loss_num_chunks > 1
            else self._backward_logits
        )

    def prepare_accumulation_group(self, *, is_last: bool) -> None:
        """HSDP reduces accumulated gradients only on the final backward group."""
        if self.parallel_dims is not None and self.parallel_dims.dp_replicate_enabled:
            for part in self.model_parts:
                part.set_requires_all_reduce(is_last)

    def synchronize_gradients(self) -> None:
        """Finish excluded and replicated gradient reductions before clipping."""
        gradients.allreduce_fsdp_excluded_grads(
            self.model_parts, parallel_dims=self.parallel_dims
        )
        allreduce_replicated_tp_grads(
            self.model_parts, parallel_dims=self.parallel_dims
        )

    def routing_metrics(self) -> dict[str, float]:
        """Collect model routing observations only when the caller requests a log."""
        return collect_routing_load_metrics(
            self.model_parts, self.parallel_dims, self.device
        )

    def preprocess(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return prepare_model_inputs(
            self.model_parts[0],
            batch,
            device=self.device,
            parallel_dims=self.parallel_dims,
            error_group=self.error_group,
        )

    def pp_microbatches(
        self, batch: Batch | TrainerBatch
    ) -> list[Batch | TrainerBatch]:
        return pipeline.split_pipeline_microbatches(
            batch, num_microbatches=self.pp_num_microbatches
        )

    def forward_backward(
        self,
        batch: Batch | TrainerBatch,
        *,
        global_valid_tokens: torch.Tensor,
        valid_tokens: int,
    ) -> StepResult:
        """Return detached loss and the caller-owned token accounting context."""
        loss_sum = self._run_forward_backward(
            batch, global_valid_tokens=global_valid_tokens
        )
        with synchronize_local_errors(
            "Training step result validation", group=self.error_group
        ):
            return StepResult(
                loss_sum=loss_sum.detach(),
                valid_tokens=valid_tokens,
                normalization_tokens=global_valid_tokens.detach(),
            )

    def evaluate(self, batch: Batch | TrainerBatch) -> torch.Tensor | None:
        """Return raw validation loss; non-final PP stages return None.

        The caller owns eval/no-grad mode, RNG restoration and reporting.
        """
        return self._run_evaluation(batch)

    def _evaluate_dense(self, batch: Batch | TrainerBatch) -> torch.Tensor:
        inputs, labels, extra_kwargs = self.preprocess(batch)
        with self.param_context(), spmd_context(self.parallel_dims):
            logits = self._training_output(self.model(inputs, **extra_kwargs), labels)
            loss = self.loss_sum(logits, labels.reshape(-1), **self._loss_kwargs)
            with synchronize_local_errors(
                "Validation loss result validation", group=self.error_group
            ):
                validate_loss_sum(loss, device=labels.device)
            return loss

    def _forward_backward_batch(
        self, batch: Batch | TrainerBatch, *, global_valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        inputs, labels, extra_kwargs = self.preprocess(batch)
        return self.forward_backward_dense(
            inputs,
            labels,
            extra_kwargs=extra_kwargs,
            global_valid_tokens=global_valid_tokens,
        )

    def forward_backward_dense(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        extra_kwargs: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Run one non-PP accumulation group without selecting execution policy."""
        with synchronize_local_errors(
            "Training loss normalization validation", group=self.error_group
        ):
            validate_normalization_tokens(global_valid_tokens, device=labels.device)
        with self.param_context(), spmd_context(self.parallel_dims):
            return self._dense_backward(
                inputs,
                labels,
                extra_kwargs=extra_kwargs,
                global_valid_tokens=global_valid_tokens,
            )

    def finish_step(
        self,
        step: int,
        *,
        accumulation: AccumulationResult,
        reduction_meshes: TrainingReductionMeshes,
        collect_metrics: bool,
        before_gradient_clipping: Callable[[], None] | None = None,
    ) -> UpdateResult:
        """Synchronize, check and commit an accumulated optimizer update.

        The diagnostic hook observes synchronized, unclipped gradients. Keep
        it before clipping and routing observation, and reject a nonfinite
        step before optimizer, scheduler or EMA can advance. Clipping policy
        is bound at construction, independently of the Trainer lifecycle.
        """
        self.synchronize_gradients()
        with synchronize_local_errors(
            "Before gradient clipping hook", group=self.error_group
        ):
            if before_gradient_clipping is not None:
                before_gradient_clipping()
        grad_norm = gradients.clip_and_check_finite(
            self.model_parts,
            parallel_dims=self.parallel_dims,
            max_norm=self.max_norm,
            precision=self.grad_norm_precision,
            expert_reduce_layout=self.expert_reduce_layout,
            step=step,
            error_group=self.error_group,
            reduction_meshes=reduction_meshes,
            loss_is_finite=accumulation.loss_is_finite,
        )
        return self.optimizer_step(
            step,
            grad_norm=grad_norm,
            auxiliary_metrics=self.routing_metrics() if collect_metrics else {},
        )
