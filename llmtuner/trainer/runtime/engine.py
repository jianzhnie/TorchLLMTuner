"""Model execution independent of the Trainer lifecycle.

The engine owns forward/backward dispatch and optimizer/scheduler/EMA updates.
Data iteration, gradient policy, metrics, and checkpointing remain Trainer
responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from torch.distributed import ProcessGroup

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.spmd import spmd_context
from ...components.loss import chunked_lm_head_cross_entropy, cross_entropy_loss
from ...components.loss_contract import (
    ChunkedLossBackward,
    LossSum,
    validate_loss_sum,
    validate_normalization_tokens,
)
from ...datasets.contracts import Batch, TrainerBatch
from ...models.contracts import DecoderStructure
from ...models.execution import ModelExecutionCapabilities, validate_training_output
from ...parallel.execution import ExecutionLayout
from ...parallel.topology import ParallelTopology
from . import pipeline
from .batch import prepare_model_inputs
from .results import StepResult, UpdateResult

ParamContext = Callable[[], AbstractContextManager[Any]]


def loss_vocab_kwargs(
    model: torch.nn.Module | None,
    parallel_dims: ParallelTopology | None,
    *,
    layout: ExecutionLayout | None = None,
) -> dict[str, Any]:
    """Resolve loss reduction from an explicit model and topology for train/eval."""
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
    if layout is not None:
        return layout.loss_kwargs(None if tp_mesh is None else tp_mesh.get_group())
    vocab_size = getattr(model, "vocab_size", None)
    if tp_mesh is None or vocab_size is None:
        return {}
    return {"tp_group": tp_mesh.get_group(), "global_vocab_size": vocab_size}


@dataclass(slots=True)
class TrainingEngine:
    """Execute forward/backward with fixed model and parallel dependencies."""

    model_parts: list[torch.nn.Module]
    parallel_dims: ParallelTopology | None
    device: torch.device
    max_context_length: int
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

    def __post_init__(self) -> None:
        if self.chunked_loss_num_chunks > 1 and self.chunked_loss_backward is None:
            raise ValueError("Configured loss does not support chunked backward")
        # Execution policy belongs to this assembly. Rebuild the engine after
        # changing topology or loss mode rather than selecting it on each batch.
        pipeline_enabled = (
            self.parallel_dims is not None and self.parallel_dims.pp_enabled
        )
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
            for layout in self.execution_layouts:
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
            if pipeline_enabled and (
                any(layout.output == "logits" for layout in self.execution_layouts)
                != self.pp_has_last_stage
            ):
                raise ValueError("Pipeline loss owner disagrees with output layout")
            stage_inputs = [
                layout.pipeline_io
                for layout in self.execution_layouts
                if layout.pipeline_io is not None
            ]
            if pipeline_enabled and stage_inputs:
                if len(stage_inputs) != len(self.execution_layouts):
                    raise ValueError("Pipeline parts must all declare stage I/O")
                if (
                    any(stage.first for stage in stage_inputs)
                    != self.pp_has_first_stage
                ):
                    raise ValueError("Pipeline token owner disagrees with stage I/O")
                if len({stage.hidden_size for stage in stage_inputs}) != 1:
                    raise ValueError("Local pipeline parts disagree on hidden width")
                if len({stage.activation_dtype for stage in stage_inputs}) != 1:
                    raise ValueError(
                        "Local pipeline parts disagree on activation dtype"
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

    @property
    def model(self) -> torch.nn.Module:
        if len(self.model_parts) != 1:
            raise RuntimeError("Dense execution requires exactly one model part")
        return self.model_parts[0]

    def loss_reduction_kwargs(self) -> dict[str, Any]:
        """Use the assembled output contract for dense train/evaluation loss."""
        return loss_vocab_kwargs(
            self.model,
            self.parallel_dims,
            layout=self.execution_layouts[0] if self.execution_layouts else None,
        )

    def preprocess(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return prepare_model_inputs(
            self.model_parts[0],
            batch,
            device=self.device,
            parallel_dims=self.parallel_dims,
            max_context_length=self.max_context_length,
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
            loss = self.loss_sum(logits, labels, **self.loss_reduction_kwargs())
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

    def _training_output(
        self,
        output: Any,
        labels: torch.Tensor,
        *,
        hidden_states: bool = False,
    ) -> torch.Tensor:
        """Coordinate local output checks after forward, before loss/backward.

        Forward may communicate and must finish outside this local boundary.
        Width is checked only when the assembly declares a replicated vocabulary;
        hidden projections and vocabulary shards retain their own contracts.
        """
        with synchronize_local_errors(
            "Model training output validation", group=self.error_group
        ):
            tensor = self._output_capabilities.output_tensor(
                output, hidden_states=hidden_states
            )
            width = None
            if not hidden_states and self.execution_layouts:
                layout = self.execution_layouts[0]
                if not layout.vocabulary_sharded:
                    width = layout.vocabulary_size
            validate_training_output(tensor, labels, output_width=width)
        return tensor

    def _backward_chunked(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        extra_kwargs: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        projection = self._chunked_projection
        if projection is None:
            raise RuntimeError("Chunked output projection is not bound")
        hidden_states = self._training_output(
            model(inputs, **extra_kwargs, skip_lm_head=True),
            labels,
            hidden_states=True,
        )
        if self.chunked_loss_backward is None:
            raise RuntimeError("Chunked loss backward is not bound")
        loss = self.chunked_loss_backward(
            projection,
            hidden_states,
            labels,
            num_chunks=self.chunked_loss_num_chunks,
            grad_scale=1.0 / global_valid_tokens,
            **self.loss_reduction_kwargs(),
        )
        with synchronize_local_errors(
            "Chunked loss result validation", group=self.error_group
        ):
            validate_loss_sum(loss, device=labels.device, requires_grad=False)
        return loss

    def _backward_logits(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *,
        extra_kwargs: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        logits = self._training_output(model(inputs, **extra_kwargs), labels)
        loss_sum = self.loss_sum(logits, labels, **self.loss_reduction_kwargs())
        del logits
        with synchronize_local_errors(
            "Training loss result validation", group=self.error_group
        ):
            validate_loss_sum(loss_sum, device=labels.device, requires_grad=True)
        (loss_sum / global_valid_tokens).backward()
        return loss_sum.detach()

    def optimizer_step(
        self,
        step: int,
        *,
        grad_norm: torch.Tensor | None = None,
        auxiliary_metrics: Mapping[str, float] | None = None,
    ) -> UpdateResult:
        """Commit an update, then advance scheduler and EMA in the fixed order."""
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError(
                "Optimizer services are not bound to the training engine"
            )
        self.optimizer.step()
        # The optimizer update is committed if scheduler coordination fails.
        with synchronize_local_errors(
            "Training scheduler update", group=self.error_group
        ):
            self.lr_scheduler.step()
        with synchronize_local_errors("Training EMA update", group=self.error_group):
            if self.ema is not None:
                # All ranks enter, including PP stages with no tracked tensors.
                self.ema.step(step)
        return UpdateResult(
            step=step,
            grad_norm=grad_norm,
            updated=True,
            scheduler_advanced=True,
            ema_advanced=self.ema is not None,
            auxiliary_metrics={} if auxiliary_metrics is None else auxiliary_metrics,
        )
