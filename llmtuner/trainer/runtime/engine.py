"""Model execution independent of the Trainer lifecycle.

The engine owns forward/backward dispatch, parallel gradient synchronization,
routing state, gradient clipping and optimizer/scheduler/EMA updates. Data
iteration, reporting cadence and checkpointing remain Trainer responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from ...accelerator.collectives import all_reduce, synchronize_local_errors
from ...accelerator.spmd import spmd_context
from ...components.loss import chunked_lm_head_cross_entropy, cross_entropy_loss
from ...components.loss_contract import (
    ChunkedLossBackward,
    LossSum,
    validate_loss_sum,
    validate_normalization_tokens,
)
from ...datasets.contracts import Batch, TrainerBatch
from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.bias_update import (
    collect_routing_load_metrics,
    reset_routing_statistics,
)
from ...models.contracts import DecoderStructure
from ...models.execution import ModelExecutionCapabilities, validate_training_output
from ...parallel.execution import ExecutionLayout
from ...parallel.topology import ParallelTopology
from . import gradients
from .batch import prepare_model_inputs
from .reductions import TrainingReductionMeshes
from .results import AccumulationResult, StepResult, UpdateResult

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
class TrainingEngine:
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
    # All update participants coordinate failures through this group.
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
        if len(self.model_parts) != 1:
            raise ValueError("Dense execution requires exactly one model part")
        if self.chunked_loss_num_chunks > 1:
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
                if layout.output != "logits":
                    raise ValueError(
                        "Dense engine requires a logits-producing model part"
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
        if not self._output_capabilities.logits:
            raise ValueError("Dense execution requires model-declared logits output")
        if (
            self.chunked_loss_num_chunks > 1
            and not self._output_capabilities.hidden_states
        ):
            raise ValueError("Chunked loss requires model-declared hidden output")
        self._loss_kwargs = loss_vocab_kwargs(
            self.parallel_dims,
            layout=self.execution_layouts[0] if self.execution_layouts else None,
        )
        self._run_forward_backward = self._forward_backward_batch
        self._run_evaluation = self._evaluate_dense
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

    def begin_step(self) -> None:
        """Reset gradient and routing accumulation before reading the step's batches."""
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer services are not bound to the training engine"
            )
        self.optimizer.zero_grad(set_to_none=True)
        reset_routing_statistics(self.model_parts)

    def prepare_auxiliary_loss(
        self, local_routing_tokens: int, *, dp_group: ProcessGroup | None
    ) -> None:
        """Normalize router loss over non-padding inputs, including masked prompts."""
        if not AuxLoss.has_pending_counts():
            return
        global_routing_tokens = torch.tensor(
            local_routing_tokens, dtype=torch.int64, device=self.device
        )
        if dp_group is not None:
            all_reduce(global_routing_tokens, group=dp_group)
        AuxLoss.set_step_denominator(
            global_routing_tokens, model_parts=self.model_parts
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
        gradients.allreduce_replicated_tp_grads(
            self.model_parts, parallel_dims=self.parallel_dims
        )

    def accumulate_results(
        self, results: Iterable[StepResult], *, retain_loss: bool
    ) -> AccumulationResult:
        """Consume executed groups without retaining a list of detached losses.

        The iterable can execute each backward lazily, preserving caller-owned
        diagnostics around individual groups. Inspect DTensor shards locally;
        gradient validation coordinates the verdict after all groups finish.
        """
        loss_sum = None
        group_count = 0
        loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        for result in results:
            group_count += 1
            value = result.loss_sum
            local = value.to_local() if isinstance(value, DTensor) else value
            loss_is_finite.logical_and_(torch.isfinite(local).all())
            if retain_loss:
                if loss_sum is None:
                    loss_sum = value.clone()
                else:
                    loss_sum.add_(value)
        with synchronize_local_errors(
            "Training accumulation result validation", group=self.error_group
        ):
            if not group_count:
                raise ValueError("Training accumulation requires at least one group")
            return AccumulationResult(loss_sum, loss_is_finite)

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
        """Return raw validation loss.

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

    def _training_output(
        self,
        output: Any,
        labels: torch.Tensor,
        *,
        hidden_states: bool = False,
    ) -> torch.Tensor:
        """Validate original token geometry, then expose flat rows to loss services.

        Forward may communicate and must finish outside this local boundary.
        Width comes from model-declared hidden geometry or the assembled
        replicated vocabulary; vocabulary shards retain their own contracts.
        """
        with synchronize_local_errors(
            "Model training output validation", group=self.error_group
        ):
            tensor = self._output_capabilities.output_tensor(
                output, hidden_states=hidden_states
            )
            width = self._output_capabilities.hidden_size if hidden_states else None
            if not hidden_states and self.execution_layouts:
                layout = self.execution_layouts[0]
                if not layout.vocabulary_sharded:
                    width = layout.vocabulary_size
            validate_training_output(tensor, labels, output_width=width)
        # Reshaping a DTensor can communicate. Keep tensor operations outside
        # the local-only validation boundary and preserve the original autograd.
        return tensor if tensor.ndim == 2 else tensor.reshape(-1, tensor.shape[-1])

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
            labels.reshape(-1),
            num_chunks=self.chunked_loss_num_chunks,
            grad_scale=1.0 / global_valid_tokens,
            **self._loss_kwargs,
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
        loss_sum = self.loss_sum(logits, labels.reshape(-1), **self._loss_kwargs)
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
        with synchronize_local_errors(
            "Training update preflight", group=self.error_group
        ):
            if self.optimizer is None or self.lr_scheduler is None:
                raise RuntimeError(
                    "Optimizer services are not bound to the training engine"
                )
            # Validate and snapshot reporting inputs before any parameter write.
            # Publish this result only after every update operation succeeds.
            result = UpdateResult(
                step=step,
                grad_norm=grad_norm,
                updated=True,
                scheduler_advanced=True,
                ema_advanced=self.ema is not None,
                auxiliary_metrics={}
                if auxiliary_metrics is None
                else auxiliary_metrics,
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
        return result

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
