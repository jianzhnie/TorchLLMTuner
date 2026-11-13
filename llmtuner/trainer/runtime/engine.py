"""Model execution independent of the Trainer lifecycle.

The engine owns forward/backward dispatch and optimizer/scheduler/EMA updates.
Data iteration, gradient policy, metrics, and checkpointing remain Trainer
responsibilities.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.spmd import spmd_context
from ...components.loss import chunked_lm_head_cross_entropy, cross_entropy_loss
from ...datasets.contracts import Batch, TrainerBatch
from ...parallel.topology import ParallelTopology
from . import pipeline
from .batch import prepare_model_inputs

ParamContext = Callable[[], AbstractContextManager[Any]]
LossSum = Callable[..., torch.Tensor]


def model_output_value(output: Any, *, field: str = "logits") -> torch.Tensor:
    """Extract the declared tensor from an explicit model output."""
    if isinstance(output, torch.Tensor):
        return output
    value = getattr(output, field, None)
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(
        f"Model output must be a Tensor or expose tensor .{field}; got "
        f"{type(output).__name__}"
    )


def loss_vocab_kwargs(
    model: torch.nn.Module | None, parallel_dims: ParallelTopology | None
) -> dict[str, Any]:
    """Resolve loss reduction from an explicit model and topology for train/eval."""
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
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
    loss_sum: LossSum = cross_entropy_loss
    optimizer: Any = None
    lr_scheduler: Any = None
    ema: Any = None
    pp_schedule: Any | None = None
    pp_has_first_stage: bool = False
    pp_has_last_stage: bool = False
    pp_loss_sentinel: torch.Tensor | None = None
    pp_num_microbatches: int = 1

    @property
    def model(self) -> torch.nn.Module:
        if len(self.model_parts) != 1:
            raise RuntimeError("Dense execution requires exactly one model part")
        return self.model_parts[0]

    def preprocess(
        self, batch: Batch | TrainerBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return prepare_model_inputs(
            self.model_parts[0],
            batch,
            device=self.device,
            parallel_dims=self.parallel_dims,
            max_context_length=self.max_context_length,
        )

    def pp_microbatches(
        self, batch: Batch | TrainerBatch
    ) -> list[Batch | TrainerBatch]:
        return pipeline.split_pipeline_microbatches(
            batch, num_microbatches=self.pp_num_microbatches
        )

    @property
    def _pp_loss_sentinel(self) -> torch.Tensor | None:
        """Preserve the existing pipeline runtime's sentinel field contract."""
        return self.pp_loss_sentinel

    def forward_backward(
        self,
        batch: Batch | TrainerBatch,
        *,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unnormalized loss sum after normalized backward."""
        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            return pipeline.run_pipeline_step(
                self, batch, global_valid_tokens=global_valid_tokens
            )
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
        model = self.model
        with self.param_context(), spmd_context(self.parallel_dims):
            if self.chunked_loss_num_chunks > 1:
                hidden_states = model_output_value(
                    model(inputs, **extra_kwargs, skip_lm_head=True),
                    field="last_hidden_state",
                )
                return chunked_lm_head_cross_entropy(
                    model.lm_head,
                    hidden_states,
                    labels,
                    num_chunks=self.chunked_loss_num_chunks,
                    grad_scale=1.0 / global_valid_tokens,
                    **self.loss_vocab_kwargs(),
                )
            logits = model_output_value(model(inputs, **extra_kwargs))
            loss_sum = self.loss_sum(logits, labels, **self.loss_vocab_kwargs())
            del logits
            (loss_sum / global_valid_tokens).backward()
        return loss_sum.detach()

    def optimizer_step(self, step: int) -> None:
        """Commit an update, then advance scheduler and EMA in the fixed order."""
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError(
                "Optimizer services are not bound to the training engine"
            )
        self.optimizer.step()
        # The optimizer update is committed if scheduler coordination fails.
        with synchronize_local_errors("Training scheduler update"):
            self.lr_scheduler.step()
        with synchronize_local_errors("Training EMA update"):
            if self.ema is not None:
                # All ranks enter, including PP stages with no tracked tensors.
                self.ema.step(step)

    def loss_vocab_kwargs(self) -> dict[str, Any]:
        """Describe the model-owned vocab layout for loss reduction."""
        return loss_vocab_kwargs(self.model, self.parallel_dims)
