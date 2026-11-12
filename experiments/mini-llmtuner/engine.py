"""Execution owns forward/backward/update, independent of Trainer and data."""

from dataclasses import dataclass

import torch

from .components import Loss
from .parallel import ParallelRuntime

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class StepResult:
    loss_sum: float
    targets: int
    grad_norm: float


class TrainingEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        loss: Loss,
        runtime: ParallelRuntime | None = None,
    ):
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.loss = scheduler, loss
        self.runtime = runtime or ParallelRuntime()

    def train_step(self, batches: list[Batch]) -> StepResult:
        targets = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
        device = next(self.model.parameters()).device
        global_targets = int(
            self.runtime.sum_data(torch.tensor(targets, device=device))
        )
        if global_targets == 0:
            raise ValueError("Optimizer step requires valid targets")
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for batch in batches:
            prepared = self.model.preprocess_inputs(batch)
            logits = self.model(**{k: v for k, v in prepared.items() if k != "labels"})
            loss = self.loss(logits, batch["labels"])
            (loss * (self.runtime.data_size / global_targets)).backward()
            loss_sum += float(loss.detach())
        norm = self.runtime.synchronize_and_clip(self.model)
        self.optimizer.step()
        self.scheduler.step()
        global_loss = float(
            self.runtime.sum_data(torch.tensor(loss_sum, device=device))
        )
        return StepResult(global_loss, global_targets, norm)
