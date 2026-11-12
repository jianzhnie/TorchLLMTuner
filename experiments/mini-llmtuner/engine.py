"""Execution owns forward/backward/update, independent of Trainer and data."""

from dataclasses import dataclass

import torch

from .components import Loss

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class StepResult:
    loss_sum: float
    targets: int
    grad_norm: float


class TrainingEngine:
    def __init__(self, model: torch.nn.Module, optimizer, scheduler, loss: Loss):
        self.model, self.optimizer = model, optimizer
        self.scheduler, self.loss = scheduler, loss

    def train_step(self, batches: list[Batch]) -> StepResult:
        targets = sum(int(batch["labels"].ne(-100).sum()) for batch in batches)
        if targets == 0:
            raise ValueError("Optimizer step requires valid targets")
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for batch in batches:
            prepared = self.model.preprocess_inputs(batch)
            logits = self.model(**{k: v for k, v in prepared.items() if k != "labels"})
            loss = self.loss(logits, batch["labels"])
            (loss / targets).backward()
            loss_sum += float(loss.detach())
        parameters = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return StepResult(loss_sum, targets, float(norm))
