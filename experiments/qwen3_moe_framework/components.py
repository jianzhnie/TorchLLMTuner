"""Replaceable training components used by the minimal engine."""

from dataclasses import dataclass
from typing import Protocol

import torch


class Loss(Protocol):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...


class CrossEntropyLoss:
    def __call__(self, logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )


class Metrics(Protocol):
    def record(self, step: int, loss: torch.Tensor, labels: torch.Tensor) -> dict: ...


class TokenMetrics:
    def record(self, step, loss, labels):
        return {
            "step": step,
            "loss": float(loss.detach()),
            "tokens": int(labels.ne(-100).sum()),
        }


class Scheduler(Protocol):
    def step(self) -> None: ...


@dataclass
class NoOpScheduler:
    def step(self) -> None:
        pass


class Profiler(Protocol):
    def step(self) -> None: ...


class NoOpProfiler:
    def step(self) -> None:
        pass


class OptimizerFactory(Protocol):
    def __call__(self, parameters, learning_rate: float) -> torch.optim.Optimizer: ...


class AdamWFactory:
    def __call__(self, parameters, learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameters, lr=learning_rate)
