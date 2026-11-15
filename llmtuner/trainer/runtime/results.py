"""Explicit values crossing execution and training lifecycle boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch


@dataclass(frozen=True, slots=True)
class StepResult:
    """One accumulation group's detached loss and normalization context.

    valid_tokens is the caller's pre-shard count, replicated across TP/CP;
    it must not be independently summed across those axes. normalization_tokens
    is the global denominator shared by every group in this optimizer step.
    """

    loss_sum: torch.Tensor
    valid_tokens: int
    normalization_tokens: torch.Tensor


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """An optimizer update completed with scheduler and optional EMA advancement.

    Failures still raise; a scheduler failure after optimizer.step does not
    produce a successful result. Current finite-step policy raises before update
    rather than returning a skipped update.

    auxiliary_metrics holds detached reporting scalars: routing observations
    captured before update and auxiliary-loss snapshots reduced at log time.
    The mapping is copied so later collector mutations cannot alter this result.
    """

    step: int
    grad_norm: torch.Tensor | None
    updated: bool
    scheduler_advanced: bool
    ema_advanced: bool
    auxiliary_metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "auxiliary_metrics", MappingProxyType(dict(self.auxiliary_metrics))
        )
