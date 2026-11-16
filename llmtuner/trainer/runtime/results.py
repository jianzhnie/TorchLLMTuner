"""Training state and result values crossing runtime boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any

import torch

from ...components.loss_contract import validate_loss_sum, validate_normalization_tokens


@dataclass(slots=True)
class TrainingProgress:
    """Checkpointable counters for one training run."""

    step: int = 0
    ntokens_seen: int = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate both counters before changing either live value."""
        step = state_dict["step"]
        ntokens_seen = state_dict["ntokens_seen"]
        for name, value in (("step", step), ("ntokens_seen", ntokens_seen)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"trainer {name} must be a nonnegative integer")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]


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

    def __post_init__(self) -> None:
        if type(self.valid_tokens) is not int or self.valid_tokens < 0:
            raise ValueError("Step valid_tokens must be a nonnegative integer")
        count = self.normalization_tokens
        validate_normalization_tokens(count)
        validate_loss_sum(self.loss_sum, device=count.device, requires_grad=False)


@dataclass(frozen=True, slots=True)
class AccumulationResult:
    """Detached reporting loss and device-local verdict for a full step window.

    Non-logging steps omit loss_sum to avoid retaining reporting tensors.
    loss_is_finite is an int32 scalar suitable for collective reduction.
    """

    loss_sum: torch.Tensor | None
    loss_is_finite: torch.Tensor

    def __post_init__(self) -> None:
        verdict = self.loss_is_finite
        if (
            not isinstance(verdict, torch.Tensor)
            or verdict.ndim != 0
            or verdict.dtype != torch.int32
        ):
            raise TypeError(
                "Accumulation finite verdict must be an int32 scalar tensor"
            )
        if self.loss_sum is not None:
            validate_loss_sum(self.loss_sum, device=verdict.device, requires_grad=False)


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
        if type(self.step) is not int or self.step < 0:
            raise ValueError("Update step must be a nonnegative integer")
        for name in ("updated", "scheduler_advanced", "ema_advanced"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"Update {name} must be boolean")
        if not self.updated and (self.scheduler_advanced or self.ema_advanced):
            raise ValueError("Scheduler and EMA cannot advance without an update")
        if self.grad_norm is not None:
            if not isinstance(self.grad_norm, torch.Tensor):
                raise TypeError("Update grad_norm must be a tensor or None")
            validate_loss_sum(
                self.grad_norm, device=self.grad_norm.device, requires_grad=False
            )
        metrics = dict(self.auxiliary_metrics)
        for name, value in metrics.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Auxiliary metric names must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Auxiliary metric {name!r} must be a real scalar")
            metrics[name] = float(value)
        object.__setattr__(self, "auxiliary_metrics", MappingProxyType(metrics))
