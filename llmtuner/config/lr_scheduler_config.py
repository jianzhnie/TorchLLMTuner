"""Warmup, stable phase, and decay configuration for learning rates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..errors import ConfigError
from .field_checks_config import require_integer_at_least


@dataclass(kw_only=True)
class LRSchedulerConfig:
    """The WSD schedule's knobs.

    The runtime side -- the ``LambdaLR`` this builds -- is
    ``llmtuner.components.optimizer.LRSchedulersContainer``.

    ``decay_ratio`` is the switch that matters. At its default of 0 there is no
    decay phase, so the factor is 1.0 throughout (after any warmup) and the
    learning rate is exactly the one the optimizer was built with -- which is
    what makes the default run comparable to every measurement taken at a
    constant lr. Setting it to a fraction appends a decay covering that fraction
    of ``total_steps``; whatever is left over after warmup and decay is the
    stable phase, at the peak learning rate.
    """

    warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to linearly ramp the learning rate from 0."},
    )
    total_steps: int | None = field(
        default=None,
        metadata={
            "help": "Length of the schedule. Defaults to --steps. Set it to "
            "decouple the curve from the run length, so a short debugging run "
            "sees the same lrs the full run would."
        },
    )
    decay_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of total_steps spent decaying the learning rate. "
            "0 (the default) never decays, holding the rate at its peak. A "
            "value below 1 leaves the intervening steps at the peak rate "
            "(WSD)."
        },
    )
    decay_type: Literal["linear", "sqrt", "cosine"] = field(
        default="linear",
        metadata={"help": "Shape of the decay phase. Ignored when decay_ratio=0."},
    )
    min_lr_factor: float = field(
        default=0.0,
        metadata={
            "help": "Floor of the decay, as a fraction of the base learning "
            "rate. 0 decays all the way to zero. Ignored when decay_ratio=0."
        },
    )

    def __post_init__(self) -> None:
        if self.decay_type not in ("linear", "sqrt", "cosine"):
            raise ConfigError(
                "lr_scheduler.decay_type must be linear/sqrt/cosine, "
                f"got {self.decay_type!r}"
            )
        require_integer_at_least(self, "warmup_steps", group="lr_scheduler", minimum=0)
        if self.total_steps is not None:
            require_integer_at_least(self, "total_steps", group="lr_scheduler")
        if (
            not isinstance(self.decay_ratio, int | float)
            or isinstance(self.decay_ratio, bool)
            or not 0.0 <= self.decay_ratio <= 1.0
        ):
            raise ConfigError(
                f"lr_scheduler.decay_ratio must be in [0, 1], got {self.decay_ratio}"
            )
        if (
            not isinstance(self.min_lr_factor, int | float)
            or isinstance(self.min_lr_factor, bool)
            or not 0.0 <= self.min_lr_factor < 1.0
        ):
            raise ConfigError(
                "lr_scheduler.min_lr_factor must be in [0, 1), got "
                f"{self.min_lr_factor}"
            )
