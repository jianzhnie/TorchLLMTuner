"""Moving-average configuration and shared EMA schedule validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from ..errors import ConfigError
from ..utils.logging import get_logger
from .field_checks_config import require_integer_at_least

logger = get_logger(__name__)


@dataclass(kw_only=True)
class EMAConfig:
    """Online EMA of model weights (see ``components/optimizer/ema.py``)."""

    decay: float | None = None
    """Fixed decay per firing: ``ema = decay * ema + (1 - decay) * param``.
    If None (default), computed dynamically from ``half_life_fraction``."""

    half_life_fraction: float = 0.05
    """Used when ``decay`` is None:
    ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``. Keeps roughly
    the most recent ``half_life_fraction`` share of updates dominant."""

    start_step: int = 0
    """Last trainer step before EMA tracking begins, so the first update fires
    at ``start_step + update_every_n_steps``."""

    step_bias: int = 0
    """Offset added to the firing count when computing ``num_updates``, for
    renumbering a new training phase without resetting EMA aging. Measured in
    EMA firings, not raw steps. A normal resume needs no bias."""

    update_every_n_steps: int = 1
    """Only fire the EMA update every N real optimizer steps."""

    buffer_patterns: list[str] = field(default_factory=list)
    """Regex patterns (``re.search``, against buffer FQNs) selecting which
    buffers also get an EMA tracked -- e.g. an MoE's ``expert_bias_E``. Empty
    (default): no buffers tracked."""

    def __post_init__(self) -> None:
        validate_ema_schedule(self)
        # A fixed decay replaces the half-life schedule outright, so a
        # half_life_fraction set alongside it would do nothing.
        default_half_life = (
            type(self).__dataclass_fields__["half_life_fraction"].default
        )
        if self.decay is not None and self.half_life_fraction != default_half_life:
            logger.warning(
                "ema.half_life_fraction=%s is ignored because ema.decay=%s is "
                "set; the decay is then fixed and the half-life schedule is "
                "never used. Leave decay unset to use half_life_fraction.",
                self.half_life_fraction,
                self.decay,
            )


class _EMAScheduleParameters(Protocol):
    decay: float | None
    half_life_fraction: float
    start_step: int
    step_bias: int
    update_every_n_steps: int


def validate_ema_schedule(options: _EMAScheduleParameters) -> None:
    """Validate shared EMA scheduling options before allocating averages."""
    require_integer_at_least(options, "update_every_n_steps", group="ema")
    require_integer_at_least(options, "start_step", "step_bias", group="ema", minimum=0)
    if (
        not isinstance(options.half_life_fraction, int | float)
        or isinstance(options.half_life_fraction, bool)
        or options.half_life_fraction <= 0
        or not math.isfinite(options.half_life_fraction)
    ):
        raise ConfigError(
            "ema.half_life_fraction must be a finite number > 0, "
            f"got {options.half_life_fraction!r}"
        )
    if options.decay is not None and not (
        isinstance(options.decay, int | float)
        and not isinstance(options.decay, bool)
        and 0 <= options.decay < 1
    ):
        raise ConfigError(
            f"ema.decay must be finite and in [0, 1), got {options.decay!r}: "
            "decay=1 never updates the EMA."
        )
