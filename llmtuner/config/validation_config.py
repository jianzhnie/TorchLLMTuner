"""Validation cadence, pass length, and optional corpus selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError
from .field_checks_config import require_integer_at_least


@dataclass(kw_only=True)
class ValidationConfig:
    """Configure a pass run by ``Trainer.validate``."""

    freq: int = 10
    """Validate every this many steps (step 1 always validates)."""

    steps: int = -1
    "Batches per validation pass; -1 consumes the finite dataset once, with "
    "exhaustion coordinated across ranks."

    dataset: str | None = None
    "Validation corpus override; None uses training data. Reuses training tokenizer "
    "and data-processing settings."

    def __post_init__(self) -> None:
        require_integer_at_least(self, "freq", group="validation")
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or not (self.steps >= 1 or self.steps == -1)
        ):
            raise ConfigError(f"validation.steps must be >= 1 or -1, got {self.steps}")
