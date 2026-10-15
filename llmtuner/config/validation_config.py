"""Validation cadence, pass length, and optional corpus selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError
from .field_checks_config import require_integer_at_least


@dataclass(kw_only=True)
class ValidationConfig:
    """Configure a pass run by ``Trainer.validate``.

    Assign to ``training.validation_config`` to enable validation. This nested
    config is programmatic; ``config/cli_config.py`` excludes it from CLI views.

    Validation reports summed next-token loss divided by global valid tokens.
    Each pass owns a fresh loader, leaves training counters unchanged, and
    restores model modes and attention layout after success or failure.
    """

    freq: int = 10
    """Validate every this many steps (step 1 always validates)."""

    steps: int = -1
    """Batches per validation pass. -1 consumes the finite dataset once
    (the loader is built with repeat=False). Ranks then stop independently,
    so -1 requires data-parallel degree 1 and a finite (non-random) dataset;
    both are rejected at trainer build time rather than hanging the pass."""

    dataset: str | None = None
    """Optional corpus override using the dataloader's dataset names.

    None selects the training corpus. Tokenizer, corpus path, packing capacity,
    and other loader settings remain those of the training data configuration.
    Attention layout follows the effective corpus; KCP rejects packed passes.
    """

    def __post_init__(self) -> None:
        require_integer_at_least(self, "freq", group="validation")
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or not (self.steps >= 1 or self.steps == -1)
        ):
            raise ConfigError(f"validation.steps must be >= 1 or -1, got {self.steps}")
