"""Validation (eval) loop config, grafted onto TrainingConfig."""

from __future__ import annotations

from dataclasses import dataclass

from llmtuner.errors import ConfigError


@dataclass(kw_only=True)
class ValidationConfig:
    """The validation (eval) loop's knobs (see ``Trainer.validate``).

    Set ``training.validation_config`` to one of these to turn validation on;
    the default ``None`` means no validation runs and the training loop is
    bit-identical to before. Programmatic-only, like ``ema_config``: a nested
    dataclass does not survive ``HfArgumentParser``.

    Validation replays the training loss forward-only: summed next-token
    cross-entropy over the whole pass, divided by the global valid-token count
    reduced across DP, so the number is independent of how the batches were
    split across ranks. It updates no parameters and touches no checkpoint
    state.
    """

    freq: int = 10
    """Validate every this many steps (step 1 always validates)."""

    steps: int = -1
    """Batches per validation pass. -1 consumes the finite dataset once
    (the loader is built with repeat=False). Ranks then stop independently,
    so -1 requires data-parallel degree 1 and a finite (non-random) dataset;
    both are rejected at trainer build time rather than hanging the pass."""

    dataset: str | None = None
    """Corpus override for validation, same vocabulary as
    ``dataloader.dataset``. None (the default) validates on the training
    corpus."""

    def __post_init__(self) -> None:
        if self.freq < 1:
            raise ConfigError(f"validation.freq must be >= 1, got {self.freq}")
        if not (self.steps >= 1 or self.steps == -1):
            raise ConfigError(f"validation.steps must be >= 1 or -1, got {self.steps}")
