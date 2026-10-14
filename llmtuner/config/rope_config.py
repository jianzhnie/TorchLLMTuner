"""Rotary embedding dimensions and mode-specific scaling parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..errors import ConfigError
from .field_checks_config import require_integer_at_least


@dataclass(kw_only=True)
class RoPEConfig:
    """RoPE hyperparameters.

    The scaling fields are grouped by the mode that reads them: ``llama`` reads
    ``scaling_factor``/``low_freq_factor``/``high_freq_factor``/
    ``original_max_position_embeddings``, ``yarn`` reads the ``rope_factor``/
    ``beta_*``/``original_seq_len``/``truncate`` set. Fields for unused modes
    are ignored; ``CosSinRoPE`` rejects ``llama`` at runtime construction.
    """

    dim: int
    max_context_length: int
    theta: float = 10000.0
    scaling: Literal["none", "llama", "yarn"] = "none"
    # llama scaling params
    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    # yarn scaling params
    rope_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 4096
    truncate: bool = True

    def __post_init__(self) -> None:
        require_integer_at_least(self, "dim", "max_context_length", group="rope")
        if self.dim % 2:
            raise ConfigError("rope.dim must be even to form rotary dimension pairs")
        if self.scaling not in ("none", "llama", "yarn"):
            raise ConfigError("rope.scaling must be one of none/llama/yarn")
        positive_fields = ["theta"]
        if self.scaling == "llama":
            positive_fields.extend(
                ["scaling_factor", "low_freq_factor", "high_freq_factor"]
            )
            require_integer_at_least(
                self, "original_max_position_embeddings", group="rope"
            )
        elif self.scaling == "yarn":
            positive_fields.extend(["rope_factor", "beta_fast", "beta_slow"])
            require_integer_at_least(self, "original_seq_len", group="rope")
        for name in positive_fields:
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or value <= 0
                or not math.isfinite(value)
            ):
                raise ConfigError(
                    f"rope.{name} must be finite and positive, got {value!r}"
                )
        if self.scaling == "llama" and self.high_freq_factor <= self.low_freq_factor:
            raise ConfigError("rope.high_freq_factor must exceed low_freq_factor")
        if self.scaling == "yarn":
            if self.rope_factor < 1:
                raise ConfigError("rope.rope_factor must be >= 1 for YaRN scaling")
            if not isinstance(self.truncate, bool):
                raise ConfigError("rope.truncate must be bool")
            if self.rope_factor > 1:
                if self.theta <= 1:
                    raise ConfigError("rope.theta must be > 1 for YaRN scaling")
                if self.beta_fast < self.beta_slow:
                    raise ConfigError("rope.beta_fast must be >= beta_slow")
