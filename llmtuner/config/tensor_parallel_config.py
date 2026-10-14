"""Tensor parallel configuration constants and validation."""

from typing import TYPE_CHECKING

from llmtuner.errors import ConfigError, UnsupportedCombinationError

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig


def validate_tensor_parallel_config(cfg: "ParallelConfig") -> None:
    """Reject unsupported TP settings before model or mesh construction."""
    if not cfg.enable_sequence_parallel:
        raise UnsupportedCombinationError(
            "parallel.enable_sequence_parallel=False is not supported: "
            "llmtuner's tensor parallelism is sequence-parallel by "
            "construction. Leave it true, or set tensor_parallel_size=1."
        )
    if cfg.tp_sequence_modules:
        raise UnsupportedCombinationError(
            "parallel.tp_sequence_modules is not supported by the fused sequence "
            "parallel engine; normalization already consumes the sequence shard"
        )
    conflicting = set(cfg.tp_colwise_modules) & set(cfg.tp_rowwise_modules)
    if conflicting:
        raise ConfigError(
            f"TP patterns {sorted(conflicting)!r} have conflicting styles in "
            "parallel.tp_colwise_modules and parallel.tp_rowwise_modules"
        )


__all__ = ["validate_tensor_parallel_config"]
