"""Pipeline parallel configuration validation."""

from typing import TYPE_CHECKING

from llmtuner.errors import ConfigError

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig


def validate_pipeline_parallel_config(cfg: "ParallelConfig") -> None:
    """Validate schedule selection before constructing pipeline stages."""
    if cfg.pipeline_parallel_schedule_csv:
        raise ConfigError(
            "parallel.pipeline_parallel_schedule_csv is not supported; "
            "use pipeline_parallel_schedule instead."
        )
    # Keep schedule discovery lazy to avoid importing pipelining at import time.
    from torch.distributed.pipelining.schedules import get_schedule_class

    try:
        get_schedule_class(cfg.pipeline_parallel_schedule)
    except ValueError as exc:
        raise ConfigError(
            "Invalid parallel.pipeline_parallel_schedule "
            f"{cfg.pipeline_parallel_schedule!r}: {exc}"
        ) from exc


__all__ = ["validate_pipeline_parallel_config"]
