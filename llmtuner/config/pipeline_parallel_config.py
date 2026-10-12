"""Pipeline parallel configuration validation."""

from llmtuner.errors import ConfigError


def validate_pipeline_parallel_config(cfg) -> None:
    if cfg.pipeline_parallel_schedule_csv:
        raise ConfigError(
            "parallel.pipeline_parallel_schedule_csv is not supported; "
            "use pipeline_parallel_schedule instead."
        )


__all__ = ["validate_pipeline_parallel_config"]
