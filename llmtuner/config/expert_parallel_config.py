"""Expert parallel configuration constants and validation."""

from llmtuner.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)

EP_DISPATCHERS = ("alltoall", "torchao", "deepep", "hybridep")


def validate_expert_parallel_config(cfg) -> None:
    if cfg.ep_token_dispatcher not in EP_DISPATCHERS:
        raise ConfigError(
            "parallel.ep_token_dispatcher must be one of "
            f"{'/'.join(EP_DISPATCHERS)}, got {cfg.ep_token_dispatcher!r}"
        )
    if cfg.ep_token_dispatcher in ("deepep", "hybridep"):
        raise EnvironmentUnsupportedError(
            f"parallel.ep_token_dispatcher={cfg.ep_token_dispatcher!r} is "
            "a registered gap, not a supported backend: its external kernels "
            "and distributed wrappers are not included in this build. Use "
            "'alltoall'."
        )
    if cfg.ep_token_dispatcher != "alltoall" and cfg.expert_parallel_size == 1:
        raise UnsupportedCombinationError(
            f"parallel.ep_token_dispatcher={cfg.ep_token_dispatcher!r} "
            "has no effect at expert_parallel_size=1; set "
            "expert_parallel_size > 1 or use 'alltoall'."
        )


__all__ = ["EP_DISPATCHERS", "validate_expert_parallel_config"]
