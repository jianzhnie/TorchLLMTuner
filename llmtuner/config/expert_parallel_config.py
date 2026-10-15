"""Expert parallel configuration constants and validation."""

from typing import TYPE_CHECKING

from llmtuner.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig


EP_DISPATCHERS = ("alltoall", "torchao", "domino", "mc2", "deepep", "hybridep")


def validate_expert_dispatcher(backend: str) -> None:
    """Validate backend availability without requiring a parallel config."""
    if backend not in EP_DISPATCHERS:
        raise ConfigError(
            "parallel.ep_token_dispatcher must be one of "
            f"{'/'.join(EP_DISPATCHERS)}, got {backend!r}"
        )
    if backend in ("deepep", "hybridep"):
        raise EnvironmentUnsupportedError(
            f"parallel.ep_token_dispatcher={backend!r} is "
            "a registered gap, not a supported backend: its external kernels "
            "and distributed wrappers are not included in this build. Use "
            "'alltoall'."
        )
    if backend == "mc2":
        raise EnvironmentUnsupportedError(
            "parallel.ep_token_dispatcher='mc2' requires the Ascend fused "
            "all-to-all/grouped-matmul operators. This build has no compatible "
            "MC2 operator registration; refusing an unfused silent fallback."
        )


def validate_expert_parallel_config(cfg: "ParallelConfig") -> None:
    validate_expert_dispatcher(cfg.ep_token_dispatcher)
    if cfg.ep_token_dispatcher != "alltoall" and cfg.expert_parallel_size == 1:
        raise UnsupportedCombinationError(
            f"parallel.ep_token_dispatcher={cfg.ep_token_dispatcher!r} "
            "has no effect at expert_parallel_size=1; set "
            "expert_parallel_size > 1 or use 'alltoall'."
        )


__all__ = [
    "EP_DISPATCHERS",
    "validate_expert_dispatcher",
    "validate_expert_parallel_config",
]
