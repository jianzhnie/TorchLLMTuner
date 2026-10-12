"""Context parallel configuration constants and validation."""

from llmtuner.errors import ConfigError, UnsupportedCombinationError

CP_STRATEGIES = frozenset({"kv_allgather", "ulysses", "kcp"})
CP_LOAD_BALANCERS = frozenset({None, "headtail", "ptrr"})


def validate_context_parallel_config(cfg) -> None:
    if cfg.context_parallel_load_balancer == "":
        raise ConfigError(
            "parallel.context_parallel_load_balancer cannot be an empty "
            "string; use None to disable load balancing."
        )
    if cfg.context_parallel_load_balancer not in CP_LOAD_BALANCERS:
        raise ConfigError(
            "parallel.context_parallel_load_balancer must be one of "
            "None/'headtail'/'ptrr', got "
            f"{cfg.context_parallel_load_balancer!r}"
        )
    if cfg.context_parallel_load_balancer == "ptrr":
        raise UnsupportedCombinationError(
            "parallel.context_parallel_load_balancer='ptrr' is not "
            "implemented in llmtuner; use 'headtail' or None."
        )
    if cfg.context_parallel_strategy not in CP_STRATEGIES:
        raise ConfigError(
            "parallel.context_parallel_strategy must be one of "
            "'kv_allgather'/'ulysses'/'kcp', got "
            f"{cfg.context_parallel_strategy!r}"
        )
    if cfg.context_parallel_strategy == "kcp":
        if cfg.tensor_parallel_size != 1:
            raise UnsupportedCombinationError("KCP requires tensor_parallel_size=1")
        if (
            cfg.context_parallel_kv_allgather_size != 1
            or cfg.context_parallel_ulysses_size != 1
        ):
            raise UnsupportedCombinationError(
                "KCP cannot be combined with KV-AllGather or Ulysses CP axes"
            )
    if (
        cfg.context_parallel_strategy == "ulysses"
        and cfg.context_parallel_load_balancer is not None
    ):
        raise UnsupportedCombinationError(
            "parallel.context_parallel_strategy='ulysses' requires "
            "context_parallel_load_balancer=None."
        )
    if (
        cfg.context_parallel_ulysses_size > 1
        and cfg.context_parallel_load_balancer is not None
    ):
        raise UnsupportedCombinationError(
            "the Ulysses CP axis requires context_parallel_load_balancer=None"
        )


__all__ = ["CP_LOAD_BALANCERS", "CP_STRATEGIES", "validate_context_parallel_config"]
