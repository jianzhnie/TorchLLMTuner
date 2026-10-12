"""FSDP configuration types and validation."""

from typing import Literal

from llmtuner.errors import ConfigError

FSDPReshardPolicy = Literal["default", "always", "never"]
FSDPSymmMemScope = Literal["all", "dense"]


def validate_fsdp_config(cfg) -> None:
    """Validate FSDP policy fields owned by ``ParallelConfig``."""
    if cfg.fsdp_reshard_after_forward not in {"default", "always", "never"}:
        raise ConfigError(
            "parallel.fsdp_reshard_after_forward must be one of "
            "'default'/'always'/'never', got "
            f"{cfg.fsdp_reshard_after_forward!r}"
        )
    if cfg.fsdp_symm_mem_scope not in ("all", "dense"):
        raise ConfigError(
            "parallel.fsdp_symm_mem_scope must be one of 'all'/'dense', got "
            f"{cfg.fsdp_symm_mem_scope!r}"
        )
    if cfg.fsdp_forward_prefetch_count < 0 or cfg.fsdp_backward_prefetch_count < 0:
        raise ConfigError("FSDP prefetch counts must be nonnegative")
    if cfg.fsdp_implementation not in ("native", "custom"):
        raise ConfigError("parallel.fsdp_implementation must be 'native' or 'custom'")
    if cfg.fsdp_ignored_modules and not cfg.fsdp_apply_modules:
        raise ConfigError(
            "parallel.fsdp_ignored_modules requires fsdp_apply_modules; "
            "ignored patterns only belong to the pattern-driven FSDP path"
        )


__all__ = ["FSDPReshardPolicy", "FSDPSymmMemScope", "validate_fsdp_config"]
