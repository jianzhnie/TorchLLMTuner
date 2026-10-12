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


__all__ = ["FSDPReshardPolicy", "FSDPSymmMemScope", "validate_fsdp_config"]
