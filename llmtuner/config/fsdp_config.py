"""FSDP configuration types and validation."""

from typing import TYPE_CHECKING, Literal

import torch

from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.errors import ConfigError

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig

FSDPReshardPolicy = Literal["default", "always", "never"]
FSDPSymmMemScope = Literal["all", "dense"]


def validate_fsdp_config(cfg: "ParallelConfig") -> None:
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
    require_integer_at_least(
        cfg,
        "fsdp_forward_prefetch_count",
        "fsdp_backward_prefetch_count",
        group="parallel",
        minimum=0,
    )
    if cfg.fsdp_implementation not in ("native", "custom"):
        raise ConfigError("parallel.fsdp_implementation must be 'native' or 'custom'")
    if cfg.fsdp_ignored_modules and not cfg.fsdp_apply_modules:
        raise ConfigError(
            "parallel.fsdp_ignored_modules requires fsdp_apply_modules; "
            "ignored patterns only belong to the pattern-driven FSDP path"
        )
    if cfg.enable_fsdp_symm_mem and (
        not torch.cuda.is_available()
        or (torch.version.hip is None and torch.cuda.get_device_capability() < (9, 0))
    ):
        raise ConfigError(
            "parallel.enable_fsdp_symm_mem is only supported on NVIDIA "
            "GPUs with compute capability 9.0 or newer."
        )


__all__ = ["FSDPReshardPolicy", "FSDPSymmMemScope", "validate_fsdp_config"]
