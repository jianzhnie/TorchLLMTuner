"""Resolve CP backend targets and validate pipeline ownership without mutation."""

from __future__ import annotations

from torch import nn

from ...config.parallel_config import ParallelConfig
from ..plan import ParallelPlan, validate_pipeline_targets
from .compressor_attention import resolve_compressor_plan
from .kcp import resolve_kcp_plan

__all__ = ["resolve_cp_plan", "validate_pipeline_cp_plan"]


def resolve_cp_plan(model: nn.Module, *, cfg: ParallelConfig) -> ParallelPlan:
    """Select the declared CP backend and resolve targets without mutation.

    Global preflight, PP-local ownership checks and post-TP/EP execution use
    the same selection rules, but each resolves its own live model bindings.
    """
    if cfg.cp_config.context_parallel_strategy == "kcp":
        return resolve_kcp_plan(model, degree=cfg.cp_size)
    if getattr(model, "context_parallel_backend", "standard") == "compressor":
        return resolve_compressor_plan(model, degree=cfg.cp_size, cfg=cfg)
    raise NotImplementedError(
        "Standard attention CP kernels have been removed; only model-owned "
        "KCP or Compressor CP backends are supported"
    )


def validate_pipeline_cp_plan(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> None:
    """Check PP-local CP targets and layouts before other stages mutate them."""
    if (
        global_plan.stage != "cp"
        or local_plan.stage != "cp"
        or global_plan.degree != local_plan.degree
        or global_plan.implementation != local_plan.implementation
    ):
        raise ValueError("Pipeline CP plans must describe the same execution")
    for original, target in validate_pipeline_targets(global_plan, local_plan, model):
        if target.layout != original.layout:
            raise ValueError(f"Pipeline CP layout changed: {target.path}")
