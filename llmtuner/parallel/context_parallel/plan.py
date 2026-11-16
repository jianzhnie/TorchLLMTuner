"""Resolve CP backend targets without mutation."""

from __future__ import annotations

from torch import nn

from ...config.parallel_config import ParallelConfig
from ..plan import FullPlanBindings, ParallelPlan
from .compressor_attention import resolve_compressor_plan
from .kcp import resolve_kcp_plan

__all__ = ["resolve_cp_plan"]


def resolve_cp_plan(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    bindings: FullPlanBindings | None = None,
) -> ParallelPlan:
    """Select the declared CP backend and resolve targets without mutation.

    Global preflight, PP-local ownership checks and post-TP/EP execution use
    the same selection rules, but each resolves its own live model bindings.
    """
    kcp = cfg.cp_config.context_parallel_strategy == "kcp"
    if (
        not kcp
        and getattr(model, "context_parallel_backend", "standard") != "compressor"
    ):
        raise NotImplementedError(
            "Standard attention CP kernels have been removed; only model-owned "
            "KCP or Compressor CP backends are supported"
        )
    if not callable(getattr(model, "set_cp_mesh", None)):
        raise TypeError(
            "CP requires the explicit model to provide set_cp_mesh input semantics"
        )
    if kcp:
        return resolve_kcp_plan(model, degree=cfg.cp_size, bindings=bindings)
    return resolve_compressor_plan(
        model, degree=cfg.cp_size, cfg=cfg, bindings=bindings
    )
