"""Install model-owned KCP or Compressor context-parallel execution."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from ...config.parallel_config import ParallelConfig
from ...utils.logging import get_logger
from .compressor_attention import apply_compressor_context_parallel
from .kcp import apply_kcp
from .plan import resolve_cp_plan

logger = get_logger(__name__)
__all__ = ["apply_cp"]


def apply_cp(
    model: nn.Module,
    *,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
) -> nn.Module:
    """Install a model-owned backend; CP=1 leaves the model untouched.

    Removed standard attention plans are rejected instead of selecting another
    numerical implementation. Backend installers consume the resolved plan and
    revalidate its live bindings before acquiring runtime resources.
    """
    if cfg.cp_size == 1:
        return model
    plan = resolve_cp_plan(model, cfg=cfg)
    if plan.implementation not in ("kcp", "compressor"):
        raise ValueError(
            f"Unsupported resolved CP implementation: {plan.implementation}"
        )
    label = "KCP" if plan.implementation == "kcp" else "DeepSeek V4 HCA CP"
    if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
        raise ValueError(f"{label} requires a cp mesh")
    cp_mesh = mesh["cp"]
    if cp_mesh.size() != plan.degree:
        raise ValueError(f"{label} mesh degree does not match configuration")

    if plan.implementation == "kcp":
        patched = apply_kcp(model, cp_mesh, resolved_plan=plan)
        model.set_cp_mesh(
            cp_mesh,
            load_balancer=cfg.cp_config.context_parallel_load_balancer,
        )
        logger.info("Applied KCP to %d KDA modules", patched)
        return model

    return apply_compressor_context_parallel(
        model, cp_mesh=cp_mesh, cfg=cfg, resolved_plan=plan
    )
