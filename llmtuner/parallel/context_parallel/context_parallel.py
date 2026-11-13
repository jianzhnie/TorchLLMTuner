"""Install model-owned KCP or Compressor context-parallel execution."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.parallel.compatibility import eager_flex_parallel

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
    numerical implementation. KCP and Compressor keep their own mesh checks.
    """
    if cfg.cp_size == 1:
        return model
    if not callable(getattr(model, "set_cp_mesh", None)):
        raise TypeError(
            "CP requires the explicit model to provide set_cp_mesh input semantics"
        )

    model_config = getattr(model, "config", None)
    if getattr(model_config, "attention_backend", "auto") == "eager_flex":
        eager_flex_parallel("CP")
    if cfg.cp_config.context_parallel_strategy == "kcp":
        if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
            raise ValueError("KCP requires a cp mesh")
        if mesh["cp"].size() != cfg.cp_size:
            raise ValueError("KCP mesh degree does not match the configured CP degree")
        if getattr(model_config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError("KCP does not support packed document attention")
        plan = resolve_cp_plan(model, cfg=cfg)
        patched = apply_kcp(model, mesh["cp"], resolved_plan=plan)
        model.set_cp_mesh(
            mesh["cp"],
            load_balancer=cfg.cp_config.context_parallel_load_balancer,
            strategy="kcp",
        )
        logger.info("Applied KCP to %d KDA modules", patched)
        return model

    if getattr(model, "context_parallel_backend", "standard") == "compressor":
        if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
            raise ValueError("DeepSeek V4 HCA CP requires a cp mesh")
        cp_mesh = mesh["cp"]
        if cp_mesh.size() != cfg.cp_size:
            raise ValueError("HCA CP mesh degree does not match configuration")
        plan = resolve_cp_plan(model, cfg=cfg)
        return apply_compressor_context_parallel(
            model, cp_mesh=cp_mesh, cfg=cfg, resolved_plan=plan
        )

    raise NotImplementedError(
        "Standard attention CP kernels have been removed; only model-owned "
        "KCP or Compressor CP backends are supported"
    )
