"""Wire CP onto a model: attach the CP flex kernel to every decoder layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel import compatibility

from ...utils.logging import get_logger
from ..attention_heads import require_heads_divisible_by
from .attention import CPFlexKernel, DualCPFlexKernel
from .sdpa_attention import CPSdpaKernel, CPSdpaUlyssesKernel, DualCPSdpaKernel

if TYPE_CHECKING:
    from ..topology import ParallelTopology

logger = get_logger(__name__)

__all__ = ["apply_cp"]

# HF names the attention submodule differently across model families; probe in
# the style of hf/model_setup.py's ``resolve_submodule_name`` rather than hardcoding one
# family's spelling.
_ATTN_MODULE_NAMES = ("self_attn", "attn", "attention")


def _require_ulysses_compatible(
    model: nn.Module, *, cfg: ParallelConfig, cp_mesh: DeviceMesh
) -> None:
    """Reject the ulysses combinations that would train a silently wrong model.

    The all-to-all is an even split of an evenly-sharded tensor, and every
    rank ends up attending the full sequence in whatever order the shards
    arrived in -- so a load balancer (which rearranges that order) and a head
    count that does not divide tp*cp are both configuration errors, raised
    here rather than discovered from the loss curve.
    """
    model_config = getattr(getattr(model, "model", None), "config", None)
    load_balancer = cfg.cp_config.context_parallel_load_balancer
    if load_balancer is not None:
        raise ValueError(
            "Ulysses CP requires context_parallel_load_balancer=None: "
            "load balancing permutes tokens but the Ulysses mask expects "
            "original sequence order. Use context_parallel_strategy='kv_allgather' "
            "for load-balanced CP."
        )
    if (
        getattr(model_config, "attn_mask_type", "causal") == "block_causal"
        and cfg.context_parallel_kv_allgather_size == 1
    ):
        # Packed sequences ARE supported: the all-to-all reassembles the
        # full token stream on every rank before attention, so the
        # full-length document mask applies unsharded (upstream's varlen
        # ulysses semantics). What makes it work is the wrapper passing
        # that mask FULL-LENGTH rather than Q-sharded -- latched through
        # ``set_cp_mesh`` below, and consumed in the kernel by mask length.
        logger.info(
            "Ulysses CP with packed sequences: the document mask is passed "
            "full-length (unsharded) to every rank."
        )
    # TP shards heads first, so what ulysses must divide evenly is each
    # rank's local head count -- equivalently, the global count must divide
    # tp * cp (upstream torchtitan's head_shard_degree in its
    # config/validation.py).
    from ..tensor_parallel.tensor_parallel import ReplicatedAttentionSequenceBoundary

    attentions = [
        getattr(layer, name)
        for layer in getattr(model, "layers", [])
        for name in _ATTN_MODULE_NAMES
        if hasattr(layer, name)
    ]
    # A TP plan may leave attention replicated (e.g. MLA). Its heads have
    # not been cut by TP; only Ulysses divides them. For mixed layouts keep
    # the stricter check required by any head-sharded layer.
    replicated = bool(attentions) and all(
        isinstance(attention, ReplicatedAttentionSequenceBoundary)
        for attention in attentions
    )
    head_degree = cp_mesh.size() * (1 if replicated else cfg.tp_size)
    require_heads_divisible_by(
        model,
        size=head_degree,
        axis="cp" if replicated else "tp*cp",
        why="ulysses must divide the heads actually held by each attention module",
    )


def apply_cp(
    model: nn.Module,
    *,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
    parallel_dims: ParallelTopology | None = None,
) -> nn.Module:
    """Attach a CP flex kernel to every decoder layer's attention module.

    No-op when ``cfg.cp_size == 1`` (the model is handed back untouched), so the
    caller needs no degree check of its own.

    The kernel declares the CP region: q/k/v reach it sequence-sharded and it
    redistributes them per the configured strategy (K/V all-gather, or ulysses
    all-to-all onto the head axis). The model inputs still have to be sharded
    along the sequence axis -- that is the trainer's half of the wiring
    (``shard_batch_for_cp``).
    """
    if cfg.cp_size == 1:
        return model

    model_config = getattr(getattr(model, "model", None), "config", None)
    if getattr(model_config, "attention_backend", "auto") == "eager_flex":
        compatibility.eager_flex_parallel("CP")
    if cfg.cp_config.context_parallel_strategy == "kcp":
        if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
            raise ValueError("KCP requires a cp mesh")
        if mesh["cp"].size() != cfg.cp_size:
            raise ValueError("KCP mesh degree does not match the configured CP degree")
        if getattr(model_config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError("KCP does not support packed document attention")
        from .kcp import apply_kcp

        patched = apply_kcp(model, mesh["cp"])
        model.set_cp_mesh(
            mesh["cp"],
            load_balancer=cfg.cp_config.context_parallel_load_balancer,
            strategy="kcp",
        )
        logger.info("Applied KCP to %d KDA modules", patched)
        return model

    if getattr(model_config, "model_type", None) == "deepseek_v4":
        if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
            raise ValueError("DeepSeek V4 HCA CP requires a cp mesh")
        cp_mesh = mesh["cp"]
        if cp_mesh.size() != cfg.cp_size:
            raise ValueError("HCA CP mesh degree does not match configuration")
        from .compressor_attention import apply_compressor_context_parallel

        return apply_compressor_context_parallel(model, cp_mesh=cp_mesh, cfg=cfg)

    # Import the HF backend only when CP is enabled; disabled CP needs no
    # Transformers attention registration dependencies.
    from ...models.hf.attention_backend import ATTENTION_BACKEND, CP_SDPA_BACKEND

    impl = getattr(model_config, "_attn_implementation", None)
    sdpa_backend = impl != ATTENTION_BACKEND
    if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
        raise ValueError(
            f"cp={cfg.cp_size} requires a device mesh with a 'cp' axis, got "
            f"{None if mesh is None else mesh.mesh_dim_names}."
        )
    cp_mesh = mesh["cp"]
    if cp_mesh.size() != cfg.cp_size:
        raise ValueError(
            f"mesh 'cp' axis has size {cp_mesh.size()}, "
            f"but cfg requests cp={cfg.cp_size}."
        )

    strategy = cfg.cp_config.context_parallel_strategy
    load_balancer = cfg.cp_config.context_parallel_load_balancer
    dual_axes = (
        cfg.context_parallel_kv_allgather_size > 1
        and cfg.context_parallel_ulysses_size > 1
    )
    kv_mesh = ulysses_mesh = None
    if dual_axes:
        if parallel_dims is None:
            raise ValueError("dual CP axes require ParallelTopology")
        kv_mesh = parallel_dims.get_optional_mesh("cp_kv")
        ulysses_mesh = parallel_dims.get_optional_mesh("cp_ulysses")
        if kv_mesh is None or ulysses_mesh is None:
            raise ValueError("dual CP axis meshes were not constructed")
        for axis, axis_mesh, degree in (
            ("cp_kv", kv_mesh, cfg.context_parallel_kv_allgather_size),
            ("cp_ulysses", ulysses_mesh, cfg.context_parallel_ulysses_size),
        ):
            if axis_mesh.size() != degree:
                raise ValueError(
                    f"mesh {axis!r} axis has size {axis_mesh.size()}, "
                    f"but cfg requests {degree}"
                )
        _require_ulysses_compatible(model, cfg=cfg, cp_mesh=ulysses_mesh)
    elif strategy == "ulysses":
        _require_ulysses_compatible(model, cfg=cfg, cp_mesh=cp_mesh)

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_cp expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )
    packed = getattr(model_config, "attn_mask_type", "causal") == "block_causal"
    if sdpa_backend:
        if dual_axes:
            cp_sdpa_kernel = DualCPSdpaKernel(
                kv_mesh=kv_mesh, ulysses_mesh=ulysses_mesh, packed=packed
            )
        elif strategy == "ulysses":
            if load_balancer is not None:
                raise RuntimeError(
                    "ulysses CP takes no load balancer: the rearrangement it "
                    "would introduce lands in the query order, and this "
                    "kernel's mask is the plain causal one (the flex kernel "
                    "expresses the permuted form through its BlockMask). Run "
                    "with context_parallel_load_balancer=None, or use the "
                    "kv_allgather strategy, which is correct with a balancer."
                )
            cp_sdpa_kernel = CPSdpaUlyssesKernel(cp_mesh=cp_mesh, packed=packed)
        else:
            if packed and load_balancer is not None:
                raise NotImplementedError(
                    "Packed single-axis KV SDPA CP requires "
                    "context_parallel_load_balancer=None until document mask "
                    "permutation is implemented and validated."
                )
            cp_sdpa_kernel = CPSdpaKernel(cp_mesh=cp_mesh, packed=packed)
        model_config._attn_implementation = CP_SDPA_BACKEND
        # The wrapper reads this to decide whether the layer needs a mask handed
        # over: the kv kernel builds its own from the offsets, while the ulysses
        # kernel consumes a packed corpus's dense document mask (it attends the
        # full sequence, so the unsharded structure applies directly).
        model_config._cp_sdpa_ulysses = strategy == "ulysses"
        model_config._cp_sdpa_dual = dual_axes
    attentions = []
    for idx, layer in enumerate(layers):
        for name in _ATTN_MODULE_NAMES:
            if hasattr(layer, name):
                attn_mod = getattr(layer, name)
                break
        else:
            raise AttributeError(
                f"{type(layer).__name__} (layer {idx}) has no attention module "
                f"under any of {_ATTN_MODULE_NAMES}. Add the model's spelling "
                "to the probe in apply_cp."
            )
        attentions.append(attn_mod)
    if sdpa_backend:
        for attn_mod in attentions:
            attn_mod._titan_cp_sdpa_kernel = cp_sdpa_kernel
    else:
        kernels = []
        for _ in attentions:
            if dual_axes:
                kernel = DualCPFlexKernel(kv_mesh=kv_mesh, ulysses_mesh=ulysses_mesh)
            else:
                kernel = CPFlexKernel(cp_mesh=cp_mesh, strategy=strategy, packed=packed)
            kernels.append(kernel)
        for attn_mod, kernel in zip(attentions, kernels, strict=True):
            attn_mod._titan_flex_kernel = kernel

    cp_options = {
        "load_balancer": load_balancer,
        "strategy": "kv_allgather" if dual_axes else strategy,
    }
    if dual_axes:
        cp_options["mask_mesh"] = kv_mesh
    model.set_cp_mesh(cp_mesh, **cp_options)
    logger.info(
        "Applied CP (%s) with degree %d",
        "kv_allgather*ulysses" if dual_axes else strategy,
        cfg.cp_size,
    )
    return model
