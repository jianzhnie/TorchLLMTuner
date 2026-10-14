"""Wire CP onto a model: attach the CP flex kernel to every decoder layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.parallel_config import ParallelConfig

from ...utils.logging import get_logger
from ..attention_heads import require_heads_divisible_by
from .attention import CPFlexKernel, DualCPFlexKernel

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
    load_balancer = cfg.context_parallel_load_balancer
    if load_balancer is not None:
        raise ValueError(
            "Ulysses CP requires context_parallel_load_balancer=None: the "
            "all-to-all is an even split of an evenly-sharded tensor, and "
            "every rank ends up attending the full sequence in whatever "
            "order the shards arrived in. A load balancer rearranges the "
            "sequence into head and tail chunks, so that order is no "
            "longer the original one and -- unlike the seq_len x seq_len "
            "causal mask, a function of tokens -- flex only takes a mask "
            "built over the attended order. The kernel therefore attends "
            "the rearranged corpus, a permuted model that trains and "
            "produces plausible numbers with nothing raised. "
            "strategy='kv_allgather' is what supports this: it gathers the "
            "rearranged shards back into the very order the mask was "
            "sharded in, so the rearrangement lands in the mask and cancels."
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
    head_degree = cp_mesh.size() * (1 if replicated else cfg.tp)
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

    No-op when ``cfg.cp == 1`` (the model is handed back untouched), so the
    caller needs no degree check of its own.

    The kernel declares the CP region: q/k/v reach it sequence-sharded and it
    redistributes them per the configured strategy (K/V all-gather, or ulysses
    all-to-all onto the head axis). The model inputs still have to be sharded
    along the sequence axis -- that is the trainer's half of the wiring
    (``shard_batch_for_cp``).
    """
    if cfg.cp == 1:
        return model

    if cfg.context_parallel_strategy == "kcp":
        if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
            raise ValueError("KCP requires a cp mesh")
        if mesh["cp"].size() != cfg.cp:
            raise ValueError("KCP mesh degree does not match the configured CP degree")
        model_config = getattr(getattr(model, "model", None), "config", None)
        if getattr(model_config, "attn_mask_type", "causal") == "block_causal":
            raise ValueError("KCP does not support packed document attention")
        from .kcp import apply_kcp

        patched = apply_kcp(model, mesh["cp"])
        model.set_cp_mesh(
            mesh["cp"],
            load_balancer=cfg.context_parallel_load_balancer,
            strategy="kcp",
        )
        logger.info("Applied KCP to %d KDA modules", patched)
        return model

    # Import the HF backend only when CP is enabled; disabled CP needs no
    # Transformers attention registration dependencies.
    from ...models.hf.attention_backend import ATTENTION_BACKEND

    impl = getattr(getattr(model, "model", None), "config", None)
    impl = getattr(impl, "_attn_implementation", None)
    if impl != ATTENTION_BACKEND:
        raise RuntimeError(
            f"CP requires the {ATTENTION_BACKEND!r} attention backend, but "
            f"this model runs {impl!r}. CP expresses the sharded attention mask "
            "as a BlockMask, which only the flex path consumes -- the same "
            "reason the sdpa fallback cannot run packed sequences. On a "
            "CPU-only machine the wrapper selects 'sdpa', so CP needs CUDA."
        )
    if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
        raise ValueError(
            f"cp={cfg.cp} requires a device mesh with a 'cp' axis, got "
            f"{None if mesh is None else mesh.mesh_dim_names}."
        )
    cp_mesh = mesh["cp"]
    if cp_mesh.size() != cfg.cp:
        raise ValueError(
            f"mesh 'cp' axis has size {cp_mesh.size()}, but cfg requests cp={cfg.cp}."
        )

    strategy = cfg.context_parallel_strategy
    load_balancer = cfg.context_parallel_load_balancer
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
        _require_ulysses_compatible(model, cfg=cfg, cp_mesh=ulysses_mesh)
    elif strategy == "ulysses":
        _require_ulysses_compatible(model, cfg=cfg, cp_mesh=cp_mesh)

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"apply_cp expects a HFTransformerModel (with .layers); got "
            f"{type(model).__name__}."
        )
    model_config = getattr(getattr(model, "model", None), "config", None)
    packed = getattr(model_config, "attn_mask_type", "causal") == "block_causal"
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
        if dual_axes:
            attn_mod._titan_flex_kernel = DualCPFlexKernel(
                kv_mesh=kv_mesh, ulysses_mesh=ulysses_mesh
            )
        else:
            attn_mod._titan_flex_kernel = CPFlexKernel(
                cp_mesh=cp_mesh, strategy=strategy, packed=packed
            )

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
        cfg.cp,
    )
    return model
