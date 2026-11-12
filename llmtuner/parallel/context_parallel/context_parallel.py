"""Wire CP onto a model: attach the CP flex kernel to every decoder layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.compatibility import eager_flex_parallel

from ...models.common.attention.backend_names import ATTENTION_BACKEND, CP_SDPA_BACKEND
from ...models.contracts import DecoderStructure
from ...utils.logging import get_logger
from ..attention_heads import head_counts, require_heads_divisible_by
from ..plan import ParallelPlan, PlanTarget
from ..tensor_parallel.tensor_parallel import ReplicatedAttentionSequenceBoundary
from .attention import CPFlexKernel, DualCPFlexKernel
from .compressor_attention import apply_compressor_context_parallel
from .kcp import apply_kcp
from .sdpa_attention import CPSdpaKernel, CPSdpaUlyssesKernel, DualCPSdpaKernel

if TYPE_CHECKING:
    from ..topology import ParallelTopology

logger = get_logger(__name__)

__all__ = ["apply_cp", "resolve_cp_plan"]


@dataclass(frozen=True)
class CPAttentionOptions:
    """Tensor-free choices whose changes invalidate a resolved CP target."""

    packed: bool
    strategy: str
    load_balancer: str | None
    tp_degree: int
    heads: tuple[tuple[str, int], ...]


def _cp_options(model: nn.Module, cfg: ParallelConfig) -> CPAttentionOptions:
    strategy = cfg.cp_config.context_parallel_strategy
    if strategy not in {"kv_allgather", "ulysses"}:
        raise ValueError(f"Unsupported standard CP strategy {strategy!r}")
    config = getattr(model, "config", None)
    if getattr(config, "attention_backend", "auto") == "eager_flex":
        eager_flex_parallel("CP")
    return CPAttentionOptions(
        packed=getattr(config, "attn_mask_type", "causal") == "block_causal",
        strategy=strategy,
        load_balancer=cfg.cp_config.context_parallel_load_balancer,
        tp_degree=cfg.tp_size,
        heads=tuple(head_counts(model)),
    )


def resolve_cp_plan(model: nn.Module, *, cfg: ParallelConfig) -> ParallelPlan:
    """Bind standard CP attention slots without kernels, mesh or model mutation.

    KCP and Compressor own different attention mathematics and retain their
    dedicated setup. Resolve again after TP, PP or any model/configuration edit.
    """
    if cfg.cp_config.context_parallel_strategy == "kcp" or (
        getattr(model, "context_parallel_backend", "standard") == "compressor"
    ):
        raise ValueError("Standard CP plans do not describe KCP or Compressor")
    if not callable(getattr(model, "set_cp_mesh", None)):
        raise TypeError("CP requires model-owned set_cp_mesh input semantics")
    attentions = _attention_modules(model)
    structure = model.decoder_structure
    blocks = model.get_submodule(structure.blocks)
    options = _cp_options(model, cfg)
    config = getattr(model, "config", None)
    sdpa_backend = getattr(config, "_attn_implementation", None) != ATTENTION_BACKEND
    dual_axes = (
        cfg.context_parallel_kv_allgather_size > 1
        and cfg.context_parallel_ulysses_size > 1
    )
    if dual_axes or options.strategy == "ulysses":
        _require_ulysses_compatible(
            model,
            cfg=cfg,
            cp_degree=cfg.context_parallel_ulysses_size if dual_axes else cfg.cp_size,
            attentions=attentions,
        )
    elif sdpa_backend and options.load_balancer is not None:
        raise NotImplementedError(
            "Single-axis KV SDPA CP requires context_parallel_load_balancer=None: "
            "its causal offsets assume contiguous shards, and load-balanced mask "
            "permutation is not implemented."
        )
    return ParallelPlan(
        stage="cp",
        targets=tuple(
            PlanTarget(
                f"{structure.blocks}.{block_id}.{structure.attention}",
                "attention",
                module,
                "sequence_shard",
                options,
            )
            for block_id, module in zip(blocks._modules, attentions, strict=True)
        ),
        degree=cfg.cp_size,
        degrees=(
            cfg.context_parallel_kv_allgather_size,
            cfg.context_parallel_ulysses_size,
        ),
        implementation=(
            "flex"
            if getattr(config, "_attn_implementation", None) == ATTENTION_BACKEND
            else "sdpa"
        ),
        operations=("install_attention", "bind_inputs"),
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )


def _attention_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    """Consume the explicit model's structure; never search attention aliases."""
    structure = getattr(model, "decoder_structure", None)
    if not isinstance(structure, DecoderStructure):
        raise TypeError("CP requires a model-owned DecoderStructure")
    return structure.bind_attentions(model)


def _require_ulysses_compatible(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    cp_degree: int,
    attentions: tuple[nn.Module, ...],
) -> None:
    """Reject the ulysses combinations that would train a silently wrong model.

    The all-to-all is an even split of an evenly-sharded tensor, and every
    rank ends up attending the full sequence in whatever order the shards
    arrived in -- so a load balancer (which rearranges that order) and a head
    count that does not divide tp*cp are both configuration errors, raised
    here rather than discovered from the loss curve.
    """
    load_balancer = cfg.cp_config.context_parallel_load_balancer
    if load_balancer is not None:
        raise ValueError(
            "Ulysses CP requires context_parallel_load_balancer=None: "
            "load balancing permutes tokens but the Ulysses mask expects "
            "original sequence order. Use context_parallel_strategy='kv_allgather' "
            "for load-balanced CP."
        )
    # TP shards heads first, so what ulysses must divide evenly is each
    # rank's local head count -- equivalently, the global count must divide
    # tp * cp (upstream torchtitan's head_shard_degree in its
    # config/validation.py).
    # A TP plan may leave attention replicated (e.g. MLA). Its heads have
    # not been cut by TP; only Ulysses divides them. For mixed layouts keep
    # the stricter check required by any head-sharded layer.
    replicated = bool(attentions) and all(
        isinstance(attention, ReplicatedAttentionSequenceBoundary)
        for attention in attentions
    )
    head_degree = cp_degree * (1 if replicated else cfg.tp_size)
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
    resolved_plan: ParallelPlan | None = None,
) -> nn.Module:
    """Execute a standard attention plan or a model's dedicated CP backend.

    A supplied plan must bind the current post-TP/PP model and CP options;
    execution uses its targets without repeating attention discovery.

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
    if not callable(getattr(model, "set_cp_mesh", None)):
        raise TypeError(
            "CP requires the explicit model to provide set_cp_mesh input semantics"
        )

    if resolved_plan is not None and (
        cfg.cp_config.context_parallel_strategy == "kcp"
        or getattr(model, "context_parallel_backend", "standard") == "compressor"
    ):
        raise ValueError("Standard CP plan does not match the requested backend")
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
        patched = apply_kcp(model, mesh["cp"])
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
        return apply_compressor_context_parallel(model, cp_mesh=cp_mesh, cfg=cfg)

    impl = getattr(model_config, "_attn_implementation", None)
    sdpa_backend = impl != ATTENTION_BACKEND
    if mesh is None or "cp" not in (mesh.mesh_dim_names or ()):
        raise ValueError(
            f"cp={cfg.cp_size} requires a device mesh with a 'cp' axis, got "
            f"{None if mesh is None else mesh.mesh_dim_names}."
        )
    plan = resolve_cp_plan(model, cfg=cfg) if resolved_plan is None else resolved_plan
    options = _cp_options(model, cfg)
    if (
        plan.stage != "cp"
        or plan.module_bindings is None
        or plan.parameter_bindings is None
        or plan.buffer_bindings is None
        or plan.decoder_structure is None
        or plan.operations != ("install_attention", "bind_inputs")
        or plan.degree != cfg.cp_size
        or plan.degrees
        != (
            cfg.context_parallel_kv_allgather_size,
            cfg.context_parallel_ulysses_size,
        )
        or plan.implementation != ("sdpa" if sdpa_backend else "flex")
        or any(
            target.kind != "attention"
            or target.layout != "sequence_shard"
            or target.spec != options
            for target in plan.targets
        )
    ):
        raise ValueError("CP plan does not match the current execution options")
    plan.validate_binding(model)
    # Check coverage against the immutable ownership snapshot, not a second
    # attention search. Every declared decoder slot must execute CP exactly once.
    structure = plan.decoder_structure
    expected_paths = tuple(
        f"{path}.{structure.attention}"
        for path, _ in plan.module_bindings
        if path.rpartition(".")[0] == structure.blocks
    )
    if tuple(target.path for target in plan.targets) != expected_paths:
        raise ValueError("CP plan does not cover every declared attention slot")
    attentions = tuple(target.module for target in plan.targets)
    if any(not isinstance(module, nn.Module) for module in attentions):
        raise ValueError("CP plan requires bound attention modules")
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

    packed = getattr(model_config, "attn_mask_type", "causal") == "block_causal"
    if sdpa_backend:
        if dual_axes:
            cp_sdpa_kernel = DualCPSdpaKernel(
                kv_mesh=kv_mesh, ulysses_mesh=ulysses_mesh, packed=packed
            )
        elif strategy == "ulysses":
            cp_sdpa_kernel = CPSdpaUlyssesKernel(cp_mesh=cp_mesh, packed=packed)
        else:
            cp_sdpa_kernel = CPSdpaKernel(cp_mesh=cp_mesh, packed=packed)
        model_config._attn_implementation = CP_SDPA_BACKEND
        # The wrapper reads this to decide whether the layer needs a mask handed
        # over: the kv kernel builds its own from the offsets, while the ulysses
        # kernel consumes a packed corpus's dense document mask (it attends the
        # full sequence, so the unsharded structure applies directly).
        model_config._cp_sdpa_ulysses = strategy == "ulysses"
        model_config._cp_sdpa_dual = dual_axes
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
