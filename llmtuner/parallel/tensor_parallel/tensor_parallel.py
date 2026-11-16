"""Step 2: tensor parallelism -- declarative sharding realized by fused TP GEMMs.

A tensor-parallel projection is two separate things: WHERE its weight is cut, and
HOW its activations move around the cut. Declarations live in ``sharding.py``;
projection execution lives in ``linear.py``. This entry resolves structural
targets, validates ownership and installs the selected transformations.

* ``ShardingConfig`` (built by the ``colwise()`` / ``rowwise()`` factories) is the
  DECLARATION -- plain data, no tensors, no collectives.
* ``ColumnParallelLinear`` / ``RowParallelLinear`` are the modules that REALIZE a
  declaration, using ``accelerator/linear_collectives.py`` for collective/GEMM math.
* ``apply_tp`` is the ENGINE -- it reads a plan (the model's explicit ``tp_config`` by
  default, or an explicit ``{pattern: ShardingConfig}`` map), cuts each target
  weight and swaps the projection out for its sharded module.

Weight layout, in the ``nn.Linear`` convention ``weight: [out_features,
in_features]``:

* ``colwise`` -- output features are split, i.e. the stored weight is cut on
  dim 0 into ``[out / tp, in]``. That is the ``w_shard_n = [N / R, K]`` contract
  ``AllGatherLinear`` documents: the op consumes the native layout and
  self-transposes inside the GEMM.
* ``rowwise`` -- input features are split, i.e. the stored weight is cut on dim 1,
  which is the layout ``LinearReduceScatter`` consumes as-is.

Activations stay sharded across the two: a column-parallel projection produces a
feature-sharded activation, which is exactly what the following row-parallel
projection consumes; the collectives are the sequence-parallel pair (all-gather
in, reduce-scatter out) fused into the GEMMs. This is the async-TP formulation,
not the older replicated-activation one -- the arithmetic is identical, but the
collective never materializes a full-sized activation and can overlap the matmul.

One site cannot host the fused gather: HF attention derives its q/k/v view
shapes from ``hidden_states.shape``, which a projection that physically
lengthens the sequence would silently mis-shape. Attention therefore takes the
same all-gather at the module boundary instead (``GatherSequenceFirst`` --
numerically identical, just unfused), its q/k/v projections become plain
feature-sharded GEMMs (``ColwiseLinearNoGather``), and its o_proj keeps the
fused reduce-scatter, which returns the activation to the sequence shard. The
MLP has no such shape derivation, so gate/up/down keep the fused realizers.

Scope: this is the mechanism. Model initialization and state loading are owned
by the explicit model; no fused QKV or FP8 path is assumed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError

from ...accelerator.capabilities import has
from ...accelerator.linear_collectives import (
    enable_symm_mem,
    resolve_autograd_collectives,
)
from ...models.common.moe.experts import GroupedExperts
from ...models.common.moe.layer import MoE, declared_moe_block
from ...models.contracts import DecoderStructure, GatedMLPStructure, MoEBlockContract
from ...utils.logging import get_logger
from .. import compatibility
from ..attention_heads import require_heads_divisible_by
from ..plan import (
    FullPlanBindings,
    ParallelPlan,
    PlanTarget,
    snapshot_model_bindings,
)
from ..replacements import ReplacementObserver
from ..vocab_parallel.vocab_parallel import VocabParallelEmbedding
from . import linear as tp_linear
from . import sharding as tp_sharding
from .sequence_boundary import (
    FeatureParallelSequenceBoundary,
    GatherSequenceFirst,
    ReplicatedAttentionSequenceBoundary,
    TPMoeSequenceBoundary,
)


def _is_model_declared_moe(module: nn.Module) -> bool:
    """Recognize an explicit MoE contract without inspecting class names."""
    return isinstance(module, MoE) or isinstance(
        getattr(module, "moe_contract", None), MoEBlockContract
    )


def _resolve_shared_expert(block: nn.Module):
    """Read shared expert ownership from the model contract or native MoE."""
    contract = getattr(block, "moe_contract", None)
    if isinstance(contract, MoEBlockContract):
        _, name, shared = contract.bind(block)
        return name, shared
    if isinstance(block, MoE):
        return "shared_experts", block.shared_experts
    raise ValueError("TP MoE requires a model-owned MoEBlockContract")


logger = get_logger(__name__)

__all__ = ["apply_tp", "resolve_tp_plan"]


def requires_full_token_history(module: nn.Module) -> bool:
    """Read the model's history requirement without inferring storage semantics.

    Recurrent model integrations must declare True. Absence declares no extra
    history boundary; projection attributes do not identify an attention module.
    """
    declared = getattr(module, "requires_full_token_history", None)
    if declared is not None:
        if type(declared) is not bool:
            raise ValueError("requires_full_token_history must be a bool")
        return declared
    return False


def _shared_expert_projections(
    shared: nn.Module,
) -> tuple[tuple[str, int, nn.Linear, nn.Parameter], ...]:
    """Consume declared gate/up/down roles for shared-expert feature sharding."""
    structure = getattr(shared, "mlp_structure", None)
    if not isinstance(structure, GatedMLPStructure):
        raise UnsupportedCombinationError(
            "TP shared expert requires a model-owned GatedMLPStructure"
        )
    projections = structure.bind(shared)
    if any(not isinstance(projection, nn.Linear) for projection in projections):
        raise UnsupportedCombinationError(
            "TP shared expert projection declarations must bind nn.Linear modules"
        )
    bindings = tuple(
        (path, dim, projection, projection.weight)
        for path, dim, projection in zip(
            (structure.gate, structure.up, structure.down),
            (0, 0, 1),
            projections,
            strict=True,
        )
    )
    dimensions = {}
    for _, dim, _, weight in bindings:
        if dimensions.setdefault(id(weight), dim) != dim:
            raise ValueError("TP shared projection aliases require conflicting axes")
    return bindings


def shard_shared_expert_for_tp(
    block: nn.Module,
    *,
    tp_size: int,
    tp_rank: int,
    shared_binding: tuple[str, nn.Module | None] | None = None,
) -> set[str]:
    """Feature-shard a shared expert's dense MLP -- no collectives involved.

    Inside the MoE sequence boundary the token stream is already gathered
    (full T, replicated across the TP group), so the shared expert needs no
    input collective: declared gate/up projections are sharded on the output
    features (dim 0) and the down projection on input features (dim 1), which
    makes its output partial over the TP group. The block's output -- routed
    plus shared -- is then partial on both halves, and the boundary
    reduce-scatter sums them (upstream's layout: shared w2 stays Partial
    until the boundary's single reduction).

    Only the gate/up/down layout is known; anything else keeps the refusal at
    the call site. Returns the sharded parameters' block-relative NAMES (not
    object ids: FSDP replaces the Parameter objects after this runs, and an id
    recorded now would never match again) so the trainer's replicated-gradient
    all-reduce excludes them (their gradients are complete per shard, exactly
    like the routed experts').
    """
    shared_name, shared = (
        _resolve_shared_expert(block) if shared_binding is None else shared_binding
    )
    assert shared is not None  # the caller checked
    names: set[str] = set()
    projections = _shared_expert_projections(shared)
    # Capture all originals first: gate/up may share a Parameter or the whole
    # Linear module. Never slice an already replaced projection a second time.
    shards: dict[int, nn.Parameter] = {}
    for _, dim, _, weight in projections:
        if id(weight) not in shards:
            shards[id(weight)] = nn.Parameter(
                tp_linear.shard_weight(weight, dim, tp_size=tp_size, tp_rank=tp_rank),
                requires_grad=weight.requires_grad,
            )
    # Allocation failure must preserve both weights and feature metadata.
    for name, dim, proj, weight in projections:
        proj._tp_init_specs = {
            "weight": tp_linear.TPInitializationSpec(
                type(proj), tuple(weight.shape), dim, tp_size, tp_rank
            )
        }
        proj.weight = shards[id(weight)]
        if dim == 0:
            proj.out_features = proj.weight.shape[0]
        else:
            proj.in_features = proj.weight.shape[1]
        names.add(f"{shared_name}.{name}.weight")
    return names


def expert_tp_hidden_size(block: nn.Module, *, tp_size: int) -> int:
    """Validate packed expert geometry and return the unsharded F dimension."""
    gate_up = block.experts.gate_up_proj
    down = block.experts.down_proj
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("TP expert weights must be three-dimensional")
    num_experts, double_hidden, dim = gate_up.shape
    if down.shape[0] != num_experts or down.shape[1] != dim:
        raise ValueError(
            f"unrecognized expert weight shapes for TP sharding: "
            f"gate_up_proj {tuple(gate_up.shape)}, down_proj {tuple(down.shape)}"
        )
    hidden = down.shape[2]
    if double_hidden != 2 * hidden:
        raise ValueError(
            f"gate_up_proj {tuple(gate_up.shape)} is not (E, 2F, D) against "
            f"down_proj's F={hidden}; the gate/up halves cannot be located."
        )
    if hidden % tp_size != 0:
        raise ValueError(
            f"expert hidden dim F={hidden} is not divisible by tp_size={tp_size}; "
            "each TP rank must hold the same F-shard of every expert."
        )
    return hidden


def shard_experts_for_tp(
    block: nn.Module, *, tp_size: int, tp_rank: int
) -> frozenset[str]:
    """Shard a HF MoE block's fused expert weights on the F dim, in place.

    Cuts ``experts.gate_up_proj (E, 2F, D)`` and ``experts.down_proj (E, D, F)``
    where the dense TP realizers cut their projections, so the partial sums the
    boundary reduce-scatter completes are exactly the rowwise half of the dense
    contract:

    * ``down_proj`` on dim 2 (its input features F) -- the rowwise direction.
    * ``gate_up_proj`` on dim 1, gate half and up half SEPARATELY: the block's
      forward splits the packed dim by ``chunk(2)``, and a plain contiguous cut
      of ``2F`` would slice across the gate/up boundary (HF's own
      ``packed_colwise`` style does the same per-half split).

    The parameter objects are replaced with the same attribute names, so
    ``state_dict`` FQNs do not change -- the shapes shrink, the same convention
    the dense TP realizers already follow. The router weight is not touched
    (Replicate). Returns the names of the sharded parameters so the trainer's
    replicated-gradient all-reduce can exclude them: each rank's F-shard
    gradient is already complete, and summing it with a *different* shard's
    gradient would corrupt it.
    """
    experts = block.experts
    gate_up = experts.gate_up_proj
    down = experts.down_proj
    hidden = expert_tp_hidden_size(block, tp_size=tp_size)

    gate_shard = tp_linear.shard_weight(
        gate_up[:, :hidden], 1, tp_size=tp_size, tp_rank=tp_rank
    )
    up_shard = tp_linear.shard_weight(
        gate_up[:, hidden:], 1, tp_size=tp_size, tp_rank=tp_rank
    )
    with torch.no_grad():
        sharded_gate_up = nn.Parameter(
            torch.cat([gate_shard, up_shard], dim=1).contiguous(),
            requires_grad=gate_up.requires_grad,
        )
        sharded_down = nn.Parameter(
            tp_linear.shard_weight(down, 2, tp_size=tp_size, tp_rank=tp_rank),
            requires_grad=down.requires_grad,
        )
    # Finish allocation before replacing either original parameter.
    experts._tp_init_specs = {
        "gate_up_proj": tp_linear.TPInitializationSpec(
            type(experts), tuple(gate_up.shape), 1, tp_size, tp_rank, packed=True
        ),
        "down_proj": tp_linear.TPInitializationSpec(
            type(experts), tuple(down.shape), 2, tp_size, tp_rank
        ),
    }
    experts.gate_up_proj = sharded_gate_up
    experts.down_proj = sharded_down
    return frozenset({"experts.gate_up_proj", "experts.down_proj"})


def supports_symm_mem(tp_mesh: DeviceMesh) -> bool:
    """Whether the fused symmetric-memory TP collectives can run on this mesh.

    They are CUDA-only; anywhere else the modules fall back to the functional-
    collective realization of the same math (``all_gather_linear`` /
    ``linear_reduce_scatter``), which is what makes TP runnable -- and testable
    -- on CPU/gloo.
    """
    if tp_mesh.device_type != "cuda":
        return False
    return has("symm_mem")


def tp_sharded_param_ids(model_parts: Iterable[torch.nn.Module]) -> set[int]:
    """Ids of parameters whose gradients must NOT be summed over the TP group.

    Everything not in this set is treated as TP-replicated by
    ``allreduce_replicated_tp_grads`` and summed (each rank's copy
    earns a token-partial gradient over its T/tp sequence shard). Three kinds
    of parameters are instead complete on their own rank, and summing them
    across TP would corrupt them:

    * the dense TP realizers' ``weight`` (``ColumnParallelLinear`` /
      ``RowParallelLinear`` / ``ColwiseLinearNoGather``) -- each rank owns a
      feature shard;
    * MoE-under-TP expert weights (ep=1): stacked parameters on the HF
      experts module, F-sharded in place by ``apply_tp``, which records their
      names on the block as ``tp_sharded_param_names`` (resolved to ids here,
      AFTER FSDP may have replaced the Parameter objects);
    * EP expert weights (tp x ep): ``GroupedExperts``'s ``w1/w3/w2``. Each EP
      rank owns a different slice of the expert COUNT, and its gradient is
      complete for those experts (the all-to-all dispatch feeds it every
      token routed to them); summing across TP would mix gradients of
      different experts. The MoE block's router weight is deliberately NOT
      excluded: replicated, its token-partial gradient is summed like any
      other replicated parameter.
    """
    sharded_ids = {
        id(module.weight)
        for part in model_parts
        for module in part.modules()
        if isinstance(
            module,
            tp_linear.ColumnParallelLinear
            | tp_linear.RowParallelLinear
            | tp_linear.ColwiseLinearNoGather
            | VocabParallelEmbedding,
        )
    }
    for part in model_parts:
        for module in part.modules():
            extra_names = getattr(module, "tp_sharded_param_names", None)
            if extra_names:
                # Names, resolved NOW: apply_tp records them before FSDP runs,
                # and FSDP replaces the Parameter objects, so ids captured at
                # apply time would never match again (the failure mode is a
                # silently wrong replicated-gradient all-reduce).
                sharded_ids.update(
                    id(module.get_parameter(name)) for name in extra_names
                )
            if isinstance(module, GroupedExperts):
                sharded_ids.update(id(p) for p in module.parameters(recurse=False))
    return sharded_ids


def resolve_tp_targets(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    plan: Mapping[str, str | tp_sharding.ShardingConfig] | None = None,
) -> tuple[
    list[tuple[str, nn.Linear, tp_sharding.ShardingConfig]], list[tuple[str, nn.Module]]
]:
    """Validate the TP plan and split its matches into dense and MoE targets.

    Returns ``(targets, moe_blocks)`` where
    ``targets`` are ``(module_path, nn.Linear, ShardingConfig)`` triples and
    ``moe_blocks`` are the HF MoE blocks to realize in place. A plan that
    resolves to nothing, or that matches no module, used to leave the model
    fully replicated while the run reported a healthy TP setup -- both are
    loud errors here, raised before any process-group access.
    """
    effective_plan = tp_sharding.resolve_effective_tp_plan(model, cfg=cfg, plan=plan)
    return _resolve_tp_targets(model, cfg=cfg, raw_plan=effective_plan)


def _resolve_tp_targets(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    raw_plan: Mapping[str, str | tp_sharding.ShardingConfig],
    module_bindings: tuple[tuple[str, nn.Module], ...] | None = None,
) -> tuple[
    list[tuple[str, nn.Linear, tp_sharding.ShardingConfig]], list[tuple[str, nn.Module]]
]:
    """Bind dense and expert targets from the already merged effective plan."""
    sharding_plan = tp_sharding.resolve_tp_sharding_plan(model, plan=raw_plan)
    if not sharding_plan:
        raise ValueError(
            f"apply_tp with tp={cfg.tp_size}: {type(model).__name__} provides no TP "
            "plan (neither a `tp_config` declaration nor an explicit "
            "`plan` argument). Refusing to run TP as a silently replicated "
            "model; shard the projections declaratively or set tp=1."
        )

    targets: list[tuple[str, nn.Linear, tp_sharding.ShardingConfig]] = []
    registered_modules = (
        module_bindings if module_bindings is not None else tuple(model.named_modules())
    )
    # Full ownership snapshots retain aliases; target application must see
    # each module once, using its first registered path.
    unique_modules: dict[int, tuple[str, nn.Module]] = {}
    for path, module in registered_modules:
        unique_modules.setdefault(id(module), (path, module))
    modules = tuple(unique_modules.values())
    for module_path, module in modules:
        if isinstance(module, nn.Linear):
            spec = tp_sharding.match_tp_sharding_spec(module_path, plan=sharding_plan)
            if spec is not None:
                if not module_path:
                    raise ValueError(
                        "TP cannot replace a root nn.Linear in place; wrap it in "
                        "a container module and target its child path"
                    )
                targets.append((module_path, module, spec))

    # MoE-under-TP is declared by spec strings that name no nn.Linear (the
    # expert weights are stacked parameters), so it has to be detected on the
    # RAW plan, before resolve_tp_sharding_plan maps those specs to None.
    plan_declares_moe = any(
        isinstance(spec, str) and spec in tp_sharding.MOE_PLAN_SPECS
        for spec in raw_plan.values()
    )
    structure = getattr(model, "decoder_structure", None)
    declared_moe_ids: set[int] = set()
    if isinstance(structure, DecoderStructure):
        for layer in model.get_submodule(structure.blocks):
            declaration = declared_moe_block(layer)
            if declaration is None:
                if plan_declares_moe:
                    raise ValueError(
                        "Explicit TP decoder requires moe_block_name "
                        "(None for a dense layer)"
                    )
                continue
            name, block = declaration
            if block is not None:
                declared_moe_ids.add(id(block))
            if (
                plan_declares_moe
                and block is not None
                and not _is_model_declared_moe(block)
            ):
                raise UnsupportedCombinationError(
                    f"Declared MoE slot {type(layer).__name__}.{name} requires "
                    "a model-owned MoEBlockContract for TP planning"
                )
    # tp x ep: TP shards only the dense parts and EP owns the routed experts
    # (the upstream alignment). The HF blocks are left whole here and swapped
    # for the native MoE stack by apply_ep, which runs next; the swapped block
    # consumes and produces the T/tp sequence shard directly (all-to-all
    # dispatch), so no sequence-boundary collectives are installed either.
    moe_deferred_to_ep = plan_declares_moe and cfg.ep_size > 1
    # Explicit models own sparse-block declarations. Never import/probe legacy
    # HF layouts while walking their dense children or unrelated modules.
    is_moe_block = (
        (lambda module: id(module) in declared_moe_ids)
        if isinstance(structure, DecoderStructure)
        else _is_model_declared_moe
    )
    moe_blocks: list[tuple[str, nn.Module]] = []
    bracketed_paths: list[str] = []
    if plan_declares_moe and not moe_deferred_to_ep:
        for module_path, module in modules:
            if isinstance(module, FeatureParallelSequenceBoundary):
                # Already bracketed by an earlier apply_tp pass (the engine is
                # idempotent per module): still counts as realized MoE TP.
                bracketed_paths.append(module_path)
                continue
            # Only model-declared mathematics enters the MoE TP path.
            if is_moe_block(module):
                moe_blocks.append((module_path, module))
        if not moe_blocks and not bracketed_paths:
            compatibility.tp_moe_specs_without_block(cfg.tp_size, model)

    # MoE-block internals never take the dense realizer path. The routed
    # experts are stacked parameters on the fused layout (no nn.Linear to
    # match) but a layout that keeps them as per-expert ``nn.Linear``s
    # (transformers 4.x with ``experts_implementation="eager"``) exposes them
    # to the default plan's ``mlp.experts.*`` patterns, and the shared expert
    # is sharded featurewise by ``shard_shared_expert_for_tp`` below,
    # collective-free inside the sequence boundary. HF plans like
    # DeepSeek-V3's still declare ``shared_experts.*_proj: colwise`` --
    # wrapping either in a dense realizer double-shards the weight (F/tp**2 at
    # the wrong offset) and double-gathers the sequence, silently computing
    # the wrong values.
    #
    # This has to hold whenever the block's experts are owned by *someone* --
    # also when the plan declares no MoE spec at all, or when the MoE
    # realization itself is deferred to ``apply_ep`` (``tp x ep``: EP owns the
    # routed experts). Missing that case left the eager per-expert projections
    # matched by the dense plan: they were F-sharded (F -> F/tp) and then the
    # EP swap moved those partial weights into ``GroupedExperts``, dropping the
    # realizers whose collectives would have summed the partials. Every expert
    # then contributed half of its output, silently.
    #
    # Already realized blocks retain plain Linear shared projections; exclude
    # their subtrees on reapplication as well.
    moe_block_paths = [path for path, _ in moe_blocks]
    moe_block_paths.extend(
        path
        for path, module in modules
        if path not in moe_block_paths
        and is_moe_block(module)
        and not isinstance(module, FeatureParallelSequenceBoundary)
    )
    if moe_block_paths or bracketed_paths:
        moe_prefixes = tuple(
            f"{p}." if p else "" for p in [*bracketed_paths, *moe_block_paths]
        )
        targets = [t for t in targets if not t[0].startswith(moe_prefixes)]
    nothing_sharded = not targets and not moe_blocks and not bracketed_paths
    if (
        nothing_sharded
        and not moe_deferred_to_ep
        and not tp_sharding.resolve_vocab_parallel_targets(
            model, raw_plan, module_bindings=modules
        )
    ):
        raise ValueError(
            f"apply_tp with tp={cfg.tp_size}: the plan patterns "
            f"{sorted(sharding_plan)} matched no nn.Linear or nn.Embedding on "
            f"{type(model).__name__}. The patterns are spelled relative to the "
            "module tree being parallelized; check the prefix (e.g. a wrapper's "
            "`model.` prefix) rather than training a replicated model by "
            "mistake."
        )
    return targets, moe_blocks


def validate_moe_tp_target(
    module_path: str,
    block: nn.Module,
    *,
    tp_size: int,
    shared_binding: tuple[str, nn.Module | None],
) -> None:
    """Validate routed/shared expert geometry before any model mutation."""
    _, shared = shared_binding
    expert_tp_hidden_size(block, tp_size=tp_size)
    if shared is not None:
        # Declared gate/up/down roles shard featurewise without collectives:
        # the outer MoE boundary already gathered the token sequence.
        if not isinstance(getattr(shared, "mlp_structure", None), GatedMLPStructure):
            compatibility.shared_expert_tp(module_path, block)
        projections = _shared_expert_projections(shared)
        gate_shape, up_shape, down_shape = (
            tuple(weight.shape) for _, _, _, weight in projections
        )
        model_dim = block.experts.gate_up_proj.shape[-1]
        if (
            len(gate_shape) != 2
            or gate_shape[1] != model_dim
            or up_shape != gate_shape
            or down_shape != (model_dim, gate_shape[0])
        ):
            raise ValueError(
                f"TP shared expert {module_path!r} requires gate/up (F, D) "
                f"and down (D, F), with D={model_dim}; got gate={gate_shape}, "
                f"up={up_shape}, down={down_shape}"
            )
        for name, dim, projection, _ in projections:
            if projection.bias is not None:
                raise ValueError(
                    f"TP shared expert {module_path}.{name} has an unsupported bias"
                )
            if projection.weight.shape[dim] % tp_size:
                raise ValueError(
                    f"TP shared expert {module_path}.{name} weight dim {dim} "
                    f"is not divisible by tp_size={tp_size}"
                )


def realize_moe_blocks_tp(
    moe_blocks: list[tuple[str, nn.Module]],
    *,
    tp_size: int,
    tp_rank: int,
    group,
    shared_bindings: Mapping[nn.Module, tuple[str, nn.Module | None]] | None = None,
) -> None:
    """Shard each HF MoE block's experts and shared expert in place, and give
    the block the TP sequence-boundary class.

    MoE-block internals never take the dense realizer path: the routed experts
    are stacked parameters (no nn.Linear to match), and the shared expert is
    sharded featurewise by ``shard_shared_expert_for_tp``, collective-free
    inside the sequence boundary.
    """
    bindings = (
        {block: _resolve_shared_expert(block) for _, block in moe_blocks}
        if shared_bindings is None
        else shared_bindings
    )
    if shared_bindings is None:
        for module_path, block in moe_blocks:
            validate_moe_tp_target(
                module_path, block, tp_size=tp_size, shared_binding=bindings[block]
            )

    for _, block in moe_blocks:
        _, shared = bindings[block]
        shared_names: set[str] = set()
        if shared is not None:
            shared_names = shard_shared_expert_for_tp(
                block, tp_size=tp_size, tp_rank=tp_rank, shared_binding=bindings[block]
            )
        # The sharded expert parameters are excluded from the trainer's
        # replicated-gradient all-reduce through this NAME set: each rank's
        # F-shard gradient is complete, and summing it with a different
        # shard's gradient would corrupt it. Names, not ids -- FSDP runs after
        # this and replaces the Parameter objects.
        block.tp_sharded_param_names = (
            shard_experts_for_tp(block, tp_size=tp_size, tp_rank=tp_rank) | shared_names
        )
        block._tp_seq_group = group
        # An explicitly selected kernel can install forward on the instance
        # (e.g. torchtitan NPU experts). It would shadow the boundary mixin and
        # silently bypass both sequence collectives after the class swap.
        if "forward" in block.__dict__:
            block._tp_moe_original_forward = block.__dict__.pop("forward")
        block.__class__ = type(
            f"TPMoe{type(block).__name__}",
            (TPMoeSequenceBoundary, type(block)),
            {},
        )


def validate_tp_model_options(model: nn.Module) -> None:
    """Reject incompatible model policies even before mesh availability checks."""
    for module in model.modules():
        validate = getattr(module, "validate_tensor_parallel", None)
        if validate is not None:
            if not callable(validate):
                raise TypeError("validate_tensor_parallel must be callable")
            validate()


def resolve_tp_plan(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    plan: Mapping[str, str | tp_sharding.ShardingConfig] | None = None,
    bindings: FullPlanBindings | None = None,
) -> ParallelPlan:
    """Resolve and validate TP locally, without a mesh or model mutation.

    The assembly layer may coordinate rank-local failures around this function.
    Existing model declarations and explicit overrides retain their precedence.
    Borrowed bindings must describe this unchanged model instance; assembly can
    reuse them across pure preflights, but never across transformations.
    """
    if cfg.tp_size <= 1:
        raise ValueError("TP plan resolution requires tp_size > 1")
    validate_tp_model_options(model)
    validate = getattr(model, "validate_parallel", None)
    # Model validators describe their full default plan. Explicit partial
    # overrides instead use the target-level checks below. Reapplication sees
    # already-sharded storage and must not validate it as an unsharded model.
    if (
        callable(validate)
        and plan is None
        and not cfg.tp_config.tp_colwise_modules
        and not cfg.tp_config.tp_rowwise_modules
        and getattr(model, "parallel_plan", None) is None
    ):
        validate(tp_degree=cfg.tp_size, ep_degree=cfg.ep_size)
    # Resolve targets once before accessing process groups or changing weights.
    raw_plan = tp_sharding.resolve_effective_tp_plan(model, cfg=cfg, plan=plan)
    bindings = snapshot_model_bindings(model) if bindings is None else bindings
    # Keep the first registered path, matching named_modules() traversal, while
    # retaining every alias in the ownership snapshot handed to ParallelPlan.
    unique_modules: dict[int, tuple[str, nn.Module]] = {}
    for path, module in bindings["module_bindings"]:
        unique_modules.setdefault(id(module), (path, module))
    modules = tuple(unique_modules.values())
    targets, moe_blocks = _resolve_tp_targets(
        model, cfg=cfg, raw_plan=raw_plan, module_bindings=bindings["module_bindings"]
    )
    vocab_targets = tp_sharding.resolve_vocab_parallel_targets(
        model, raw_plan, module_bindings=bindings["module_bindings"]
    )

    # A head count that does not divide ``tp`` has no valid local head split,
    # and no downstream guard catches it: the projection-level check in
    # ``shard_weight`` only sees the feature dim, which 8 KV heads at
    # head_dim=128 (1024 features) satisfy at tp=16. Upstream rejects this at
    # config parse (upstream torchtitan ``config/validation.py``'s
    # ``head_shard_degree``); here the
    # model's own config is the first place the counts exist.
    structure = getattr(model, "decoder_structure", None)
    declared_attentions: set[int] | None = None
    if structure is not None and not isinstance(structure, DecoderStructure):
        raise TypeError("TP decoder_structure must be a DecoderStructure declaration")
    if isinstance(structure, DecoderStructure) and structure.attention is not None:
        declared_attentions = {
            id(module) for module in structure.bind_attentions(model)
        }
        attention_paths = [
            path for path, module in modules if id(module) in declared_attentions
        ]
    else:
        # Explicit projection-only plans have no attention semantics. Model
        # integrations declare their attention slots through DecoderStructure.
        attention_paths = []
    attention_is_sharded = any(
        any(path.startswith(f"{attention}.") for attention in attention_paths)
        for path, _, _ in targets
    )
    if not attention_paths or attention_is_sharded:
        require_heads_divisible_by(
            model,
            size=cfg.tp_size,
            axis="tp",
            why=(
                "tensor parallelism shards attention heads across the TP group, so "
                "each rank must hold a whole number of heads"
            ),
        )

    # Validate every dense target before replacing any parameters or modules.
    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in bindings["parameter_bindings"]:
        parameter_paths.setdefault(id(parameter), []).append(name)
    # MoE slicing also replaces Parameters. A shared expert module would be
    # sliced twice, while individually tied weights would silently lose their
    # identity. Permit only the same shared MLP's gate/up tie: both use the
    # identical column slice, and the shared realizer preserves that identity.
    # Reject all other aliases before any constructor or MoE mutation runs.
    for module_path, block in moe_blocks:
        shared_name, shared = _resolve_shared_expert(block)
        prefix = f"{module_path}." if module_path else ""
        shared_structure = getattr(shared, "mlp_structure", None)
        tied_shared_paths = (
            {
                f"{prefix}{shared_name}.{name}.weight"
                for name in (shared_structure.gate, shared_structure.up)
            }
            if isinstance(shared_structure, GatedMLPStructure)
            else set()
        )
        sharded_parameters = list(block.experts.parameters())
        if shared is not None:
            sharded_parameters.extend(shared.parameters())
        for parameter in sharded_parameters:
            aliases = parameter_paths[id(parameter)]
            if len(aliases) > 1 and set(aliases) != tied_shared_paths:
                raise ValueError(
                    f"TP over MoE {module_path!r} cannot preserve shared weight "
                    f"paths {aliases!r}; shared MoE shards require an "
                    "alias-preserving TP implementation"
                )
    for module_path, embedding in vocab_targets:
        if embedding.max_norm is not None:
            raise ValueError(
                f"Vocab parallel over {module_path} does not support max_norm"
            )
        if embedding.scale_grad_by_freq:
            raise ValueError(
                f"Vocab parallel over {module_path} does not support scale_grad_by_freq"
            )
        if embedding.sparse:
            raise ValueError(
                f"Vocab parallel over {module_path} does not support sparse gradients"
            )
        if len(parameter_paths[id(embedding.weight)]) > 1:
            raise ValueError(
                f"Vocab parallel over {module_path} cannot preserve tied weights; "
                "disable vocabulary sharding or untie the embedding first"
            )
        if embedding.num_embeddings % cfg.tp_size:
            raise ValueError(
                f"Vocab parallel over {module_path}: num_embeddings "
                f"({embedding.num_embeddings}) is not divisible by "
                f"tp_size={cfg.tp_size}"
            )
    for module_path, inner, spec in targets:
        aliases = parameter_paths[id(inner.weight)]
        if len(aliases) > 1:
            raise ValueError(
                f"TP over {module_path} cannot preserve shared weight paths "
                f"{aliases!r}; leave these projections replicated"
            )
        if inner.bias is not None:
            raise ValueError(
                f"TP over {module_path} has a bias, which this minimal engine does "
                "not shard; HF decoder projections are bias-free."
            )
        dimension = 0 if spec.kind == "colwise" else 1
        size = inner.weight.shape[dimension]
        if size % cfg.tp_size:
            raise ValueError(
                f"TP over {module_path}: weight dim {dimension} (size {size}) "
                f"is not divisible by tp_size={cfg.tp_size}"
            )
    replicated_attention = [
        module
        for path, module in modules
        if (path in attention_paths or requires_full_token_history(module))
        and not isinstance(
            module, GatherSequenceFirst | ReplicatedAttentionSequenceBoundary
        )
        and not any(target.startswith(f"{path}.") for target, _, _ in targets)
    ]
    # GDN's convolution and recurrence need the complete token history, even
    # though HF's colwise_gather_output projections stay replicated here.
    # Head-sharded GDN requires a different boundary and is not implemented.
    for path, module in modules:
        if requires_full_token_history(module) and any(
            target.startswith(f"{path}.") for target, _, _ in targets
        ):
            raise ValueError("TP linear_attn requires replicated GDN projections")

    # Attention gathers at its input boundary. Inner column projections must
    # retain those rows, and row projections scatter them back to the residual.
    # Only the default fused column GEMM is rewritten; custom realizers must
    # explicitly declare compatible activation semantics.
    attention_parents: dict[str, nn.Module] = {}
    realized_targets = []
    for module_path, inner, spec in targets:
        parent_path, _, _ = module_path.rpartition(".")
        parent = model.get_submodule(parent_path)
        is_attention = (
            declared_attentions is not None and id(parent) in declared_attentions
        )
        if is_attention and isinstance(parent, ReplicatedAttentionSequenceBoundary):
            raise UnsupportedCombinationError(
                "Cannot shard projections inside an installed replicated attention "
                "boundary; rebuild the model with the final TP plan"
            )
        if spec.kind == "colwise" and is_attention:
            if spec.implementation is tp_linear.ColumnParallelLinear:
                spec = replace(
                    spec,
                    implementation=tp_linear.ColwiseLinearNoGather,
                    sequence_transform="preserve",
                )
            if spec.sequence_transform != "preserve":
                raise ValueError(
                    f"Attention TP projection {module_path!r} must preserve token "
                    "rows after the attention boundary gather"
                )
            attention_parents[parent_path] = parent
        if (
            is_attention
            and spec.kind == "rowwise"
            and spec.sequence_transform != "scatter"
        ):
            raise ValueError(
                f"Attention TP projection {module_path!r} must scatter token rows "
                "back to the sequence-sharded residual"
            )
        realized_targets.append((module_path, inner, spec))

    shared_bindings = {}
    for path, block in moe_blocks:
        binding = _resolve_shared_expert(block)
        validate_moe_tp_target(path, block, tp_size=cfg.tp_size, shared_binding=binding)
        shared_bindings[block] = binding

    sequence_metadata = []
    module_paths = {}
    for path, module in modules:
        module_paths[id(module)] = path
        configure = getattr(module, "configure_sequence_parallel", None)
        if configure is not None:
            if not callable(configure):
                raise TypeError("configure_sequence_parallel must be callable")
            sequence_metadata.append(
                PlanTarget(path, "decoder", module, "sequence_parallel", configure)
            )
    return ParallelPlan(
        stage="tp",
        targets=(
            *sequence_metadata,
            *(
                PlanTarget(path, "dense", module, spec.kind, spec)
                for path, module, spec in realized_targets
            ),
            *(
                PlanTarget(
                    path, "moe", module, "fused_gate_up", shared_bindings[module]
                )
                for path, module in moe_blocks
            ),
            *(
                PlanTarget(path, "vocab", module, "embedding_rowwise")
                for path, module in vocab_targets
            ),
            *(
                PlanTarget(path, "attention", parent, "gather")
                for path, parent in attention_parents.items()
            ),
            *(
                PlanTarget(
                    module_paths[id(attention)],
                    "attention",
                    attention,
                    "replicate_tokens",
                )
                for attention in replicated_attention
            ),
        ),
        degree=cfg.tp_size,
        implementation="unbound",
        token_history_requirements=tuple(
            (path, getattr(module, "requires_full_token_history", None))
            for path, module in modules
        ),
        **bindings,
    )


def apply_tp(
    model: nn.Module,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
    *,
    plan: Mapping[str, str | tp_sharding.ShardingConfig] | None = None,
    resolved_plan: ParallelPlan | None = None,
    on_replacement: ReplacementObserver | None = None,
) -> nn.Module:
    """Tensor-parallelize ``model`` in place. No-op when ``tp == 1``.

    ``plan`` defaults to the model's explicit ``tp_config`` (HF ships one for Qwen3,
    Llama, ...). Pass an explicit ``{path_pattern: ShardingConfig}`` to override
    it -- e.g. to leave a projection replicated or to use a different realizer.
    """
    if cfg.tp_size <= 1:
        return model
    # Validate model geometry before touching the mesh or resolving targets.
    # This keeps configuration errors deterministic even for callers that pass
    # a sentinel or partially constructed mesh during preflight.
    validate_tp_model_options(model)
    if mesh is None:
        raise ValueError(f"tp={cfg.tp_size} requires a device mesh with a 'tp' axis")
    if resolved_plan is not None and plan is not None:
        raise ValueError("Pass either plan or resolved_plan, not both")
    resolved = (
        resolve_tp_plan(model, cfg=cfg, plan=plan)
        if resolved_plan is None
        else resolved_plan
    )
    if resolved.stage != "tp":
        raise ValueError(f"Expected a TP plan, got stage {resolved.stage!r}")
    if resolved.degree != cfg.tp_size:
        raise ValueError("Resolved TP degree does not match configuration")
    resolved.validate_binding(model)
    targets = tuple(
        (target.path, target.module, target.spec)
        for target in resolved.targets
        if target.kind == "dense"
    )
    moe_blocks = tuple(
        (target.path, target.module)
        for target in resolved.targets
        if target.kind == "moe"
    )
    vocab_targets = tuple(
        (target.path, target.module)
        for target in resolved.targets
        if target.kind == "vocab"
    )
    attention_parents = tuple(
        target.module
        for target in resolved.targets
        if target.kind == "attention" and target.layout == "gather"
    )
    replicated_attention = tuple(
        target.module
        for target in resolved.targets
        if target.kind == "attention" and target.layout == "replicate_tokens"
    )

    sequence_metadata = tuple(
        target
        for target in resolved.targets
        if target.kind == "decoder" and target.layout == "sequence_parallel"
    )
    for target in sequence_metadata:
        if not callable(target.spec) or (
            getattr(target.module, "configure_sequence_parallel", None) != target.spec
        ):
            raise ValueError(
                "Resolved sequence parallel configuration binding is stale"
            )

    tp_size = mesh["tp"].size()
    if tp_size != cfg.tp_size:
        raise ValueError(
            f"TP mesh size {tp_size} does not match configured tp={cfg.tp_size}"
        )
    use_symm_mem = supports_symm_mem(mesh["tp"])
    execution_plan = replace(
        resolved, implementation="symm_mem" if use_symm_mem else "functional"
    )
    execution_plan.validate_binding(model)
    group = mesh["tp"].get_group()
    tp_rank = mesh["tp"].get_local_rank()
    if use_symm_mem:
        enable_symm_mem(group)

    # Resolve optional imports before tensor graphs are compiled.
    resolve_autograd_collectives()

    # Prepare every dense replacement before changing the model. A failing
    # custom constructor must not leave earlier projections or MoE blocks sharded.
    # Constructors are responsible for avoiding side effects on their inputs.
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    # Deepest paths first, so replacing a module never hides an inner target.
    for module_path, inner, spec in sorted(targets, key=lambda t: -t[0].count(".")):
        parent_path, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        wrapped = spec.implementation(
            inner.weight,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group=group,
            use_symm_mem=use_symm_mem,
        )
        spec.validate_implementation(wrapped)
        weight = wrapped._parameters.get("weight")
        if not isinstance(weight, nn.Parameter):
            raise TypeError(
                f"TP replacement {module_path!r} must register a weight parameter"
            )
        parameter_paths = tuple(
            name for name, _ in wrapped.named_parameters(remove_duplicate=False)
        )
        if parameter_paths != ("weight",):
            raise ValueError(
                f"TP replacement {module_path!r} must expose only its weight "
                f"parameter; found registrations {parameter_paths!r}. "
                "Additional parameters require an explicit transformation contract"
            )
        expected_shape = list(inner.weight.shape)
        expected_shape[0 if spec.kind == "colwise" else 1] //= tp_size
        if tuple(weight.shape) != tuple(expected_shape):
            raise ValueError(
                f"TP replacement {module_path!r} weight shape {tuple(weight.shape)} "
                f"does not match declared shard {tuple(expected_shape)}"
            )
        if (
            weight.dtype != inner.weight.dtype
            or weight.device != inner.weight.device
            or weight.requires_grad != inner.weight.requires_grad
        ):
            raise ValueError(
                f"TP replacement {module_path!r} must preserve weight dtype, "
                "device and requires_grad"
            )
        wrapped.train(inner.training)
        wrapped._tp_init_specs = {
            "weight": tp_linear.TPInitializationSpec(
                type(inner),
                tuple(inner.weight.shape),
                0 if spec.kind == "colwise" else 1,
                tp_size,
                tp_rank,
            )
        }
        replacements.append((parent, attr, wrapped))

    local_vocab_size = 0
    for module_path, inner in vocab_targets:
        parent_path, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        local_vocab_size = inner.num_embeddings // tp_size
        vocab_start = tp_rank * local_vocab_size
        wrapped = VocabParallelEmbedding(
            tp_linear.shard_weight(inner.weight, 0, tp_size=tp_size, tp_rank=tp_rank),
            source=inner,
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_start + local_vocab_size,
            tp_group=group,
        )
        wrapped.train(inner.training)
        wrapped._tp_init_specs = {
            "weight": tp_linear.TPInitializationSpec(
                type(inner),
                tuple(inner.weight.shape),
                0,
                tp_size,
                tp_rank,
            )
        }
        replacements.append((parent, attr, wrapped))

    realize_moe_blocks_tp(
        moe_blocks,
        tp_size=tp_size,
        tp_rank=tp_rank,
        group=group,
        shared_bindings={
            target.module: target.spec
            for target in resolved.targets
            if target.kind == "moe"
        },
    )
    for parent, attr, wrapped in replacements:
        setattr(parent, attr, wrapped)

    for parent in attention_parents:
        if isinstance(parent, GatherSequenceFirst):
            continue  # already gathered (apply_tp is idempotent per module)
        parent._tp_seq_group = group
        if "forward" in parent.__dict__:
            parent._tp_attention_original_forward = parent.__dict__.pop("forward")
        parent.__class__ = type(
            f"TPGather{type(parent).__name__}",
            (GatherSequenceFirst, type(parent)),
            {},
        )

    for attention in replicated_attention:
        attention._tp_seq_group = group
        if "forward" in attention.__dict__:
            attention._tp_attention_original_forward = attention.__dict__.pop("forward")
        attention.__class__ = type(
            f"TPReplicated{type(attention).__name__}",
            (ReplicatedAttentionSequenceBoundary, type(attention)),
            {},
        )

    layouts = model.__dict__.setdefault("_tp_checkpoint_layouts", {})
    for path, _, spec in targets:
        layouts[f"{path}.weight"] = (0 if spec.kind == "colwise" else 1, False, group)
    for path, _ in vocab_targets:
        layouts[f"{path}.weight"] = (0, False, group)
    for path, block in moe_blocks:
        prefix = f"{path}." if path else ""
        layouts[f"{prefix}experts.gate_up_proj"] = (1, True, group)
        layouts[f"{prefix}experts.down_proj"] = (2, False, group)
        shared_name, shared = _resolve_shared_expert(block)
        if shared is not None:
            for name, dim, _, _ in _shared_expert_projections(shared):
                layouts[f"{prefix}{shared_name}.{name}.weight"] = (dim, False, group)

    for target in sequence_metadata:
        target.spec(degree=tp_size, rank=tp_rank)

    if on_replacement is not None:
        # TP preserves FQNs, including packed expert parameters. Resolve these
        # exact paths once after installation and report identity changes to
        # downstream FSDP bindings; do not infer replacements by shape.
        modules = {
            source: model.get_submodule(path)
            for path, source in execution_plan.module_bindings or ()
            if model.get_submodule(path) is not source
        }
        parameters = {
            source: (model.get_parameter(path),)
            for path, source in execution_plan.parameter_bindings or ()
            if model.get_parameter(path) is not source
        }
        on_replacement(modules, parameters)
    model.parallel_plan = execution_plan
    logger.info(
        "Applied TP degree %d: dense=%d, MoE=%d, vocab=%d, collective=%s",
        execution_plan.degree,
        execution_plan.count("dense"),
        execution_plan.count("moe"),
        execution_plan.count("vocab"),
        execution_plan.implementation,
    )
    return model
