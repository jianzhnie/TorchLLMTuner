"""Step 2: tensor parallelism -- declarative sharding realized by fused TP GEMMs.

A tensor-parallel projection is two separate things: WHERE its weight is cut, and
HOW its activations move around the cut. This file keeps those apart.

* ``ShardingConfig`` (built by the ``colwise()`` / ``rowwise()`` factories) is the
  DECLARATION -- plain data, no tensors, no collectives.
* ``ColumnParallelLinear`` / ``RowParallelLinear`` are the modules that REALIZE a
  declaration, built on the fused collective+GEMM primitives in ``linear_ops.py``.
* ``apply_tp`` is the ENGINE -- it reads a plan (the model's HF ``tp_config`` by
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

Scope: this is the mechanism. No meta-init (weights come from the HF model as
usual), no fused QKV (HF keeps q/k/v as separate projections), no FP8.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.parallel_config import ParallelConfig

from ...accelerator.capabilities import has
from ...models.common.moe.experts import GroupedExperts
from .. import compatibility
from ..attention_heads import require_heads_divisible_by
from ..module_patterns import module_name_matches, resolve_module_patterns
from .linear_ops import (
    AllGatherLinear,
    LinearReduceScatter,
    all_gather_along,
    all_gather_linear,
    functional_collectives,
    linear_reduce_scatter,
    reduce_scatter_along,
)

__all__ = [
    "ColumnParallelLinear",
    "ColwiseLinearNoGather",
    "RowParallelLinear",
    "ShardingConfig",
    "apply_tp",
    "colwise",
    "rowwise",
]

ShardKind = Literal["colwise", "rowwise"]


# -- declaration --------------------------------------------------------------


def shard_weight(
    weight: torch.Tensor, dim: int, *, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Cut ``weight`` into ``tp_size`` pieces along ``dim``; keep this rank's.

    A plain slice with no collective: every rank starts from the same full weight
    (the model is built identically everywhere), so each just drops the rest.
    """
    if weight.shape[dim] % tp_size != 0:
        raise ValueError(
            f"weight dim {dim} (size {weight.shape[dim]}) is not divisible by "
            f"tp_size={tp_size}"
        )
    return torch.chunk(weight.detach(), tp_size, dim=dim)[tp_rank].contiguous()


class ColumnParallelLinear(nn.Module):
    """Column-parallel projection: output features split across TP ranks.

    Stores the weight as ``[out_features / tp, in_features]`` -- HF's own
    ``[out, in]`` layout cut on dim 0, which is the ``w_shard_n = [N / R, K]``
    layout ``AllGatherLinear`` contracts for (the op self-transposes).

    forward all-gathers the sequence shard and leaves the activation feature-
    sharded; backward is the dual (reduce-scatter of the input gradient, local
    weight gradient).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            shard_weight(weight, 0, tp_size=tp_size, tp_rank=tp_rank),
            requires_grad=weight.requires_grad
            if isinstance(weight, nn.Parameter)
            else True,
        )
        self.group = group
        self.tp_size = tp_size
        self.use_symm_mem = use_symm_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The collective primitives are strictly 2D (sequence-major rows), but
        # the HF decoder feeds [B, T, K] hidden states: fold the leading dims
        # and restore them after, with the row count multiplied by tp_size --
        # that is the all-gathered sequence length.
        if x.ndim not in (2, 3):
            raise ValueError("TP projection input must be [T, K] or [1, T, K]")
        if x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(
                "TP realizers gather/reduce rows in rank order; a batched "
                f"[B, T, K] input with B={x.shape[0]} would interleave batches "
                "across ranks after the fold. Flatten to [T, K] or keep B=1 "
                "(the HF wrapper always runs B=1)."
            )
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = AllGatherLinear.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = all_gather_linear(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] * self.tp_size, y_2d.shape[-1])


class RowParallelLinear(nn.Module):
    """Row-parallel projection: input features split across TP ranks.

    Stores the weight as ``[out_features, in_features / tp]`` -- HF's layout cut
    on dim 1, which is what ``LinearReduceScatter`` expects.

    forward multiplies the feature-sharded activation and reduce-scatters the
    partial sums back to a sequence shard; backward is the dual (all-gather of the
    output gradient, local weight gradient).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            shard_weight(weight, 1, tp_size=tp_size, tp_rank=tp_rank),
            requires_grad=weight.requires_grad
            if isinstance(weight, nn.Parameter)
            else True,
        )
        self.group = group
        self.tp_size = tp_size
        self.use_symm_mem = use_symm_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror of ColumnParallelLinear.forward: fold to 2D for the collective, then
        # restore the leading dims with the row count divided by tp_size --
        # that is the reduce-scattered sequence shard.
        if x.ndim not in (2, 3):
            raise ValueError("TP projection input must be [T, K] or [1, T, K]")
        if x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(
                "TP realizers gather/reduce rows in rank order; a batched "
                f"[B, T, K] input with B={x.shape[0]} would interleave batches "
                "across ranks after the fold. Flatten to [T, K] or keep B=1 "
                "(the HF wrapper always runs B=1)."
            )
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = LinearReduceScatter.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = linear_reduce_scatter(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] // self.tp_size, y_2d.shape[-1])


class ColwiseLinearNoGather(nn.Module):
    """Column-parallel projection without the fused sequence all-gather.

    Same weight shard as :class:`ColumnParallelLinear` (``[out / tp, in]``) but a plain
    local GEMM, for sites whose input is already full-sequence: the attention
    boundary gather (``GatherSequenceFirst``) runs upstream, because HF
    attention derives q/k/v shapes from ``hidden_states`` and cannot absorb a
    projection whose output is physically longer than its input.

    The backward is exact without any collective here: the rowwise o_proj's
    backward all-gather reassembles the full-sequence, total-loss output
    gradient before attention's backward runs, so the local ``dy.T @ x`` is the
    complete weight gradient for this rank's feature shard.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        del group, use_symm_mem  # no collective of its own; kept for the engine
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = nn.Parameter(
            shard_weight(weight, 0, tp_size=tp_size, tp_rank=tp_rank),
            requires_grad=weight.requires_grad
            if isinstance(weight, nn.Parameter)
            else True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class GatherSequenceFirst:
    """Mixin that all-gathers the TP sequence shard before HF attention runs.

    Installed by ``apply_tp`` via a ``__class__`` swap (not a module wrapper),
    so module paths, ``state_dict`` keys and later attach points (``apply_cp``'s
    ``_titan_flex_kernel``) are all untouched. ``hidden_states`` arrives as this
    rank's ``[B, T / tp, K]`` sequence shard and is gathered along the sequence
    dim to the length the inner forward expects -- the full sequence, or the CP
    shard when CP is on (the gather spans the TP group only, and a TP group
    collectively holds exactly one CP shard).

    The gather's autograd dual is a reduce-scatter, which sums each rank's
    input-gradient contribution back to its own token shard -- the exact dual
    the sequence-parallel layout needs.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_attention_original_forward")
        return (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )


def looks_like_attention(module: nn.Module) -> bool:
    """HF attention modules hold q/k/v projections as direct attributes.

    This is the site test for the boundary gather: such a module reshapes its
    projections' outputs by the input's shape, so the gather must happen before
    it, not inside the projections. Probed structurally rather than by class
    name because HF spells the class differently per family (``LlamaAttention``,
    ``Qwen3Attention``, ...).
    """
    return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))


class ReplicatedAttentionSequenceBoundary:
    """Run unsharded attention on the gathered sequence, return local tokens.

    Some HF plans shard only MLPs (e.g. DeepSeek MLA). Attention still needs
    the full sequence and RoPE positions. Its output is replicated, so slice
    tokens without summing; the input gather sums partial gradients in backward.
    Replicated attention parameter gradients are reduced by the trainer.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_attention_original_forward")
        output = (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )
        attention = output[0] if isinstance(output, tuple) else output
        local = attention.chunk(self._tp_seq_group.size(), dim=-2)[
            self._tp_seq_group.rank()
        ].contiguous()
        return (local, *output[1:]) if isinstance(output, tuple) else local


# -- MoE under TP ---------------------------------------------------------------

# The HF tp_config spec strings that declare MoE-under-TP (transformers 5.x
# spelling, e.g. Qwen3Moe's ``base_model_tp_config``). They name no nn.Linear --
# the expert weights are stacked parameters on the experts module -- so they
# resolve to None in the plan and are realized by ``realize_moe_blocks_tp``
# below, which is what keeps the refusal-to-replicate validation honest.
MOE_PLAN_SPECS = frozenset({"packed_colwise", "packed_rowwise", "moe_tp_experts"})


class TPMoeSequenceBoundary:
    """Mixin that brackets a HF MoE block with the TP sequence collectives.

    The MoE block under TP is the dense colwise/rowwise pair with the feature
    shard kept internal: the input arrives as this rank's ``(B, T / tp, D)``
    sequence shard and is all-gathered (backward: reduce-scatter of the input
    gradient); the block then runs on the full token stream with its expert
    weights sharded on the F dim (``shard_experts_for_tp``), producing an
    output that is partial over the TP group; the boundary reduce-scatter sums
    the partials and returns a ``(B, T / tp, D)`` sequence shard (backward:
    all-gather of the output gradient). Weight layout and collectives are the
    same dual pair the dense TP realizers use.

    The router is deliberately untouched: its weight stays replicated, every
    rank computes the identical routing on the gathered stream, and the
    trainer's ``_allreduce_replicated_tp_grads`` sums its gradient (each rank's
    copy earns a different partial through the sharded expert outputs).

    Installed by ``__class__`` swap (like ``GatherSequenceFirst``), so module
    paths, ``state_dict`` keys and later attach points are all untouched.
    """

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        gathered = all_gather_along(hidden_states, -2, self._tp_seq_group)
        original = self.__dict__.get("_tp_moe_original_forward")
        out = (
            original(gathered, *args, **kwargs)
            if original is not None
            else super().forward(gathered, *args, **kwargs)
        )
        # This probe sits between the two collectives; it is only safe because
        # the block's output TYPE is structural -- identical on every rank of
        # the TP group -- so either all ranks raise here or none do. Never let
        # a data-dependent type reach this check.
        if not isinstance(out, torch.Tensor):
            from .. import compatibility

            compatibility.tp_moe_non_tensor_output(self, out)
        return reduce_scatter_along(out, -2, self._tp_seq_group)


def shard_shared_expert_for_tp(
    block: nn.Module, *, tp_size: int, tp_rank: int
) -> set[str]:
    """Feature-shard a shared expert's dense MLP -- no collectives involved.

    Inside the MoE sequence boundary the token stream is already gathered
    (full T, replicated across the TP group), so the shared expert needs no
    input collective: ``gate_proj``/``up_proj`` are sharded on the output
    features (dim 0) and ``down_proj`` on the input features (dim 1), which
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
    from ..expert_parallel.hf_inspection import resolve_shared_expert

    shared_name, shared = resolve_shared_expert(block)
    assert shared is not None  # the caller checked
    names: set[str] = set()
    projections = [
        (name, dim, getattr(shared, name), getattr(shared, name).weight)
        for name, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1))
        if isinstance(getattr(shared, name, None), nn.Linear)
    ]
    # Capture all originals first: gate/up may share a Parameter or the whole
    # Linear module. Never slice an already replaced projection a second time.
    shards: dict[int, nn.Parameter] = {}
    for name, dim, proj, weight in projections:
        if id(weight) not in shards:
            shards[id(weight)] = nn.Parameter(
                shard_weight(weight, dim, tp_size=tp_size, tp_rank=tp_rank),
                requires_grad=weight.requires_grad,
            )
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

    def _shard(w: torch.Tensor, dim_: int) -> torch.Tensor:
        return torch.chunk(w.detach(), tp_size, dim=dim_)[tp_rank].contiguous()

    gate_shard = _shard(gate_up[:, :hidden], 1)
    up_shard = _shard(gate_up[:, hidden:], 1)
    with torch.no_grad():
        experts.gate_up_proj = nn.Parameter(
            torch.cat([gate_shard, up_shard], dim=1).contiguous(),
            requires_grad=gate_up.requires_grad,
        )
        experts.down_proj = nn.Parameter(
            _shard(down, 2), requires_grad=down.requires_grad
        )
    return frozenset({"experts.gate_up_proj", "experts.down_proj"})


@dataclass(frozen=True)
class ShardingConfig:
    """How one projection is sharded on the TP axis.

    Attributes:
        kind: ``"colwise"`` or ``"rowwise"`` -- the direction the weight is cut.
        implementation: the module class that realizes this declaration. Paired
            here rather than looked up from ``kind`` so a caller can swap in a
            different realizer (e.g. a fused-QKV variant) without touching the
            engine.
    """

    kind: ShardKind
    implementation: type[nn.Module]

    def __post_init__(self) -> None:
        if self.kind not in ("colwise", "rowwise"):
            raise ValueError(f"Unknown shard kind {self.kind!r}")


def colwise() -> ShardingConfig:
    """Output features split; activations stay feature-sharded after the GEMM."""
    return ShardingConfig(kind="colwise", implementation=ColumnParallelLinear)


def rowwise() -> ShardingConfig:
    """Input features split; activations reduce-scatter back to a sequence shard."""
    return ShardingConfig(kind="rowwise", implementation=RowParallelLinear)


# -- engine -------------------------------------------------------------------


def model_tp_config(model: nn.Module) -> Mapping[str, str | ShardingConfig]:
    """The TP plan ``model`` declares, or ``{}``.

    Prefers the ``tp_config`` property over the raw ``_tp_config`` attribute.
    ``transformers`` exposes the plan as a property on ``PreTrainedModel`` (its
    instance ``_tp_config`` built from ``config.base_model_tp_config`` plus each
    child module's own plan), and ``HFTransformerModel`` re-parents the HF model
    under ``self.model``, so *its* plan has to be rewritten to survive the extra
    level. A wrapper that exposes only the property -- the shape HF itself
    guarantees -- would otherwise be read as "no plan" and shard nothing, so
    this is the one place the two spellings are reconciled.
    """
    return (
        getattr(model, "tp_config", None)
        or getattr(model, "_tp_config", None)
        or getattr(model, "_tp_plan", None)
        or getattr(model, "tp_plan", None)
        or {}
    )


def resolve_tp_sharding_plan(
    model: nn.Module, *, plan: Mapping[str, str | ShardingConfig] | None = None
) -> dict[str, ShardingConfig | None]:
    """Normalize an explicit or model-declared TP plan for dense projections.

    Patterns retain insertion order. Colwise/rowwise specs become ShardingConfig;
    replicated specs become None so matching leaves their modules intact.
    Replicated projection gradients are reduced by the trainer.

    MoE specs also become None: they name stacked expert parameters rather than
    nn.Linear modules. resolve_tp_targets reads the raw plan to select the
    structural MoE path. EP specs and unsupported replicated-activation layouts
    raise instead of silently applying sequence-parallel realizers.
    """
    if plan is None:
        plan = model_tp_config(model)
    resolved: dict[str, ShardingConfig | None] = {}
    for pattern, spec in plan.items():
        if isinstance(spec, ShardingConfig):
            resolved[pattern] = spec
        elif spec == "colwise":
            resolved[pattern] = colwise()
        elif spec == "rowwise":
            resolved[pattern] = rowwise()
        elif spec in {"replicated_with_grad_allreduce", "colwise_gather_output"}:
            # Nothing for apply_tp to do: the projection stays whole on every
            # rank. The ``_with_grad_allreduce`` half is already implemented --
            # _allreduce_replicated_tp_grads sums exactly these parameters'
            # gradients -- so this entry only has to be understood, not acted
            # on. Recorded as None rather than dropped so match still stops
            # here instead of falling through to a broader later pattern.
            resolved[pattern] = None
        elif spec in MOE_PLAN_SPECS:
            # MoE-under-TP, realized structurally in apply_tp (weight sharding
            # in shard_experts_for_tp, activation collectives in
            # TPMoeSequenceBoundary). None for
            # the same first-match-wins reason as above, and so a pattern that
            # happens to match an nn.Linear is left whole rather than wrongly
            # swapped for a dense realizer.
            resolved[pattern] = None
        else:
            raise ValueError(
                f"Unsupported TP plan entry for {pattern!r}: {spec!r}. This "
                "engine realizes 'colwise' / 'rowwise' (plus the replicated "
                "'replicated_with_grad_allreduce' / 'colwise_gather_output' and "
                "the MoE specs, which it leaves whole), and refuses HF's "
                "replicated-activation styles: 'colwise_rep' / 'rowwise_rep' "
                "keep the activation full on every rank, which is not the "
                "layout the surrounding SP realizers produce or consume. "
                "Rewrite the plan to 'colwise' / 'rowwise' for a model whose "
                "ops tolerate sequence-parallel activations, or run it without "
                "TP."
            )
    return resolved


def match_tp_sharding_spec(
    module_path: str, *, plan: dict[str, ShardingConfig | None]
) -> ShardingConfig | None:
    """Return the first matching TP specification, preserving plan order.

    A matching ``None`` entry leaves the module replicated and stops lookup;
    later sharding patterns cannot override it. No match also returns ``None``.
    """
    for pattern, spec in plan.items():
        if module_name_matches(module_path, pattern=pattern):
            return spec
    return None


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


def enable_symm_mem(group) -> None:
    """Register ``group`` for symmetric-memory collectives.

    ``torch.ops.symm_mem.fused_all_gather_matmul`` (and its reduce-scatter dual)
    only work on a group registered here; PyTorch does not yet do this
    automatically for the TP group. CUDA-only -- call only when
    ``supports_symm_mem`` held for the mesh.
    """
    import warnings

    from torch.distributed._symmetric_memory import enable_symm_mem_for_group

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group.group_name)


def tp_sharded_param_ids(model_parts: Iterable[torch.nn.Module]) -> set[int]:
    """Ids of parameters whose gradients must NOT be summed over the TP group.

    Everything not in this set is treated as TP-replicated by
    ``Trainer._allreduce_replicated_tp_grads`` and summed (each rank's copy
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
            module, ColumnParallelLinear | RowParallelLinear | ColwiseLinearNoGather
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
    plan: Mapping[str, str | ShardingConfig] | None = None,
) -> tuple[list[tuple[str, nn.Linear, ShardingConfig]], list[tuple[str, nn.Module]]]:
    """Validate the TP plan and split its matches into dense and MoE targets.

    Returns ``(targets, moe_blocks)`` where
    ``targets`` are ``(module_path, nn.Linear, ShardingConfig)`` triples and
    ``moe_blocks`` are the HF MoE blocks to realize in place. A plan that
    resolves to nothing, or that matches no module, used to leave the model
    fully replicated while the run reported a healthy TP setup -- both are
    loud errors here, raised before any process-group access.
    """
    configured_plan: dict[str, str] = {}
    for pattern in cfg.tp_config.tp_colwise_modules:
        configured_plan[pattern] = "colwise"
    for pattern in cfg.tp_config.tp_rowwise_modules:
        if pattern in configured_plan:
            raise ValueError(f"TP pattern {pattern!r} has conflicting styles")
        configured_plan[pattern] = "rowwise"
    if cfg.tp_config.tp_sequence_modules:
        raise ValueError(
            "tp_sequence_modules is not valid for llmtuner's fused sequence "
            "parallel engine; normalization modules already consume the "
            "sequence shard without a separate transformation"
        )
    raw_plan = model_tp_config(model) if plan is None else plan
    if configured_plan:
        # Pipeline assembly validates against the complete model before split:
        # an override may intentionally select only another stage's modules.
        if cfg.pp_size == 1:
            resolve_module_patterns(
                [name for name, _ in model.named_modules()],
                patterns=list(configured_plan),
                plan_name="TP override",
            )
        # Matching uses the first pattern; explicit overrides must come first.
        raw_plan = {
            **configured_plan,
            **{
                key: value
                for key, value in raw_plan.items()
                if key not in configured_plan
            },
        }

    sharding_plan = resolve_tp_sharding_plan(model, plan=raw_plan)
    if not sharding_plan:
        raise ValueError(
            f"apply_tp with tp={cfg.tp_size}: {type(model).__name__} provides no TP "
            "plan (neither a `tp_config`/`_tp_config` declaration nor an explicit "
            "`plan` argument). Refusing to run TP as a silently replicated "
            "model; shard the projections declaratively or set tp=1."
        )

    targets: list[tuple[str, nn.Linear, ShardingConfig]] = []
    for module_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            spec = match_tp_sharding_spec(module_path, plan=sharding_plan)
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
        isinstance(spec, str) and spec in MOE_PLAN_SPECS for spec in raw_plan.values()
    )
    # tp x ep: TP shards only the dense parts and EP owns the routed experts
    # (the upstream alignment). The HF blocks are left whole here and swapped
    # for the native MoE stack by apply_ep, which runs next; the swapped block
    # consumes and produces the T/tp sequence shard directly (all-to-all
    # dispatch), so no sequence-boundary collectives are installed either.
    moe_deferred_to_ep = plan_declares_moe and cfg.ep_size > 1
    moe_blocks: list[tuple[str, nn.Module]] = []
    bracketed_paths: list[str] = []
    from ..expert_parallel.hf_inspection import is_hf_moe_block

    if plan_declares_moe and not moe_deferred_to_ep:
        for module_path, module in model.named_modules():
            if getattr(module, "_tp_moe_boundary", False):
                # Already bracketed by an earlier apply_tp pass (the engine is
                # idempotent per module): still counts as realized MoE TP.
                bracketed_paths.append(module_path)
                continue
            # Blocks the probe refuses (GPT-OSS's transposed, bias-bearing
            # experts) raise out of it here, before any weight is touched.
            if is_hf_moe_block(module):
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
        for path, module in model.named_modules()
        if path not in moe_block_paths
        and is_hf_moe_block(module)
        and not getattr(module, "_tp_moe_boundary", False)
    )
    if moe_block_paths or bracketed_paths:
        moe_prefixes = tuple(
            f"{p}." if p else "" for p in [*bracketed_paths, *moe_block_paths]
        )
        targets = [t for t in targets if not t[0].startswith(moe_prefixes)]
    nothing_sharded = not targets and not moe_blocks and not bracketed_paths
    if nothing_sharded and not moe_deferred_to_ep:
        raise ValueError(
            f"apply_tp with tp={cfg.tp_size}: the plan patterns "
            f"{sorted(sharding_plan)} matched no nn.Linear on "
            f"{type(model).__name__}. The patterns are spelled relative to the "
            "module tree being parallelized; check the prefix (e.g. a wrapper's "
            "`model.` prefix) rather than training a replicated model by "
            "mistake."
        )
    return targets, moe_blocks


def realize_moe_blocks_tp(
    moe_blocks: list[tuple[str, nn.Module]], *, tp_size: int, tp_rank: int, group
) -> None:
    """Shard each HF MoE block's experts and shared expert in place, and give
    the block the TP sequence-boundary class.

    MoE-block internals never take the dense realizer path: the routed experts
    are stacked parameters (no nn.Linear to match), and the shared expert is
    sharded featurewise by ``shard_shared_expert_for_tp``, collective-free
    inside the sequence boundary.
    """
    from ..expert_parallel.hf_inspection import resolve_shared_expert

    for module_path, block in moe_blocks:
        _, shared = resolve_shared_expert(block)
        expert_tp_hidden_size(block, tp_size=tp_size)
        if shared is not None:
            # The gate/up/down layout shards featurewise with no collectives
            # (the boundary already gathered the sequence); anything else --
            # e.g. Qwen2Moe's multiplicative shared_expert_gate -- keeps the
            # refusal.
            if not all(
                isinstance(getattr(shared, n, None), nn.Linear)
                for n in ("gate_proj", "up_proj", "down_proj")
            ):
                compatibility.shared_expert_tp(module_path, block)
            gate_shape = tuple(shared.gate_proj.weight.shape)
            up_shape = tuple(shared.up_proj.weight.shape)
            down_shape = tuple(shared.down_proj.weight.shape)
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
            for name, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                projection = getattr(shared, name)
                if projection.bias is not None:
                    raise ValueError(
                        f"TP shared expert {module_path}.{name} has an unsupported bias"
                    )
                if projection.weight.shape[dim] % tp_size:
                    raise ValueError(
                        f"TP shared expert {module_path}.{name} weight dim {dim} "
                        f"is not divisible by tp_size={tp_size}"
                    )

    for _, block in moe_blocks:
        _, shared = resolve_shared_expert(block)
        shared_names: set[str] = set()
        if shared is not None:
            shared_names = shard_shared_expert_for_tp(
                block, tp_size=tp_size, tp_rank=tp_rank
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
        block._tp_moe_boundary = True
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


def apply_tp(
    model: nn.Module,
    mesh: DeviceMesh | None,
    cfg: ParallelConfig,
    *,
    plan: Mapping[str, str | ShardingConfig] | None = None,
) -> nn.Module:
    """Tensor-parallelize ``model`` in place. No-op when ``tp == 1``.

    ``plan`` defaults to the model's HF ``tp_config`` (HF ships one for Qwen3,
    Llama, ...). Pass an explicit ``{path_pattern: ShardingConfig}`` to override
    it -- e.g. to leave a projection replicated or to use a different realizer.
    """
    if cfg.tp_size <= 1:
        return model
    if any(
        getattr(getattr(module, "config", None), "attention_backend", "auto")
        == "eager_flex"
        for module in model.modules()
    ):
        compatibility.eager_flex_parallel("TP")
    if any(getattr(module, "_llmtuner_fused_qkv", False) for module in model.modules()):
        compatibility.fused_qkv_tp()
    if mesh is None:
        raise ValueError(f"tp={cfg.tp_size} requires a device mesh with a 'tp' axis")

    # Resolve targets once before accessing process groups or changing weights.
    targets, moe_blocks = resolve_tp_targets(model, cfg=cfg, plan=plan)

    # A head count that does not divide ``tp`` has no valid local head split,
    # and no downstream guard catches it: the projection-level check in
    # ``shard_weight`` only sees the feature dim, which 8 KV heads at
    # head_dim=128 (1024 features) satisfy at tp=16. Upstream rejects this at
    # config parse (upstream torchtitan ``config/validation.py``'s
    # ``head_shard_degree``); here the
    # model's own config is the first place the counts exist.
    attention_paths = [
        path
        for path, _ in model.named_modules()
        if path.rsplit(".", 1)[-1] == "self_attn"
    ]
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
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(id(parameter), []).append(name)
    # MoE slicing also replaces Parameters. A shared expert module would be
    # sliced twice, while individually tied weights would silently lose their
    # identity. Permit only the same shared MLP's gate/up tie: both use the
    # identical column slice, and the shared realizer preserves that identity.
    # Reject all other aliases before any constructor or MoE mutation runs.
    for module_path, block in moe_blocks:
        from ..expert_parallel.hf_inspection import resolve_shared_expert

        shared_name, shared = resolve_shared_expert(block)
        prefix = f"{module_path}." if module_path else ""
        tied_shared_paths = {
            f"{prefix}{shared_name}.{name}.weight" for name in ("gate_proj", "up_proj")
        }
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
        for path, module in model.named_modules()
        if path.rsplit(".", 1)[-1] == "self_attn"
        and not hasattr(module, "_tp_seq_group")
        and not any(target.startswith(f"{path}.") for target, _, _ in targets)
    ]

    tp_size = mesh["tp"].size()
    if tp_size != cfg.tp_size:
        raise ValueError(
            f"TP mesh size {tp_size} does not match configured tp={cfg.tp_size}"
        )
    group = mesh["tp"].get_group()
    tp_rank = mesh["tp"].get_local_rank()
    use_symm_mem = supports_symm_mem(mesh["tp"])
    if use_symm_mem:
        enable_symm_mem(group)

    # Resolve optional imports before tensor graphs are compiled.
    functional_collectives()

    # Prepare every dense replacement before changing the model. A failing
    # custom constructor must not leave earlier projections or MoE blocks sharded.
    # Constructors are responsible for avoiding side effects on their inputs.
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    # Deepest paths first, so replacing a module never hides an inner target.
    attention_parents: dict[str, nn.Module] = {}
    for module_path, inner, spec in sorted(targets, key=lambda t: -t[0].count(".")):
        parent_path, _, attr = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        implementation = spec.implementation
        if spec.kind == "colwise" and looks_like_attention(parent):
            # HF attention reshapes q/k/v by the input's shape, so the fused
            # in-GEMM sequence gather would silently mis-shape them. Take the
            # same gather at the module boundary instead (below) and give the
            # projection a plain feature-sharded GEMM. Only the default
            # realizer is swapped out; an explicitly provided one is the
            # caller's responsibility.
            if implementation is ColumnParallelLinear:
                implementation = ColwiseLinearNoGather
            attention_parents[parent_path] = parent

        wrapped = implementation(
            inner.weight,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group=group,
            use_symm_mem=use_symm_mem,
        )
        wrapped.train(inner.training)
        replacements.append((parent, attr, wrapped))

    realize_moe_blocks_tp(moe_blocks, tp_size=tp_size, tp_rank=tp_rank, group=group)
    for parent, attr, wrapped in replacements:
        setattr(parent, attr, wrapped)

    for parent in attention_parents.values():
        if getattr(parent, "_tp_seq_group", None) is not None:
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
    for path, block in moe_blocks:
        prefix = f"{path}." if path else ""
        layouts[f"{prefix}experts.gate_up_proj"] = (1, True, group)
        layouts[f"{prefix}experts.down_proj"] = (2, False, group)
        for name in block.tp_sharded_param_names:
            if not name.startswith("experts."):
                layouts[f"{prefix}{name}"] = (
                    1 if ".down_proj." in name else 0,
                    False,
                    group,
                )

    return model
