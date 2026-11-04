"""Plan and install native MoE replacements before FSDP wrapping.

Layout detection and model-family restrictions belong to ``hf_inspection.py``;
weight conversion belongs to ``block_conversion.py`` or ``kimi_conversion.py``.
This module coordinates those operations after weights are loaded.

All replacements, shared-parameter bindings and TP checkpoint registrations
are planned before any decoder slot is replaced. Unsupported aliases or layouts
therefore fail without leaving a mixture of HF and native blocks. Failed planning
also restores auxiliary-loss group counts.

Installation preserves repeated block references and marks decoder ownership
with ``moe_enabled`` and a non-registered ``moe`` alias. FSDP uses those markers
to shard experts over the sparse mesh rather than the dense DP mesh. The alias
must remain outside the module registry to avoid duplicate state-dict keys.

``on_replacement`` receives the planned module and parameter mapping before
installation; callers use it to migrate metadata tied to the original objects.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from llmtuner.config.parallel_config import (
    ExpertParallelConfig,
    validate_router_aux_loss_coefficient,
)
from llmtuner.errors import UnsupportedCombinationError

from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.layer import MoE
from ...models.kimi.moe import KimiSparseMoE
from ...utils.logging import get_logger
from .block_conversion import convert_block
from .hf_inspection import (
    is_hf_moe_block,
    moe_block_of,
    per_expert_projection_names,
    returns_router_logits,
    router_of,
)
from .kimi_conversion import convert_kimi_block

logger = get_logger(__name__)

__all__ = ["swap_hf_moe_blocks"]

MoEReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]


def _bind_shared_projection(
    owner: nn.Module,
    names: tuple[str, ...],
    sources: tuple[torch.Tensor, ...],
    *,
    converted_parameters: dict[int, tuple[nn.Parameter, ...]],
    stacked_projection_sources: dict[int, tuple[int, ...]],
    external_parameters: dict[int, list[str]],
) -> None:
    """Bind one new projection while preserving source ownership and row order.

    Only replacement modules and local planning indexes are changed here;
    installation into the live model remains the caller's final phase.
    """
    previous = [converted_parameters.get(id(s)) for s in sources]
    if all(entry is None for entry in previous):
        targets = tuple(getattr(owner, name) for name in names)
        # Unchanged layouts can keep the original Parameter, also
        # preserving consumers outside the converted MoE subtree.
        # Split gate/up, stacked per-expert and EP-sharded
        # projections have new layouts.
        if (
            len(sources) == 1
            and len(targets) == 1
            and isinstance(sources[0], nn.Parameter)
            and type(sources[0]) is type(targets[0])
            and sources[0].shape == targets[0].shape
            and sources[0].dtype == targets[0].dtype
            and sources[0].device == targets[0].device
        ):
            setattr(owner, names[0], sources[0])
            targets = (sources[0],)
        for source in sources:
            external = external_parameters.get(id(source), [])
            if external and not (len(targets) == 1 and targets[0] is source):
                raise UnsupportedCombinationError(
                    "MoE conversion changes the layout of a "
                    "parameter also registered outside converted "
                    f"blocks at {external}; split, stacked or "
                    "EP-sharded expert aliases require an adapted "
                    "external consumer. Register aliases to the "
                    "converted expert parameters after EP conversion."
                )
            converted_parameters[id(source)] = targets
        if len(sources) > 1:
            stacked_projection_sources[id(targets[0])] = tuple(
                id(source) for source in sources
            )
        return
    shared = previous[0]
    consistent = shared is not None and len(shared) == len(names)
    for entry in previous:
        if (
            entry is None
            or len(entry) != len(shared or ())
            or any(a is not b for a, b in zip(entry, shared or (), strict=True))
        ):
            consistent = False
    if not consistent:
        raise UnsupportedCombinationError(
            "MoE conversion cannot preserve a partially shared expert "
            "projection: stacking binds the whole projection to one "
            "grouped parameter, so a shared projection must be shared "
            "whole, with the same owner, by every converted block."
        )
    if len(sources) > 1 and stacked_projection_sources.get(id(shared[0])) != tuple(
        id(source) for source in sources
    ):
        raise UnsupportedCombinationError(
            "MoE conversion cannot preserve a shared projection with "
            "different expert order; stacked rows must refer to the "
            "same source parameters in the same order."
        )
    for name, parameter in zip(names, shared, strict=True):
        setattr(owner, name, parameter)


def _plan_tp_checkpoint_layouts(
    module_paths: list[tuple[str, nn.Module]],
    candidates: list[tuple[nn.Module, str, nn.Module, bool]],
    converted_blocks: dict[int, MoE],
    replaced_roots: list[str],
) -> list[tuple[nn.Module, dict]]:
    """Plan TP registration renames without modifying any live owner."""
    # TP registers physical HF slices before this swap. Gate/up becomes two
    # independent parameters, so each checkpoint slice now has its own name
    # and no longer uses packed-half gathering. Preserve the registration's
    # owner, which may be the model root rather than the replaced block.
    converted_layouts = []
    for owner_path, owner in module_paths:
        layouts = owner.__dict__.get("_tp_checkpoint_layouts")
        if layouts is None:
            continue
        expert_owner = next(
            (
                converted_blocks[id(block)].routed_experts.inner_experts
                for _, _, block, _ in candidates
                if not isinstance(block, KimiSparseMoE) and owner is block.experts
            ),
            None,
        )
        if expert_owner is not None:
            updated = {}
            for name, layout in layouts.items():
                names = {
                    "gate_up_proj": ("w1_EFD", "w3_EFD"),
                    "down_proj": ("w2_EDF",),
                }.get(name, (name,))
                for target_name in names:
                    updated[target_name] = (
                        (layout[0], False, layout[2])
                        if name in ("gate_up_proj", "down_proj")
                        else layout
                    )
            converted_layouts.append((expert_owner, updated))
            continue
        updated = {}
        for name, layout in layouts.items():
            full_name = f"{owner_path}.{name}" if owner_path else name
            replacement = None
            for root in replaced_roots:
                if full_name == f"{root}.experts.gate_up_proj":
                    replacement = ("w1_EFD", "w3_EFD")
                    break
                if full_name == f"{root}.experts.down_proj":
                    replacement = ("w2_EDF",)
                    break
            if replacement is None:
                for root in replaced_roots:
                    prefix = f"{root}.shared_expert."
                    if full_name.startswith(prefix) and "shared_expert." in name:
                        # The reused shared module keeps its own relative
                        # registrations. Only an ancestor's path needs renaming.
                        parent, suffix = name.rsplit("shared_expert.", 1)
                        name = f"{parent}shared_experts.{suffix}"
                        break
                updated[name] = layout
                continue
            prefix = name.rsplit("experts.", 1)[0]
            for parameter_name in replacement:
                updated[f"{prefix}routed_experts.inner_experts.{parameter_name}"] = (
                    layout[0],
                    False,
                    layout[2],
                )
        converted_layouts.append((converted_blocks.get(id(owner), owner), updated))
    return converted_layouts


def swap_hf_moe_blocks(
    model: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None = None,
    router_aux_loss_coef: float | None = None,
    quantile_balancing: bool = False,
    fixed_router: bool = False,
    token_dispatcher: str = "alltoall",
    torchao_pad_multiple: int = 16,
    tp_enabled: bool = False,
    on_replacement: MoEReplacementObserver | None = None,
) -> int:
    """Replace every HF MoE block in ``model`` with llmtuner's MoE, in place.

    Args:
        model: a ``HFTransformerModel`` (anything exposing ``.layers``).
        ep_group: the EP process group. ``None`` (or a size-1 group is not
            special-cased -- pass ``None``) keeps all experts local and routes
            them with the reordering-only dispatcher; a multi-rank group
            shards the experts across it and routes with all-to-alls.
        router_aux_loss_coef: coefficient for the per-forward load-balance
            loss. ``None`` takes the HF config's ``router_aux_loss_coef``
            (Qwen3Moe has one; DeepSeek-V3's config has no such field and gets
            no loss). A float here overrides the config for every MoE layer.
        quantile_balancing: replace the sign-based load-balancing bias with
            quantile-balanced routing (``QuantileBalancedTopKRouter``): the
            bias is then re-solved from a required-bias histogram once per
            optimizer step instead of nudged by the sign rule, and
            ``load_balance_coeff`` is forced off. Requires sigmoid router
            scores and no group-limited routing.
        fixed_router: use FSDPTurbo fixed assignments and uniform debug weights.
        token_dispatcher: EP dispatch backend (``"alltoall"`` default,
            ``"torchao"`` optional-import adapter, ``"mc2"`` Ascend fusion,
            ``"domino"`` two-slice pipeline, ``"deepep"``/``"hybridep"``
            registered gaps refused here). See
            ``ParallelConfig.ep_token_dispatcher``.
        torchao_pad_multiple: padding multiple for the ``"torchao"`` backend.
        tp_enabled: the model is also tensor-parallelized (tp x ep). Used only
            for fail-fast validation of unverified combinations (currently a
            shared expert): the swap itself is layout-identical either way --
            the swapped block consumes and produces the T/tp sequence shard
            directly, which is the layout upstream's ep+sp MoE uses.
        on_replacement: receive module and fused-parameter mappings after all
            replacements are planned, before any block is installed. Used by
            assembly to preserve programmatic references across layout changes.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        UnsupportedCombinationError: if no layer carries a recognizable HF MoE
            block. EP on a dense model is a config mistake; swapping nothing
            would run it replicated.
        NotImplementedError: if a MoE block uses a layout this swap does not
            implement -- GPT-OSS's transposed, bias-bearing experts (see
            ``fused_experts_of``) or DeepSeek-V2's ``group_limited_greedy``
            routing (see ``read_expert_groups``).
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise UnsupportedCombinationError(
            f"swap_hf_moe_blocks expects a model with .layers; got "
            f"{type(model).__name__}."
        )
    # Backend gating, before any probing: ParallelConfig.__post_init__ is the
    # primary gate; this is the defensive copy for callers that reach the swap
    # directly.
    ExpertParallelConfig(
        router_aux_loss_coef=router_aux_loss_coef,
        moe_quantile_balancing=quantile_balancing,
        fixed_router=fixed_router,
        ep_token_dispatcher=token_dispatcher,
        ep_torchao_pad_multiple=torchao_pad_multiple,
    )

    hf_config = getattr(getattr(model, "model", None), "config", None)
    aux_loss_coeff = (
        router_aux_loss_coef
        if router_aux_loss_coef is not None
        else getattr(hf_config, "router_aux_loss_coef", None)
    )
    validate_router_aux_loss_coefficient(aux_loss_coeff)
    # The load-balance coefficient belongs to a *native* llmtuner run; an HF
    # checkpoint has no equivalent field. It is therefore derived from the
    # probed bias: a block that carries ``e_score_correction_bias`` is one the
    # model was trained with the auxiliary-loss-free scheme on (DeepSeek-V3,
    # GLM4), and gets the llmtuner default. Everything else stays off, so a
    # frozen zero bias never lands in the checkpoint.
    default_coeff = getattr(hf_config, "load_balance_coeff", 1e-3)

    # Probe every layer before constructing replacements: a later unsupported
    # expert layout must not leave earlier layers converted.
    candidates: list[tuple[nn.Module, str, nn.Module, bool]] = []
    for layer in layers:
        attr, block = moe_block_of(layer)
        # Dense layers of a mixed model (Qwen3Moe's decoder_sparse_step, and the
        # first_k_dense_replace of the DeepSeek/GLM4 families) are left alone.
        if block is None or not (
            isinstance(block, KimiSparseMoE) or is_hf_moe_block(block)
        ):
            continue
        wants_router_logits = returns_router_logits(block)
        # The replacement has no router logits to hand back (llmtuner's router
        # keeps its scores internal), so a config that asks HF to consume them
        # would get ``None`` where it expects a tensor. Refuse rather than
        # answer with a value that is wrong in a way no loss curve shows.
        if wants_router_logits and getattr(hf_config, "output_router_logits", False):
            raise UnsupportedCombinationError(
                f"{type(block).__name__} returns router logits and its config "
                "enables output_router_logits, but the swapped-in llmtuner MoE "
                "keeps its router scores internal, so it could only hand back "
                "None. Turn output_router_logits off -- llmtuner runs its own "
                "load-balancing loss -- before swapping."
            )
        candidates.append((layer, attr, block, wants_router_logits))

    if not candidates:
        raise UnsupportedCombinationError(
            f"no HF MoE block found on {type(model).__name__} "
            f"({type(getattr(model, 'model', model)).__name__}): no layer's "
            "``mlp`` has the router-gate + fused gate_up_proj/down_proj shape "
            "this swap recognizes. Add the model family's layout to the probe "
            "in parallel/expert_parallel/hf_inspection.py."
        )

    module_paths = list(model.named_modules(remove_duplicate=False))
    # Index owners once; retain every registered path for aliased decoder layers.
    candidate_attrs: dict[int, list[str]] = {}
    for layer, attr, _, _ in candidates:
        candidate_attrs.setdefault(id(layer), []).append(attr)
    replaced_roots = [
        f"{path}.{attr}" if path else attr
        for path, module in module_paths
        for attr in candidate_attrs.get(id(module), ())
    ]
    external_parameters: dict[int, list[str]] = {}
    external_buffers: dict[int, list[str]] = {}
    for path, module in module_paths:
        if any(path == root or path.startswith(f"{root}.") for root in replaced_roots):
            continue
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            fqn = f"{path}.{name}" if path else name
            external_parameters.setdefault(id(parameter), []).append(fqn)
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            fqn = f"{path}.{name}" if path else name
            external_buffers.setdefault(id(buffer), []).append(fqn)

    replacements: list[tuple[nn.Module, str, MoE]] = []
    converted_blocks: dict[int, MoE] = {}
    converted_parameters: dict[int, tuple[nn.Parameter, ...]] = {}
    stacked_projection_sources: dict[int, tuple[int, ...]] = {}
    original_parameters = (
        {id(parameter): parameter for parameter in model.parameters()}
        if on_replacement is not None
        else {}
    )
    converted_biases: dict[int, torch.Tensor] = {}
    aux_group_counts = dict(AuxLoss._group_counts)
    # Stacking a projection cannot preserve individual expert aliases. Reject
    # them before constructing any replacement, including cross-block aliases.
    kimi_expert_ids = {
        id(parameter)
        for _, _, block, _ in candidates
        if isinstance(block, KimiSparseMoE)
        for parameter in block.experts.parameters()
    }
    owners: dict[int, tuple[int, str]] = {}
    checked_kimi_blocks: set[int] = set()
    for _, _, block, _ in candidates:
        if not isinstance(block, KimiSparseMoE) or id(block) in checked_kimi_blocks:
            continue
        checked_kimi_blocks.add(id(block))
        for name, parameter in block.experts.named_parameters(remove_duplicate=False):
            owner = (id(block), name)
            previous = owners.setdefault(id(parameter), owner)
            if previous != owner:
                raise UnsupportedCombinationError(
                    "Kimi EP cannot stack a shared individual expert parameter "
                    f"at experts.{name}; expert aliases require a layout adapter"
                )
    owners.clear()
    for path, module in module_paths:
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) not in kimi_expert_ids:
                continue
            owner = (id(module), name)
            previous = owners.setdefault(id(parameter), owner)
            if previous != owner or id(parameter) in external_parameters:
                raise UnsupportedCombinationError(
                    f"Kimi EP cannot stack a shared individual expert parameter "
                    f"at {path}.{name}; expert aliases require a layout adapter"
                )
    try:
        for layer, attr, block, wants_router_logits in candidates:
            # Repeated layer slots may reference one physical HF block. Preserve
            # that sharing rather than cloning trainable weights and statistics.
            if id(block) in converted_blocks:
                replacements.append((layer, attr, converted_blocks[id(block)]))
                continue
            if isinstance(block, KimiSparseMoE):
                moe = convert_kimi_block(
                    block,
                    ep_group=ep_group,
                    aux_loss_coeff=aux_loss_coeff,
                    quantile_balancing=quantile_balancing,
                    token_dispatcher=token_dispatcher,
                    tp_enabled=tp_enabled,
                )
                moe.router.fixed_router = fixed_router
                if on_replacement is not None:
                    grouped = moe.routed_experts.inner_experts
                    for expert in block.experts:
                        for source_name, target_name in (
                            ("w1", "w1_EFD"),
                            ("w2", "w2_EDF"),
                            ("w3", "w3_EFD"),
                        ):
                            source = getattr(expert, source_name).weight
                            converted_parameters[id(source)] = (
                                getattr(grouped, target_name),
                            )
                converted_blocks[id(block)] = moe
                replacements.append((layer, attr, moe))
                continue
            router_gate = router_of(block)
            load_balance_coeff = (
                float(default_coeff)
                if getattr(router_gate, "e_score_correction_bias", None) is not None
                else None
            )
            moe = convert_block(
                block,
                ep_group=ep_group,
                aux_loss_coeff=aux_loss_coeff,
                load_balance_coeff=load_balance_coeff,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
                torchao_pad_multiple=torchao_pad_multiple,
                tp_enabled=tp_enabled,
            )
            # Distinct blocks may tie only a projection. The fused gate/up
            # source maps to two parameters, which must be reused as a pair; a
            # per-expert (4.x) block ties each of its E projection weights to
            # the one stacked parameter they were converted into. Bind after
            # dtype/device conversion, before installing any block.
            grouped = moe.routed_experts.inner_experts
            projection_names = per_expert_projection_names(block.experts)
            if projection_names is None:
                projections: list[
                    tuple[nn.Module, tuple[str, ...], tuple[torch.Tensor, ...]]
                ] = [
                    (moe.router.gate, ("weight",), (router_gate.weight,)),
                    (grouped, ("w1_EFD", "w3_EFD"), (block.experts.gate_up_proj,)),
                    (grouped, ("w2_EDF",), (block.experts.down_proj,)),
                ]
            else:
                gate_name, up_name, down_name = projection_names
                members = list(block.experts)
                projections = [
                    (moe.router.gate, ("weight",), (router_gate.weight,)),
                    (
                        grouped,
                        ("w1_EFD",),
                        tuple(getattr(e, gate_name).weight for e in members),
                    ),
                    (
                        grouped,
                        ("w3_EFD",),
                        tuple(getattr(e, up_name).weight for e in members),
                    ),
                    (
                        grouped,
                        ("w2_EDF",),
                        tuple(getattr(e, down_name).weight for e in members),
                    ),
                ]
            for owner, names, sources in projections:
                _bind_shared_projection(
                    owner,
                    names,
                    sources,
                    converted_parameters=converted_parameters,
                    stacked_projection_sources=stacked_projection_sources,
                    external_parameters=external_parameters,
                )
            # Shared bias consumers contribute counts or histograms to one
            # policy-specific update in bias_update.py.
            source_bias = getattr(router_gate, "e_score_correction_bias", None)
            if isinstance(source_bias, torch.Tensor) and moe.expert_bias_E is not None:
                if id(source_bias) not in converted_biases:
                    target_bias = moe.expert_bias_E
                    if (
                        source_bias.dtype == target_bias.dtype
                        and source_bias.device == target_bias.device
                    ):
                        # Bias updates must remain visible to registered consumers
                        # outside the replaced subtree, just like shared weights.
                        moe.expert_bias_E = source_bias
                    elif id(source_bias) in external_buffers:
                        raise UnsupportedCombinationError(
                            "MoE conversion cannot preserve an external router bias "
                            "alias while changing its dtype or device: "
                            f"{external_buffers[id(source_bias)]}"
                        )
                    converted_biases[id(source_bias)] = moe.expert_bias_E
                else:
                    moe.expert_bias_E = converted_biases[id(source_bias)]
            # Fixed assignment and constant scores belong to the router,
            # shared by every dispatcher.
            moe.router.fixed_router = fixed_router
            # Keep the host layer's unpacking contract intact: the block it was
            # written against returns a tuple on 4.x Mixtral/OLMoE/GraniteMoe.
            moe.returns_router_logits = wants_router_logits
            converted_blocks[id(block)] = moe
            replacements.append((layer, attr, moe))
        converted_layouts = _plan_tp_checkpoint_layouts(
            module_paths, candidates, converted_blocks, replaced_roots
        )

        if on_replacement is not None:
            module_targets = {}
            for _, _, block, _ in candidates:
                moe = converted_blocks[id(block)]
                module_targets[block] = moe
                module_targets[block.experts] = moe.routed_experts.inner_experts
                module_targets[router_of(block)] = moe.router.gate
            on_replacement(
                module_targets,
                {
                    original_parameters[source]: targets
                    for source, targets in converted_parameters.items()
                },
            )

    except BaseException:
        # Failed replacement planning must not inflate auxiliary-loss metric divisors.
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(aux_group_counts)
        raise

    # Install only after every replacement was constructed successfully.
    # A later conversion error must not leave a mixture of HF and native blocks.
    for layer, attr, moe in replacements:
        # Replace in the slot the block was actually found in. Writing to a
        # different attribute would leave the original block in place and route
        # around the swap entirely.
        setattr(layer, attr, moe)
        # FSDP's MoE branch (fully_shard_parallel/plan.py) keys off these two
        # attributes, as does upstream's swap (moe_replacement.py):
        # ``moe_enabled`` marks the block as sparse, and ``moe`` is where the
        # branch reads the expert weights from. Without them the experts are
        # sharded as dense parameters over the dense DP mesh, mixing ranks of
        # different EP coordinates into one FSDP group. ``object.__setattr__``
        # keeps ``moe`` out of the module registry: it is the same module as
        # the block attribute, and registering it would double every expert
        # weight in the state_dict.
        layer.moe_enabled = True
        object.__setattr__(layer, "moe", moe)

    for owner, layouts in converted_layouts:
        owner._tp_checkpoint_layouts = layouts

    logger.info("Swapped %d HF MoE blocks for llmtuner MoE blocks", len(replacements))
    return len(replacements)
