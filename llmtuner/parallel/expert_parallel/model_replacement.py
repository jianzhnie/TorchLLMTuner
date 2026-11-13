"""Plan and install native MoE replacements before FSDP wrapping.

Explicit models supply decoder and MoE slot declarations. Only undeclared
legacy callers use ``hf_inspection.py``; weight conversion belongs to
``block_conversion.py`` or ``kimi_conversion.py``. This module resolves those
bindings before installing replacements after weights are loaded.

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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from llmtuner.config.parallel_config import (
    ExpertParallelConfig,
    validate_expert_dispatcher,
    validate_router_aux_loss_coefficient,
)
from llmtuner.errors import UnsupportedCombinationError

from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.layer import MoE, declared_moe_block
from ...models.contracts import DecoderStructure, MoEBlockContract
from ...models.kimi.kimi_k3.moe import KimiSparseMoE
from ...utils.logging import get_logger
from ..plan import ParallelPlan, PlanTarget
from ..plan import ReplacementObserver as MoEReplacementObserver
from .block_conversion import (
    MoERoutingBinding,
    PackedExpertBinding,
    convert_block,
    resolve_moe_routing,
    resolve_packed_experts,
    validate_moe_conversion,
)
from .kimi_conversion import (
    convert_kimi_block,
    validate_kimi_conversion,
)

logger = get_logger(__name__)

__all__ = ["resolve_ep_plan", "swap_moe_blocks", "swap_hf_moe_blocks"]


def declared_router_output(block: nn.Module) -> bool:
    """Resolve explicit output semantics without coercing or inspecting forward.

    A retained per-block flag must agree with the model-owned MoE contract.
    Contract-only declarations do not need to duplicate the flag.
    """
    contract = getattr(block, "moe_contract", None)
    expected = (
        contract.router_logits_in_output
        if isinstance(contract, MoEBlockContract)
        else False
    )
    declared = getattr(block, "router_logits_in_output", expected)
    if type(declared) is not bool:
        raise ValueError("MoE router_logits_in_output must be boolean")
    if isinstance(contract, MoEBlockContract) and declared != expected:
        raise ValueError("MoE router output flag conflicts with its declared contract")
    return declared


def validate_router_output_conversion(
    block: nn.Module, wants_router_logits: bool, config: Any
) -> None:
    """Preserve the output restriction even when config changes after parsing."""
    if wants_router_logits and getattr(config, "output_router_logits", False):
        raise UnsupportedCombinationError(
            f"{type(block).__name__} returns router logits and its config "
            "enables output_router_logits, but the swapped-in llmtuner MoE "
            "keeps its router scores internal, so it could only hand back "
            "None. Turn output_router_logits off -- llmtuner runs its own "
            "load-balancing loss -- before swapping."
        )


def _legacy_inspection():
    """Load external-layout probes only for undeclared legacy models."""
    from . import hf_inspection

    return hf_inspection


def bind_shared_projection(
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
    consistent = (
        shared is not None
        and len(shared) == len(names)
        and all(
            entry is not None
            and len(entry) == len(shared)
            and all(a is b for a, b in zip(entry, shared, strict=True))
            for entry in previous
        )
    )
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


def plan_tp_checkpoint_layouts(
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


def validate_kimi_expert_aliases(
    candidates: list[tuple[nn.Module, str, nn.Module, bool]],
    module_paths: list[tuple[str, nn.Module]],
    external_parameters: dict[int, list[str]],
) -> None:
    """Reject expert aliases that stacking cannot preserve, before conversion.

    Repeated references to one block are allowed. Sharing individual expert
    parameters within or across blocks, or with external consumers, is not.
    This preflight reads the original model without allocating replacements.
    """
    # Stacking a projection cannot preserve individual expert aliases. Reject
    # them before constructing any replacement, including cross-block aliases.
    kimi_expert_ids = {
        id(parameter)
        for _, _, block, _ in candidates
        if isinstance(block, KimiSparseMoE)
        for parameter in block.experts.parameters()
    }
    if not kimi_expert_ids:
        return
    # Different Parameter objects (or a buffer view) can share storage without
    # sharing identity. Stacking would silently sever that ownership too.
    expert_storage: dict[int, int] = {}
    for _, module in module_paths:
        for parameter in module.parameters(recurse=False):
            if (
                id(parameter) in kimi_expert_ids
                and type(parameter) is nn.Parameter
                and parameter.layout == torch.strided
            ):
                storage = parameter.untyped_storage()._cdata
                previous = expert_storage.setdefault(storage, id(parameter))
                if previous != id(parameter):
                    raise UnsupportedCombinationError(
                        "Kimi EP cannot stack expert parameters sharing storage"
                    )
    for path, module in module_paths:
        buffers = tuple(module.named_buffers(recurse=False))
        buffer_ids = {id(value) for _, value in buffers}
        tensors = tuple(module.named_parameters(recurse=False)) + buffers
        for name, tensor in tensors:
            if type(tensor) not in (torch.Tensor, nn.Parameter):
                continue
            if tensor.layout != torch.strided:
                continue
            owner = expert_storage.get(tensor.untyped_storage()._cdata)
            if owner is not None and (id(tensor) != owner or id(tensor) in buffer_ids):
                raise UnsupportedCombinationError(
                    "Kimi EP cannot preserve expert storage alias "
                    f"at {path}.{name}; an explicit layout adapter is required"
                )
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


def decoder_layers_for_ep(model: nn.Module) -> Iterable[nn.Module]:
    """Bind explicit decoder ownership; retain legacy entry points separately.

    A declared model must not recover from a malformed contract by reading a
    similarly named convenience property. Resolve current registered children
    so PP copies and parameter replacement cannot leave cached references.
    """
    structure = getattr(model, "decoder_structure", None)
    if structure is not None:
        if not isinstance(structure, DecoderStructure):
            raise TypeError("EP decoder_structure must be a DecoderStructure")
        return structure.bind(model).blocks
    layers = getattr(model, "layers", None)
    if layers is None:
        raise UnsupportedCombinationError(
            f"EP requires declared decoder structure or legacy .layers; "
            f"got {type(model).__name__}."
        )
    return layers


def find_moe_candidates(
    model: nn.Module, layers: Iterable[nn.Module], hf_config: Any
) -> list[tuple[nn.Module, str, nn.Module, bool]]:
    """Resolve declared slots before allocating; inspect only legacy models."""
    explicit_model = isinstance(
        getattr(model, "decoder_structure", None), DecoderStructure
    )
    inspection = None
    candidates = []
    for layer in layers:
        declared = declared_moe_block(layer)
        if declared is not None:
            attr, block = declared
        else:
            if explicit_model:
                raise ValueError(
                    f"Explicit decoder {type(layer).__name__} requires "
                    "moe_block_name (None for a dense layer); "
                    "legacy MoE inspection is disabled for declared models"
                )
            inspection = inspection or _legacy_inspection()
            attr, block = inspection.moe_block_of(layer)
        if block is None:
            continue
        if (
            explicit_model
            and not isinstance(block, (KimiSparseMoE, MoE))
            and not isinstance(getattr(block, "moe_contract", None), MoEBlockContract)
        ):
            raise UnsupportedCombinationError(
                f"Declared MoE slot {type(layer).__name__}.{attr} requires a "
                "model-owned MoEBlockContract; legacy layout inspection is disabled"
            )
        explicit = declared is not None
        if not (
            isinstance(block, KimiSparseMoE)
            or isinstance(block, MoE)
            or (explicit and resolve_packed_experts(block, degree=1) is not None)
            or (
                not explicit
                and inspection is not None
                and inspection.is_hf_moe_block(block)
            )
        ):
            if hasattr(layer, "moe_block_name"):
                raise UnsupportedCombinationError(
                    f"Declared MoE slot {type(layer).__name__}.{attr} holds "
                    f"{type(block).__name__}, whose expert layout has no EP "
                    "conversion adapter. Provide an explicit layout adapter "
                    "before enabling EP conversion."
                )
            continue
        if explicit:
            wants_router_logits = declared_router_output(block)
        else:
            inspection = inspection or _legacy_inspection()
            wants_router_logits = inspection.returns_router_logits(block)
        validate_router_output_conversion(block, wants_router_logits, hf_config)
        candidates.append((layer, attr, block, wants_router_logits))

    if not candidates:
        raise UnsupportedCombinationError(
            f"no declared or compatible MoE block found on {type(model).__name__}; "
            "EP requires a sparse decoder slot with a supported expert layout. "
            "Declare moe_block_name and a model-owned expert layout adapter."
        )
    return candidates


@dataclass(frozen=True, slots=True)
class MoEConversionTarget:
    """Decoder slot, output convention and routing choices fixed by parsing."""

    owner: nn.Module
    attribute: str
    router_logits: bool
    routing: MoERoutingBinding | None = None
    packed_experts: PackedExpertBinding | None = None
    contract: MoEBlockContract | None = None
    kimi_math: tuple[object, ...] | None = None
    declared_slot: bool = False


def validate_packed_expert_aliases(
    targets: Iterable[PlanTarget],
    module_paths: Iterable[tuple[str, nn.Module]],
    replaced_roots: list[str],
    *,
    degree: int,
) -> None:
    """Reject aliases that packed splitting or expert sharding cannot retain.

    Same-parameter sharing between converted blocks remains supported. Local
    down projections keep their Parameter and need no alias restriction. Read
    storage metadata only; do not allocate expert views or replacements here.
    """
    changed: set[int] = set()
    storages: dict[int, int] = {}
    for target in targets:
        binding = target.spec
        if not isinstance(binding, MoEConversionTarget):
            continue
        packed = binding.packed_experts
        if packed is None:
            continue
        for parameter in (packed.gate_up, *((packed.down,) if degree > 1 else ())):
            changed.add(id(parameter))
            if type(parameter) is nn.Parameter and parameter.layout == torch.strided:
                storage = parameter.untyped_storage()._cdata
                previous = storages.setdefault(storage, id(parameter))
                if previous != id(parameter):
                    raise UnsupportedCombinationError(
                        "EP cannot preserve packed expert parameters sharing storage"
                    )
    if not changed:
        return
    for path, module in module_paths:
        local = any(
            path == root or path.startswith(root + ".") for root in replaced_roots
        )
        for is_buffer, tensors in (
            (False, module.named_parameters(recurse=False, remove_duplicate=False)),
            (True, module.named_buffers(recurse=False, remove_duplicate=False)),
        ):
            for name, tensor in tensors:
                aliases_changed = id(tensor) in changed
                if (
                    type(tensor) in (torch.Tensor, nn.Parameter)
                    and tensor.layout == torch.strided
                ):
                    owner = storages.get(tensor.untyped_storage()._cdata)
                    aliases_changed |= owner is not None and owner != id(tensor)
                if aliases_changed and (
                    is_buffer or not local or id(tensor) not in changed
                ):
                    raise UnsupportedCombinationError(
                        "EP cannot preserve packed expert storage alias at "
                        f"{path + '.' if path else ''}{name}; split or sharded "
                        "expert state requires an adapted consumer"
                    )


def validate_ep_model_options(config: Any, implementation: str) -> None:
    """Reject unsupported execution choices before binding or changing a model.

    Parsing and execution share this check: a supplied plan does not authorize
    options changed after parsing, including score multiplication placement.
    """
    validate_expert_dispatcher(implementation)
    if getattr(config, "experts_implementation", "native") == "pre_down":
        raise UnsupportedCombinationError(
            "EP MoE replacement cannot preserve experts_implementation='pre_down'; "
            "use native experts or retain the local/DP/FSDP expert path. "
            "This also applies to single-rank fixed-router replacement."
        )


def validate_ep_model_geometry(
    model: nn.Module, *, degree: int, tp_enabled: bool
) -> None:
    """Validate native geometry while the model still owns unsharded storage.

    Reuse this before execution: config fields can change without invalidating
    module or tensor bindings. TP validates native geometry before sharding;
    never apply that native contract to already transformed parameters.
    """
    validate = getattr(model, "validate_parallel", None)
    if (
        callable(validate)
        and not tp_enabled
        and getattr(model, "parallel_plan", None) is None
        and getattr(model, "ep_parallel_plan", None) is None
    ):
        validate(tp_degree=1, ep_degree=degree)


def resolve_ep_plan(
    model: nn.Module,
    *,
    degree: int = 1,
    implementation: str = "alltoall",
    tp_enabled: bool = False,
    quantile_balancing: bool = False,
    has_ep_group: bool | None = None,
) -> ParallelPlan:
    """Bind EP conversion targets without allocating weights or running collectives.

    Generic routing restrictions are checked with the supplied execution options.
    Group presence defaults to degree > 1; pass it explicitly for size-one groups.
    Execution revalidates restrictions defensively if options or metadata change.
    Legacy models may still use inspection here; generic conversion consumes
    the selected routing rather than reading it again. Re-resolve after config,
    TP or PP changes. Kimi retains its dedicated conversion contract.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("EP degree must be a positive integer")
    config = getattr(model, "config", None)
    validate_ep_model_options(config, implementation)
    validate_ep_model_geometry(model, degree=degree, tp_enabled=tp_enabled)
    layers = decoder_layers_for_ep(model)
    candidates = find_moe_candidates(model, layers, config)
    modules = tuple(model.named_modules(remove_duplicate=False))
    targets = []
    for owner, attribute, block, router_logits in candidates:
        paths = [path for path, module in modules if module is owner]
        if not paths:
            raise ValueError("EP decoder owner is not registered in this model")
        routing = None
        packed_experts = None
        if isinstance(block, KimiSparseMoE):
            validate_kimi_conversion(
                block,
                degree=degree,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=implementation,
            )
        else:
            routing = resolve_moe_routing(block)
            validate_moe_conversion(
                block,
                routing,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=implementation,
                has_ep_group=degree > 1 if has_ep_group is None else has_ep_group,
            )
            packed_experts = resolve_packed_experts(block, degree=degree)
        targets.append(
            PlanTarget(
                f"{paths[0]}.{attribute}" if paths[0] else attribute,
                "moe",
                block,
                "kimi_per_expert"
                if isinstance(block, KimiSparseMoE)
                else (
                    "fused_gate_up" if packed_experts is not None else "legacy_adapter"
                ),
                MoEConversionTarget(
                    owner,
                    attribute,
                    router_logits,
                    routing,
                    packed_experts,
                    getattr(block, "moe_contract", None),
                    block.expert_parallel_math_signature()
                    if isinstance(block, KimiSparseMoE)
                    else None,
                    declared_slot=hasattr(owner, "moe_block_name"),
                ),
            )
        )
    module_paths = list(modules)
    roots = conversion_roots(candidates, module_paths)
    validate_packed_expert_aliases(targets, module_paths, roots, degree=degree)
    if any(isinstance(block, KimiSparseMoE) for _, _, block, _ in candidates):
        external_parameters, _ = external_state_owners(module_paths, roots)
        validate_kimi_expert_aliases(candidates, module_paths, external_parameters)
    return ParallelPlan(
        stage="ep",
        targets=tuple(targets),
        degree=degree,
        implementation=implementation,
        operations=("convert", "bind_shared_state", "install"),
        module_bindings=modules,
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )


def conversion_roots(
    candidates: list[tuple[nn.Module, str, nn.Module, bool]],
    module_paths: list[tuple[str, nn.Module]],
) -> list[str]:
    """Retain every registered decoder alias when describing replaced subtrees."""
    candidate_attrs: dict[int, list[str]] = {}
    for layer, attr, _, _ in candidates:
        candidate_attrs.setdefault(id(layer), []).append(attr)
    return [
        f"{path}.{attr}" if path else attr
        for path, module in module_paths
        for attr in candidate_attrs.get(id(module), ())
    ]


def external_state_owners(
    module_paths: list[tuple[str, nn.Module]], replaced_roots: list[str]
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
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
    return external_parameters, external_buffers


def swap_moe_blocks(
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
    resolved_plan: ParallelPlan | None = None,
) -> int:
    """Replace every HF MoE block in ``model`` with llmtuner's MoE, in place.

    Args:
        model: an explicit model (anything exposing ``.layers``).
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
        resolved_plan: optional target plan resolved for this exact model and
            EP degree/backend. Stale module or parameter bindings are rejected.
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
    decoder_layers_for_ep(model)
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

    hf_config = getattr(model, "config", None)
    validate_ep_model_options(hf_config, token_dispatcher)
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

    # Bind targets after preceding stages (TP/PP) have stabilized this instance.
    degree = 1 if ep_group is None else dist.get_world_size(ep_group)
    execution_plan = resolved_plan or resolve_ep_plan(
        model,
        degree=degree,
        implementation=token_dispatcher,
        tp_enabled=tp_enabled,
        quantile_balancing=quantile_balancing,
        has_ep_group=ep_group is not None,
    )
    if (
        execution_plan.stage != "ep"
        or execution_plan.degree != degree
        or execution_plan.implementation != token_dispatcher
    ):
        raise ValueError("EP plan does not match the requested execution")
    execution_plan.validate_binding(model)
    candidates = []
    routing_bindings = {}
    packed_bindings = {}
    for target in execution_plan.targets:
        binding = target.spec
        if target.kind != "moe" or not isinstance(binding, MoEConversionTarget):
            raise ValueError("EP plan requires resolved MoE conversion targets")
        if binding.owner._modules.get(binding.attribute) is not target.module:
            raise ValueError(f"Stale EP decoder slot: {target.path}")
        # Module identity alone cannot detect a model changing a sparse slot
        # to dense, or declaring a different child after plan resolution.
        if hasattr(binding.owner, "moe_block_name") != binding.declared_slot or (
            binding.declared_slot
            and getattr(binding.owner, "moe_block_name", None) != binding.attribute
        ):
            raise ValueError(f"Stale EP decoder declaration: {target.path}")
        if isinstance(target.module, KimiSparseMoE) and (
            target.module.expert_parallel_math_signature() != binding.kimi_math
        ):
            raise ValueError(f"Stale EP Kimi mathematics: {target.path}")
        if getattr(target.module, "moe_contract", None) != binding.contract:
            raise ValueError(f"Stale EP model contract: {target.path}")
        if binding.contract is not None:
            try:
                current_output = declared_router_output(target.module)
            except ValueError as error:
                raise ValueError(
                    f"Stale EP router output contract: {target.path}"
                ) from error
            if current_output != binding.router_logits:
                raise ValueError(f"Stale EP router output contract: {target.path}")
        validate_router_output_conversion(
            target.module, binding.router_logits, hf_config
        )
        candidates.append(
            (binding.owner, binding.attribute, target.module, binding.router_logits)
        )
        if isinstance(target.module, KimiSparseMoE):
            # Revalidate all targets before constructing the first replacement:
            # execution options/config can change without replacing tensors.
            validate_kimi_conversion(
                target.module,
                degree=degree,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
            )
        else:
            if binding.routing is None or binding.routing.source is not target.module:
                raise ValueError("EP plan requires a matching routing binding")
            validate_moe_conversion(
                target.module,
                binding.routing,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
                has_ep_group=ep_group is not None,
            )
            routing_bindings[id(target.module)] = binding.routing
            if binding.packed_experts is not None:
                binding.packed_experts.validate(target.module)
                packed_bindings[id(target.module)] = binding.packed_experts
    if not candidates:
        raise ValueError("EP plan must contain conversion targets")
    if isinstance(getattr(model, "decoder_structure", None), DecoderStructure):
        # EP replaces every declared sparse slot, in decoder order. A caller
        # can rebuild a frozen plan with missing/repeated targets while all
        # module snapshots remain valid; that must not produce a mixed model.
        expected_slots = []
        for layer in decoder_layers_for_ep(model):
            declared = declared_moe_block(layer)
            if declared is None:
                raise ValueError("Explicit EP requires declared decoder MoE slots")
            attribute, block = declared
            if block is not None:
                expected_slots.append((id(layer), attribute, id(block)))
        actual_slots = [
            (id(owner), attribute, id(block))
            for owner, attribute, block, _ in candidates
        ]
        if actual_slots != expected_slots:
            raise ValueError("EP plan must cover declared MoE slots in decoder order")
    validate_ep_model_geometry(model, degree=degree, tp_enabled=tp_enabled)

    module_paths = list(model.named_modules(remove_duplicate=False))
    replaced_roots = conversion_roots(candidates, module_paths)
    validate_packed_expert_aliases(
        execution_plan.targets, module_paths, replaced_roots, degree=degree
    )
    external_parameters, external_buffers = external_state_owners(
        module_paths, replaced_roots
    )

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
    validate_kimi_expert_aliases(candidates, module_paths, external_parameters)
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
            router_gate = routing_bindings[id(block)].router
            load_balance_coeff = (
                float(default_coeff)
                if getattr(router_gate, "e_score_correction_bias", None) is not None
                else None
            )
            moe = convert_block(
                block,
                routing=routing_bindings[id(block)],
                packed_experts=packed_bindings.get(id(block)),
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
            projection_names = (
                None
                if id(block) in packed_bindings
                else _legacy_inspection().per_expert_projection_names(block.experts)
            )
            projections: list[
                tuple[nn.Module, tuple[str, ...], tuple[torch.Tensor, ...]]
            ] = [(moe.router.gate, ("weight",), (router_gate.weight,))]
            if projection_names is None:
                projections.extend(
                    [
                        (grouped, ("w1_EFD", "w3_EFD"), (block.experts.gate_up_proj,)),
                        (grouped, ("w2_EDF",), (block.experts.down_proj,)),
                    ]
                )
            else:
                members = list(block.experts)
                projections.extend(
                    (
                        grouped,
                        (target_name,),
                        tuple(
                            getattr(expert, source_name).weight for expert in members
                        ),
                    )
                    for source_name, target_name in zip(
                        projection_names, ("w1_EFD", "w3_EFD", "w2_EDF"), strict=True
                    )
                )
            for owner, names, sources in projections:
                bind_shared_projection(
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
        converted_layouts = plan_tp_checkpoint_layouts(
            module_paths, candidates, converted_blocks, replaced_roots
        )

        if on_replacement is not None:
            module_targets = {}
            for _, _, block, _ in candidates:
                moe = converted_blocks[id(block)]
                module_targets[block] = moe
                module_targets[block.experts] = moe.routed_experts.inner_experts
                router = (
                    block.gate
                    if isinstance(block, KimiSparseMoE)
                    else routing_bindings[id(block)].router
                )
                module_targets[router] = moe.router.gate
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

    model.ep_parallel_plan = execution_plan
    logger.info("Converted %d model MoE blocks for EP execution", len(replacements))
    return len(replacements)


def swap_hf_moe_blocks(model: nn.Module, **kwargs: Any) -> int:
    """Compatibility alias for the former HF-specific entry point."""
    return swap_moe_blocks(model, **kwargs)
