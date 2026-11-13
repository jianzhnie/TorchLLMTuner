"""Parameter ownership and checkpoint bindings for planned EP replacements.

These helpers inspect registered aliases and populate detached replacement
modules. They never install decoder blocks or perform distributed collectives.
The caller validates all targets before publishing any replacement.
"""

from __future__ import annotations

import torch
from torch import nn

from ...errors import UnsupportedCombinationError
from ...models.common.moe.layer import MoE
from ...models.kimi.kimi_k3.moe import KimiSparseMoE


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
