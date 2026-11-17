"""Parameter ownership and checkpoint bindings for planned EP replacements.

These helpers inspect registered aliases and populate detached replacement
modules. They never install decoder blocks or perform distributed collectives.
The caller validates all targets before publishing any replacement.
"""

from __future__ import annotations

from torch import nn

from ...errors import UnsupportedCombinationError
from ...models.common.moe.layer import MoE


def bind_shared_projection(
    owner: nn.Module,
    names: tuple[str, ...],
    source: nn.Parameter,
    *,
    converted_parameters: dict[int, tuple[nn.Parameter, ...]],
    external_parameters: dict[int, list[str]],
) -> None:
    """Bind one new projection while preserving source ownership and row order.

    Only replacement modules and local planning indexes are changed here;
    installation into the live model remains the caller's final phase.
    """
    shared = converted_parameters.get(id(source))
    if shared is None:
        targets = tuple(getattr(owner, name) for name in names)
        # Unchanged layouts can keep the original Parameter, also
        # preserving consumers outside the converted MoE subtree.
        # Split gate/up and EP-sharded projections have new layouts.
        if (
            len(targets) == 1
            and type(source) is type(targets[0])
            and source.shape == targets[0].shape
            and source.dtype == targets[0].dtype
            and source.device == targets[0].device
        ):
            setattr(owner, names[0], source)
            targets = (source,)
        external = external_parameters.get(id(source), [])
        if external and not (len(targets) == 1 and targets[0] is source):
            raise UnsupportedCombinationError(
                "MoE conversion changes the layout of a "
                "parameter also registered outside converted "
                f"blocks at {external}; split or EP-sharded expert aliases "
                "require an adapted external consumer. Register aliases to "
                "the converted expert parameters after EP conversion."
            )
        converted_parameters[id(source)] = targets
        return
    if len(shared) != len(names):
        raise UnsupportedCombinationError(
            "MoE conversion requires shared projections to have the same "
            "number of target parameters in every converted block."
        )
    for name, parameter in zip(names, shared, strict=True):
        setattr(owner, name, parameter)


def plan_tp_checkpoint_layouts(
    module_paths: list[tuple[str, nn.Module]],
    expert_replacements: dict[nn.Module, nn.Module],
    converted_blocks: dict[int, MoE],
    replaced_roots: list[str],
    *,
    router_replacements: dict[nn.Module, nn.Module],
) -> list[tuple[nn.Module, dict]]:
    """Plan TP registration renames without modifying any live owner."""
    # TP registers physical HF slices before this swap. Gate/up becomes two
    # independent parameters, so each checkpoint slice now has its own name
    # and no longer uses packed-half gathering. Preserve the registration's
    # owner, which may be the model root rather than the replaced block.
    projection_names = {
        "gate_up_proj": ("w1_EFD", "w3_EFD"),
        "down_proj": ("w2_EDF",),
    }
    projection_replacements = {
        f"{root}.experts.{name}": targets
        for root in replaced_roots
        for name, targets in projection_names.items()
    }
    child_renames = []
    for path, block in module_paths:
        if id(block) not in converted_blocks:
            continue
        if block._modules.get("experts") not in expert_replacements:
            continue  # Model-owned conversions own their internal paths.
        contract = block.moe_contract
        prefix = f"{path}." if path else ""
        child_renames.append(
            (path, f"{prefix}{contract.router}.", f"{prefix}router.gate.")
        )
        if contract.shared_expert is not None:
            child_renames.append(
                (
                    path,
                    f"{prefix}{contract.shared_expert}.",
                    f"{prefix}shared_experts.",
                )
            )
    converted_layouts = []
    for owner_path, owner in module_paths:
        layouts = owner.__dict__.get("_tp_checkpoint_layouts")
        if layouts is None:
            continue
        router_owner = router_replacements.get(owner)
        if router_owner is not None:
            # Router projections keep their local parameter names and axes;
            # only their owning module changes during EP conversion.
            converted_layouts.append((router_owner, dict(layouts)))
            continue
        expert_owner = expert_replacements.get(owner)
        if expert_owner is not None:
            updated = {}
            for name, layout in layouts.items():
                names = projection_names.get(name, (name,))
                for target_name in names:
                    updated[target_name] = (
                        (layout[0], False, layout[2])
                        if name in ("gate_up_proj", "down_proj")
                        else layout
                    )
            converted_layouts.append((expert_owner, updated))
            continue
        updated = {}
        # Ancestor ownership is constant for all registrations on this module.
        renames = [
            (source, target)
            for root, source, target in child_renames
            if not owner_path or owner_path == root or root.startswith(owner_path + ".")
        ]
        for name, layout in layouts.items():
            full_name = f"{owner_path}.{name}" if owner_path else name
            replacement = projection_replacements.get(full_name)
            if replacement is None:
                for source, target in renames:
                    # Only ancestor registrations need path changes. A reused
                    # child's own local names remain relative to that child.
                    if full_name.startswith(source):
                        renamed = target + full_name[len(source) :]
                        name = renamed[len(owner_path) + 1 :] if owner_path else renamed
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


def validate_model_expert_ownership(
    candidates: list[tuple[nn.Module, str, nn.Module, bool]],
    module_paths: list[tuple[str, nn.Module]],
    external_parameters: dict[int, list[str]],
) -> None:
    """Invoke each model-owned ownership rule once for its declared blocks.

    Group by the underlying hook function, not by model names or layouts.
    Repeated decoder aliases of one physical block are passed only once.
    """
    validators = {}
    for _, _, block, _ in candidates:
        hook = getattr(block, "validate_expert_parallel_ownership", None)
        if not callable(hook):
            continue
        key = getattr(hook, "__func__", hook)
        if key not in validators:
            validators[key] = (hook, {})
        validators[key][1].setdefault(id(block), block)
    for hook, blocks in validators.values():
        hook(
            blocks=tuple(blocks.values()),
            module_paths=module_paths,
            external_parameters=external_parameters,
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
    root_paths = set(replaced_roots)
    root_prefixes = tuple(root + "." for root in replaced_roots)
    for path, module in module_paths:
        if path in root_paths or path.startswith(root_prefixes):
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
