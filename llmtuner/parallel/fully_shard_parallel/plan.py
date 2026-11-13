"""Resolve FSDP patterns, shared ownership, policies and local PP bindings."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import Shard

from llmtuner.config.parallel_config import ParallelConfig

from ...accelerator.capabilities import require
from ...models.common.moe.experts import GroupedExperts
from ..module_patterns import compile_module_pattern, resolve_module_patterns
from ..plan import ParallelPlan, PlanTarget
from .fine_grained import get_fsdp_strategy, validate_fsdp_hook_target
from .ownership import named_module_paths
from .policies import (
    fsdp_shard_size,
    get_fsdp_reshard_after_forward_policy,
)

__all__ = [
    "FSDPPatternPlan",
    "resolve_fsdp_pattern_plan",
    "resolve_pipeline_fsdp_configs",
]


def _resolve_pattern_units(
    model: nn.Module, cfg: ParallelConfig
) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]], set[nn.Parameter]]:
    named_modules = dict(model.named_modules())
    # Keep physical paths available, while accepting the model's paths before
    # AC/whole-model compile inserted transparent wrappers. The outer module
    # wins at a wrapper boundary so selecting a block still selects its wrapper.
    module_paths = named_module_paths(model)
    ignored_roots = resolve_module_patterns(
        list(module_paths),
        patterns=cfg.fsdp_config.fsdp_ignored_modules,
        plan_name="FSDP ignored",
    )

    # Ignoring a module excludes its complete subtree, matching the recursive
    # ignored-parameter collection below.
    ignored_modules = {
        module for root in ignored_roots for module in module_paths[root].modules()
    }
    canonical_names = {module: name for name, module in named_modules.items()}

    unit_options: dict[str, dict[str, Any]] = {}
    for pattern, overrides in cfg.fsdp_config.fsdp_apply_modules.items():
        compiled = compile_module_pattern(pattern)
        matches = [
            name
            for name, module in module_paths.items()
            if (name or pattern == "")
            and module not in ignored_modules
            and compiled.fullmatch(name)
        ]
        if not matches:
            raise ValueError(f"FSDP apply pattern {pattern!r} matched no module")
        for name in matches:
            canonical = canonical_names[module_paths[name]]
            unit_options.setdefault(canonical, {}).update(overrides)
    if not unit_options:
        raise ValueError("fsdp_apply_modules selected no FSDP unit")

    ignored_params = {
        parameter
        for name in ignored_roots
        for parameter in module_paths[name].parameters(recurse=True)
    }
    return named_modules, unit_options, ignored_params


def validate_fsdp_ignored_parameters(
    exclusions: Any, *, parameters: set[nn.Parameter], unit_name: str
) -> None:
    """Reject stale or foreign exclusions before ownership is rebound."""
    if exclusions is None:
        return
    if not isinstance(exclusions, set) or any(
        not isinstance(parameter, nn.Parameter) for parameter in exclusions
    ):
        raise TypeError(
            f"FSDP unit {unit_name!r}: ignored_params must be a set of Parameters "
            "or None"
        )
    if exclusions - parameters:
        raise ValueError(
            f"FSDP unit {unit_name!r}: ignored_params contains parameters not "
            "registered in the model; use the current model's Parameter objects"
        )


def resolve_pipeline_fsdp_configs(
    model: nn.Module, parts: list[nn.Module], cfg: ParallelConfig
) -> list[ParallelConfig]:
    """Bind a whole-model pattern plan to copied pipeline parts by registered FQN.

    Global patterns must match before selecting local units. Module and parameter
    overrides refer to the copied objects, while meshes and policies retain their
    identity. A part without a selected child still gets its root FSDP state.
    """
    source_modules, unit_options, _ = _resolve_pattern_units(model, cfg)
    source_parameters = set(model.parameters())
    for name, options in unit_options.items():
        validate_fsdp_hook_target([source_modules[name]], options.get("hook_module"))
        validate_fsdp_ignored_parameters(
            options.get("ignored_params"), parameters=source_parameters, unit_name=name
        )
    source_paths = dict(model.named_modules(remove_duplicate=False))
    module_aliases: dict[nn.Module, list[str]] = {}
    for path, module in source_paths.items():
        module_aliases.setdefault(module, []).append(path)
    parameter_names: dict[nn.Parameter, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(parameter, []).append(name)
    ignored_names = resolve_module_patterns(
        list(source_paths),
        patterns=cfg.fsdp_config.fsdp_ignored_modules,
        plan_name="FSDP ignored",
    )

    def local_module_binding(
        source: nn.Module, local_modules: dict[str, nn.Module]
    ) -> tuple[str, nn.Module] | None:
        candidates = {
            local_modules[path]: path
            for path in reversed(module_aliases[source])
            if path in local_modules
            and not (
                isinstance(local_modules[path], nn.Identity)
                and not isinstance(source, nn.Identity)
            )
        }
        if len(candidates) > 1:
            raise ValueError(
                "Pipeline FSDP module aliases no longer share one owner: "
                f"{module_aliases[source]}"
            )
        return next(((path, module) for module, path in candidates.items()), None)

    def local_parameter_bindings(
        exclusions: set[nn.Parameter],
        local_parameters: dict[str, nn.Parameter],
    ) -> set[nn.Parameter]:
        rebound = set()
        for parameter in exclusions:
            candidates = {
                local_parameters[path]
                for path in parameter_names[parameter]
                if path in local_parameters
            }
            if len(candidates) > 1:
                raise ValueError(
                    "Pipeline FSDP parameter aliases no longer share one owner: "
                    f"{parameter_names[parameter]}"
                )
            rebound.update(candidates)
        return rebound

    configs = []
    for part in parts:
        local_modules = dict(part.named_modules(remove_duplicate=False))
        local_parameters = dict(part.named_parameters(remove_duplicate=False))

        local_options = {}
        for name, overrides in unit_options.items():
            binding = local_module_binding(source_modules[name], local_modules)
            if binding is None:
                continue
            local_name, _ = binding
            options = overrides.copy()
            hook = options.get("hook_module")
            if hook is not None:
                hook_binding = local_module_binding(hook, local_modules)
                if hook_binding is None:
                    raise ValueError(
                        f"FSDP unit {name!r}: hook_module must belong to its "
                        "pipeline part"
                    )
                options["hook_module"] = hook_binding[1]
            exclusions = options.get("ignored_params")
            if exclusions is not None:
                options["ignored_params"] = local_parameter_bindings(
                    exclusions, local_parameters
                )
            local_options.setdefault(local_name, {}).update(options)
        local_ignored = [
            binding[0]
            for name in ignored_names
            if (binding := local_module_binding(source_paths[name], local_modules))
            is not None
        ]
        configs.append(
            replace(
                cfg,
                fsdp_config=replace(
                    cfg.fsdp_config,
                    fsdp_apply_modules=local_options or {"": {}},
                    fsdp_ignored_modules=list(dict.fromkeys(local_ignored)),
                ),
            )
        )
    return configs


def _resolve_shared_fsdp_units(
    model: nn.Module,
    *,
    named_modules: dict[str, nn.Module],
    unit_options: dict[str, dict[str, Any]],
    ignored_params: set[nn.Parameter],
    common: dict[str, Any],
) -> list[tuple[list[str], dict[str, Any]]]:
    """Build a validated child-first plan without transforming the model.

    Shared parameter owners and dependency cycles must use one FSDP state.
    Return canonical module names and independent effective option dictionaries.
    """
    unit_names = sorted(unit_options, key=lambda item: item.count("."), reverse=True)
    # Shared parameters require one state managing all of their owners. Resolve
    # ownership before any transformation replaces the original Parameters.
    groups = {name: {name} for name in [*unit_names, ""]}
    parameter_owners: dict[nn.Parameter, set[str]] = {}
    # An aliased child may execute under several selected parents. Walking only
    # its canonical path would hide the other owners and split shared storage
    # across independent FSDP states.
    for module_name, module in model.named_modules(remove_duplicate=False):
        owner = next(
            (
                name
                for name in unit_names
                if module_name == name or module_name.startswith(f"{name}.")
            ),
            "",
        )
        for parameter in module.parameters(recurse=False):
            if parameter not in ignored_params:
                parameter_owners.setdefault(parameter, set()).add(owner)
    for owners in parameter_owners.values():
        merged = set().union(*(groups[name] for name in owners))
        for name in merged:
            groups[name] = merged
    grouped_names = list({frozenset(group) for group in groups.values()})
    registration_order = {name: index for index, name in enumerate(named_modules)}
    # Shared groups may span different depths. Depth sorting cannot ensure
    # child-before-parent order, and cross-branch sharing may form a cycle.
    # Collapse cycles into a single state, then order the remaining groups.
    while True:
        dependencies = {
            group: {
                child_group
                for child_group in grouped_names
                if child_group != group
                and any(
                    parent == "" or child.startswith(f"{parent}.")
                    for parent in group
                    for child in child_group
                )
            }
            for group in grouped_names
        }
        sorter = TopologicalSorter(dependencies)
        try:
            sorter.prepare()
        except CycleError as error:
            cycle = set(error.args[1])
            merged = frozenset().union(*cycle)
            grouped_names = [group for group in grouped_names if group not in cycle]
            grouped_names.append(merged)
            continue
        grouped_names = []
        while sorter.is_active():
            ready = sorted(
                sorter.get_ready(),
                key=lambda group: min(registration_order[name] for name in group),
            )
            grouped_names.extend(ready)
            sorter.done(*ready)
        break
    grouped_options: list[tuple[list[str], dict[str, Any]]] = []
    for group in grouped_names:
        names = sorted(group, key=registration_order.__getitem__)
        effective_options = []
        for name in names:
            options = common | unit_options.get(name, {})
            # Compare the effective exclusion boundary: None, empty sets, and
            # redundant global exclusions all represent the same policy.
            options["ignored_params"] = ignored_params | (
                options["ignored_params"] or set()
            )
            effective_options.append(options)
        options = effective_options[0]
        for other_options in effective_options[1:]:
            if options != other_options:
                raise ValueError(
                    f"FSDP units {names} share parameters but have conflicting "
                    "options; configure the shared owners consistently"
                )
        grouped_options.append((names, options))
    return grouped_options


def validate_fsdp_pattern_options(
    named_modules: dict[str, nn.Module],
    unit_options: dict[str, dict[str, Any]],
    *,
    implementation: str,
) -> None:
    """Reject invalid unit overrides before any model transformation."""
    for name in [*unit_options, ""]:
        if isinstance(named_modules[name], nn.ModuleList | nn.ModuleDict):
            raise ValueError(
                f"FSDP unit {name!r} is a container without forward; select its "
                "children or a parent module that implements forward"
            )
    allowed_overrides = {
        name
        for name, parameter in inspect.signature(fully_shard).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    } | {"hook_module"}
    parameters = {
        parameter
        for module in named_modules.values()
        for parameter in module.parameters(recurse=False)
    }
    for name, options in unit_options.items():
        unknown = options.keys() - allowed_overrides
        if unknown:
            raise ValueError(
                f"FSDP unit {name!r} has unsupported overrides: {sorted(unknown)}"
            )
        for key, policy_type in (
            ("mp_policy", MixedPrecisionPolicy),
            ("offload_policy", OffloadPolicy),
        ):
            if key not in options:
                continue
            policy = options[key]
            if policy is None and implementation == "custom":
                continue  # The custom engine constructs its default policy.
            if not isinstance(policy, policy_type):
                raise TypeError(
                    f"FSDP unit {name!r}: {key} must be {policy_type.__name__}"
                )
        placement = options.get("shard_placement_fn")
        if placement is not None and not callable(placement):
            raise TypeError(
                f"FSDP unit {name!r}: shard_placement_fn must be callable or None"
            )
        validate_fsdp_ignored_parameters(
            options.get("ignored_params"), parameters=parameters, unit_name=name
        )
    # Validate every hook override before sharding any unit. A late rejection
    # would otherwise leave earlier units transformed on a failed assembly.
    for name, options in unit_options.items():
        if "hook_module" not in options:
            continue
        if implementation == "native":
            raise ValueError("native FSDP does not accept a hook_module override")
        validate_fsdp_hook_target([named_modules[name]], options["hook_module"])


@dataclass(frozen=True, slots=True)
class FSDPPatternPlan:
    """Validated child-first units and backend, without sharding model storage.

    Bindings belong to the current instance. Execution copies each unit's option
    dictionary before attaching runtime placement functions. Resolve again after
    any structural or parameter replacement; do not cache plans across stages.
    """

    bindings: ParallelPlan
    named_modules: Mapping[str, nn.Module]
    units: tuple[tuple[tuple[str, ...], Mapping[str, Any]], ...]
    expert_placements: Mapping[nn.Parameter, Shard]
    expert_counts: Mapping[nn.Parameter, int]
    shard_fn: Callable[..., Any]
    ep_degree: int
    edp_mesh: DeviceMesh | None
    prefetch: tuple[int, int]
    symm_mem_scope: str | None

    def __post_init__(self) -> None:
        # Snapshot containers, preserving the identity of modules, policies
        # and borrowed meshes. The executor copies options before adding its
        # placement functions; no tensor storage is duplicated here.
        for name in ("named_modules", "expert_placements", "expert_counts"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "units",
            tuple(
                (tuple(names), MappingProxyType(dict(options)))
                for names, options in self.units
            ),
        )


def resolve_fsdp_pattern_plan(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    cfg: ParallelConfig,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    edp_mesh: DeviceMesh | None = None,
) -> FSDPPatternPlan:
    """Resolve ownership, policies and capabilities before wrapping any unit.

    Only local metadata and backend availability are inspected. Process groups
    must already exist; this resolver performs no collective or tensor copies.
    """
    named_modules, unit_options, ignored_params = _resolve_pattern_units(model, cfg)
    validate_fsdp_pattern_options(
        named_modules,
        unit_options,
        implementation=cfg.fsdp_config.fsdp_implementation,
    )
    expert_placements: dict[nn.Parameter, Shard] = {}
    expert_counts: dict[nn.Parameter, int] = {}
    if cfg.ep_size > 1:
        if edp_mesh is None:
            raise ValueError("pattern FSDP with EP requires a sparse FSDP mesh")
        degree = fsdp_shard_size(edp_mesh)
        expert_placements = {
            parameter: Shard(1 if degree > module.num_experts else 0)
            for module in model.modules()
            if isinstance(module, GroupedExperts)
            for parameter in module.parameters(recurse=False)
            if parameter not in ignored_params
        }
    else:
        expert_counts = {
            parameter: module.num_experts
            for module in model.modules()
            if isinstance(module, GroupedExperts)
            for parameter in module.parameters(recurse=False)
            if parameter not in ignored_params
        }

    shard_fn = get_fsdp_strategy(
        cfg.fsdp_config.fsdp_implementation
    ).get_fully_shard_fn()
    common: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
        "ignored_params": ignored_params,
        "reshard_after_forward": get_fsdp_reshard_after_forward_policy(
            cfg.fsdp_config.fsdp_reshard_after_forward, cfg.pp_size > 1
        ),
    }
    if cfg.fsdp_config.fsdp_cpu_offload:
        common["offload_policy"] = CPUOffloadPolicy(
            pin_memory=cfg.fsdp_config.fsdp_cpu_offload_pin_memory
        )

    grouped_options = _resolve_shared_fsdp_units(
        model,
        named_modules=named_modules,
        unit_options=unit_options,
        ignored_params=ignored_params,
        common=common,
    )
    for names, options in grouped_options:
        validate_fsdp_hook_target(
            [named_modules[name] for name in names], options.get("hook_module")
        )
        # Reject every effective reshard policy before transforming any unit.
        # HSDP uses the shard axis, not the total mesh size. Named SPMD meshes
        # currently accept only bool/None for sharded DP in PyTorch 2.14.
        reshard = options["reshard_after_forward"]
        if reshard is None or isinstance(reshard, bool):
            continue
        if not isinstance(reshard, int):
            raise ValueError(
                f"FSDP units {names}: reshard_after_forward must be bool, int "
                f"or None, got {reshard!r}"
            )
        mesh_dims = options.get("dp_mesh_dims")
        if mesh_dims is not None:
            if mesh_dims.shard is not None:
                raise NotImplementedError(
                    f"FSDP units {names}: integer reshard_after_forward is not "
                    "supported with sharded dp_mesh_dims in PyTorch 2.14"
                )
            # Replicate-only named DP has no post-forward sharding policy.
            continue
        mesh = options["mesh"]
        if mesh is None:
            degree = torch.distributed.get_world_size()
        else:
            degree = mesh.size(0 if mesh.ndim == 1 else 1)
        if reshard < 1 or reshard > degree or degree % reshard:
            raise ValueError(
                f"FSDP units {names}: reshard_after_forward must be a positive "
                f"factor of shard mesh size {degree}, got {reshard}"
            )
    if (
        cfg.ep_size > 1
        and expert_placements
        and any(
            options.get("shard_placement_fn") is None for _, options in grouped_options
        )
    ):
        # Check before wrapping even units with custom placement functions:
        # a later default unit must not fail after earlier units were mutated.
        require(
            "fsdp_per_param_mesh",
            feature="expert parallelism combined with pattern-selected FSDP",
        )
    return FSDPPatternPlan(
        bindings=ParallelPlan(
            stage="fsdp",
            degree=fsdp_shard_size(dp_mesh),
            degrees=(cfg.ep_size,),
            implementation=cfg.fsdp_config.fsdp_implementation,
            targets=tuple(
                PlanTarget(name, "fsdp_unit", named_modules[name])
                for names, _ in grouped_options
                for name in names
            ),
            operations=("fully_shard",),
            module_bindings=tuple(model.named_modules(remove_duplicate=False)),
            parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
            buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
        ),
        named_modules=named_modules,
        units=tuple((tuple(names), options) for names, options in grouped_options),
        expert_placements=expert_placements,
        expert_counts=expert_counts,
        shard_fn=shard_fn,
        ep_degree=cfg.ep_size,
        edp_mesh=edp_mesh,
        prefetch=(
            cfg.fsdp_config.fsdp_forward_prefetch_count,
            cfg.fsdp_config.fsdp_backward_prefetch_count,
        ),
        symm_mem_scope=(
            cfg.fsdp_config.fsdp_symm_mem_scope
            if cfg.fsdp_config.enable_fsdp_symm_mem
            else None
        ),
    )
