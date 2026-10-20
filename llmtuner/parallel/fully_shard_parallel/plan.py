"""Pattern selection and explicit prefetch plans for FSDP units."""

from __future__ import annotations

import inspect
from dataclasses import replace
from graphlib import CycleError, TopologicalSorter
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import Shard

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.utils.module_names import canonical_fqn

from ...accelerator.capabilities import require
from ..module_patterns import compile_module_pattern, resolve_module_patterns
from .fine_grained import get_fsdp_strategy, validate_fsdp_hook_target


class FSDPPatternReferences:
    """Keep programmatic overrides bound across module/parameter replacement.

    Capture registered aliases before transforms, then resolve logical paths
    immediately before FSDP. Objects retained under renamed paths remain valid.
    Layout conversions that remove every alias require an explicit mapping;
    rejecting them prevents silently losing an exclusion or hook override.
    """

    def __init__(self, model: nn.Module, cfg: ParallelConfig) -> None:
        self.cfg = cfg
        self.module_paths: dict[nn.Module, list[str]] = {}
        self.parameter_paths: dict[nn.Parameter, list[str]] = {}
        self.module_targets: dict[nn.Module, nn.Module] = {}
        self.parameter_targets: dict[nn.Parameter, tuple[nn.Parameter, ...]] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            self.module_paths.setdefault(module, []).append(canonical_fqn(name))
        for name, parameter in model.named_parameters(remove_duplicate=False):
            self.parameter_paths.setdefault(parameter, []).append(canonical_fqn(name))
        self.unit_modules = dict(model.named_modules())
        module_paths = dict(model.named_modules(remove_duplicate=False))
        for name, module in tuple(module_paths.items()):
            module_paths.setdefault(canonical_fqn(name), module)
        self.ignored_modules: dict[str, nn.Module | None] = {}
        ignored = set()
        for pattern in cfg.fsdp_config.fsdp_ignored_modules:
            compiled = compile_module_pattern(pattern)
            matches = {
                name: module
                for name, module in module_paths.items()
                if compiled.fullmatch(name)
            }
            self.ignored_modules.update(matches or {pattern: None})
            ignored.update(
                child for module in matches.values() for child in module.modules()
            )
        self.ignored_parameters = {
            parameter
            for module in ignored
            for parameter in module.parameters(recurse=False)
        }
        canonical_names = {module: name for name, module in self.unit_modules.items()}
        self.unit_options: dict[str, dict[str, Any]] = {}
        for pattern, options in cfg.fsdp_config.fsdp_apply_modules.items():
            compiled = compile_module_pattern(pattern)
            matches = [
                canonical_names[module]
                for name, module in module_paths.items()
                if (name or pattern == "")
                and module not in ignored
                and compiled.fullmatch(name)
            ]
            # Patterns may intentionally name the native EP layout, which
            # does not exist until conversion. Validate those at FSDP entry.
            for name in matches or [pattern]:
                self.unit_options.setdefault(name, {}).update(options)

    def record_replacements(
        self,
        modules: dict[nn.Module, nn.Module],
        parameters: dict[nn.Parameter, tuple[nn.Parameter, ...]],
    ) -> None:
        """Record explicit one-to-many mappings supplied by the conversion."""
        contributors: dict[nn.Parameter, set[nn.Parameter]] = {}
        for source, targets in parameters.items():
            for target in targets:
                contributors.setdefault(target, set()).add(source)
        globally_ignored = {
            parameter
            for module in self.ignored_modules.values()
            if module is not None
            for parameter in module.parameters()
        }
        for name, options in self.unit_options.items():
            exclusions = options.get("ignored_params") or set()
            if not isinstance(exclusions, set):
                continue  # Preserve the FSDP override type validation.
            excluded = globally_ignored | exclusions
            for sources in contributors.values():
                if sources & excluded and not sources <= excluded:
                    raise UnsupportedCombinationError(
                        f"FSDP pattern {name!r}: cannot exclude part of a stacked "
                        "expert parameter; exclude all source projections in "
                        "the stack or use an unstacked expert layout"
                    )
        self.module_targets.update(modules)
        self.parameter_targets.update(parameters)

    def bind(self, model: nn.Module) -> ParallelConfig:
        modules = {}
        for name, module in model.named_modules(remove_duplicate=False):
            # The outer transparent wrapper owns the logical boundary.
            modules.setdefault(canonical_fqn(name), module)
        parameters = {
            canonical_fqn(name): parameter
            for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        current_modules = set(modules.values())
        current_parameters = set(parameters.values())

        def bind_parameters(
            exclusions: set[nn.Parameter], context: str
        ) -> set[nn.Parameter]:
            rebound = set()
            for parameter in exclusions:
                targets = {
                    parameters[path]
                    for path in self.parameter_paths.get(parameter, ())
                    if path in parameters
                }
                targets.update(
                    target
                    for target in self.parameter_targets.get(parameter, ())
                    if target in current_parameters
                )
                if parameter in current_parameters:
                    targets.add(parameter)
                if not targets:
                    raise ValueError(
                        f"{context}: ignored_params lost its registered paths "
                        "during model transformation; configure exclusions "
                        "on the converted model"
                    )
                rebound.update(targets)
            return rebound

        overrides = {}
        for name, source in self.unit_options.items():
            pattern = canonical_fqn(name)
            if pattern not in modules:
                target = self.module_targets.get(self.unit_modules.get(name))
                pattern = next(
                    (path for path, module in modules.items() if module is target),
                    pattern,
                )
            options = source.copy()
            hook = options.get("hook_module")
            if isinstance(hook, nn.Module):
                targets = [
                    modules[path]
                    for path in self.module_paths.get(hook, ())
                    if path in modules
                ]
                if targets:
                    options["hook_module"] = targets[0]
                elif self.module_targets.get(hook) in current_modules:
                    options["hook_module"] = self.module_targets[hook]
                elif hook not in current_modules:
                    raise ValueError(
                        f"FSDP pattern {pattern!r}: hook_module lost its registered "
                        "paths during model transformation; configure the "
                        "override on the converted model"
                    )
            exclusions = options.get("ignored_params")
            if isinstance(exclusions, set):
                options["ignored_params"] = bind_parameters(
                    exclusions, f"FSDP pattern {pattern!r}"
                )
            overrides[pattern] = options
        ignored_names = []
        for name, source_module in self.ignored_modules.items():
            name = canonical_fqn(name)
            if name not in modules and source_module is not None:
                target = self.module_targets.get(source_module, source_module)
                name = next(
                    (path for path, module in modules.items() if module is target), name
                )
            ignored_names.append(name)
        # A source subtree can split across owners (Kimi gate weight and
        # correction bias). Preserve moved parameters without excluding the
        # target parameter's entire new parent subtree.
        module_exclusions = {
            parameter
            for pattern in ignored_names
            for name, module in modules.items()
            if compile_module_pattern(pattern).fullmatch(name)
            for parameter in module.parameters()
        }
        moved_exclusions = (
            bind_parameters(self.ignored_parameters, "FSDP ignored modules")
            - module_exclusions
        )
        if moved_exclusions:
            overrides.setdefault("", {})
            for options in overrides.values():
                exclusions = options.get("ignored_params")
                if exclusions is None or isinstance(exclusions, set):
                    options["ignored_params"] = (exclusions or set()) | moved_exclusions
        return replace(
            self.cfg,
            fsdp_config=replace(
                self.cfg.fsdp_config,
                fsdp_apply_modules=overrides,
                fsdp_ignored_modules=list(dict.fromkeys(ignored_names)),
            ),
        )


def set_fsdp_prefetch_windows(
    modules: list[nn.Module], *, forward_count: int, backward_count: int
) -> None:
    """Configure explicit forward and backward prefetch target windows.

    Empty backward targets retain PyTorch's implicit prefetch policy. This
    helper does not disable that policy or infer arbitrary forward execution
    order; callers supply the ordered units.
    """
    for name, count in (
        ("forward_count", forward_count),
        ("backward_count", backward_count),
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {count!r}")
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("FSDP prefetch units must be distinct module objects")
    for index, module in enumerate(modules):
        for method in (
            "set_modules_to_forward_prefetch",
            "set_modules_to_backward_prefetch",
        ):
            if not callable(getattr(module, method, None)):
                raise TypeError(f"FSDP prefetch unit {index} lacks callable {method}")
    # Grouped fully_shard calls attach one state to several modules. Configure
    # each state once, using its first occurrence in the supplied forward order.
    # Otherwise later members overwrite the same state's prefetch windows.
    units = []
    seen_states = set()
    for module in modules:
        state = module._get_fsdp_state() if isinstance(module, FSDPModule) else module
        if id(state) not in seen_states:
            seen_states.add(id(state))
            units.append(module)
    modules = units
    for index, module in enumerate(modules):
        forward = modules[index + 1 : index + 1 + forward_count]
        module.set_modules_to_forward_prefetch(forward)  # type: ignore[attr-defined]
        backward = list(reversed(modules[max(0, index - backward_count) : index]))
        module.set_modules_to_backward_prefetch(backward)  # type: ignore[attr-defined]


def _resolve_pattern_units(
    model: nn.Module, cfg: ParallelConfig
) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]], set[nn.Parameter]]:
    named_modules = dict(model.named_modules())
    module_paths = dict(model.named_modules(remove_duplicate=False))
    # Keep physical paths available, while accepting the model's paths before
    # AC/whole-model compile inserted transparent wrappers. The outer module
    # wins at a wrapper boundary so selecting a block still selects its wrapper.
    for name, module in tuple(module_paths.items()):
        logical_name = canonical_fqn(
            name, wrapper_segments=("_checkpoint_wrapped_module", "_orig_mod")
        )
        module_paths.setdefault(logical_name, module)
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


def resolve_pipeline_fsdp_configs(
    model: nn.Module, parts: list[nn.Module], cfg: ParallelConfig
) -> list[ParallelConfig]:
    """Bind a whole-model pattern plan to copied pipeline parts by registered FQN.

    Global patterns must match before selecting local units. Module and parameter
    overrides refer to the copied objects, while meshes and policies retain their
    identity. A part without a selected child still gets its root FSDP state.
    """
    source_modules, unit_options, _ = _resolve_pattern_units(model, cfg)
    for name, options in unit_options.items():
        validate_fsdp_hook_target([source_modules[name]], options.get("hook_module"))
        exclusions = options.get("ignored_params")
        if exclusions is not None and (
            not isinstance(exclusions, set)
            or any(not isinstance(parameter, nn.Parameter) for parameter in exclusions)
        ):
            raise TypeError(
                f"FSDP unit {name!r}: ignored_params must be a Parameter set"
            )
    module_names = {module: name for name, module in source_modules.items()}
    parameter_names: dict[nn.Parameter, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(parameter, []).append(name)
    ignored_names = resolve_module_patterns(
        list(dict(model.named_modules(remove_duplicate=False))),
        patterns=cfg.fsdp_config.fsdp_ignored_modules,
        plan_name="FSDP ignored",
    )
    configs = []
    for part in parts:
        local_modules = dict(part.named_modules())
        local_parameters = dict(part.named_parameters(remove_duplicate=False))
        local_options = {}
        for name, overrides in unit_options.items():
            if name not in local_modules:
                continue
            options = overrides.copy()
            hook = options.get("hook_module")
            if hook is not None:
                target = module_names.get(hook)
                if target is None or target not in local_modules:
                    raise ValueError(
                        f"FSDP unit {name!r}: hook_module must belong to its "
                        "pipeline part"
                    )
                options["hook_module"] = local_modules[target]
            exclusions = options.get("ignored_params")
            if exclusions is not None:
                options["ignored_params"] = {
                    local_parameters[path]
                    for parameter in exclusions
                    for path in parameter_names.get(parameter, ())
                    if path in local_parameters
                }
            local_options[name] = options
        configs.append(
            replace(
                cfg,
                fsdp_config=replace(
                    cfg.fsdp_config,
                    fsdp_apply_modules=local_options or {"": {}},
                    fsdp_ignored_modules=[
                        name for name in ignored_names if name in local_modules
                    ],
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
    for module_name, module in model.named_modules():
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


def apply_fsdp_by_pattern(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    cfg: ParallelConfig,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    edp_mesh: DeviceMesh | None = None,
) -> None:
    """Apply independently configured FSDP units selected by module patterns."""
    named_modules, unit_options, ignored_params = _resolve_pattern_units(model, cfg)
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
            if policy is None and cfg.fsdp_config.fsdp_implementation == "custom":
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
        exclusions = options.get("ignored_params")
        if exclusions is not None and (
            not isinstance(exclusions, set)
            or any(not isinstance(param, nn.Parameter) for param in exclusions)
        ):
            raise TypeError(
                f"FSDP unit {name!r}: ignored_params must be a set of Parameters "
                "or None"
            )
    # Validate every hook override before sharding any unit. A late rejection
    # would otherwise leave earlier units transformed on a failed assembly.
    for name, options in unit_options.items():
        if "hook_module" not in options:
            continue
        if cfg.fsdp_config.fsdp_implementation == "native":
            raise ValueError("native FSDP does not accept a hook_module override")
        validate_fsdp_hook_target([named_modules[name]], options["hook_module"])
    # The entry module imports this plan; defer runtime-policy imports.
    from .fully_shard_parallel import (
        build_fsdp_placement_fn,
        disable_fsdp_gradient_division,
        enable_fsdp_symm_mem,
        fsdp_shard_size,
        get_fsdp_reshard_after_forward_policy,
    )

    expert_placements: dict[nn.Parameter, Shard] = {}
    from ...models.common.moe.experts import GroupedExperts

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
    registration_order = {name: index for index, name in enumerate(named_modules)}
    prefetch_units = []
    for names, options in grouped_options:
        if cfg.ep_size == 1 and expert_counts:
            degree = fsdp_shard_size(options["mesh"])
            expert_placements = {
                parameter: Shard(1)
                for parameter, count in expert_counts.items()
                if degree > count
            }
        if (
            expert_placements or (cfg.ep_size == 1 and expert_counts)
        ) and "shard_placement_fn" not in options:
            if cfg.ep_size > 1:
                assert edp_mesh is not None
                # Same two-mesh-in-one-unit requirement as the decoder path;
                # refuse rather than let the placement function's import fail
                # obscurely. The pattern path has no nested-unit fallback: its
                # units are pattern-selected, not block-bounded.
                require(
                    "fsdp_per_param_mesh",
                    feature="expert parallelism combined with pattern-selected FSDP",
                )
                options["shard_placement_fn"] = build_fsdp_placement_fn(
                    expert_placements, dp_mesh=options["mesh"], edp_mesh=edp_mesh
                )
            else:
                options["shard_placement_fn"] = expert_placements.get
        modules = [named_modules[name] for name in names]
        shard_fn(modules[0] if len(modules) == 1 else modules, **options)
        if "" not in names:
            prefetch_units.append(
                (min(registration_order[n] for n in names), modules[0])
            )
    # Use stable module registration order, independent of pattern ordering.
    # Arbitrary model forward order cannot be inferred from named_modules().
    set_fsdp_prefetch_windows(
        [module for _, module in sorted(prefetch_units, key=lambda item: item[0])],
        forward_count=cfg.fsdp_config.fsdp_forward_prefetch_count,
        backward_count=cfg.fsdp_config.fsdp_backward_prefetch_count,
    )

    enable_fsdp_symm_mem(
        model,
        cfg.fsdp_config.fsdp_symm_mem_scope
        if cfg.fsdp_config.enable_fsdp_symm_mem
        else None,
    )
    disable_fsdp_gradient_division(model)


__all__ = ["apply_fsdp_by_pattern", "set_fsdp_prefetch_windows"]
