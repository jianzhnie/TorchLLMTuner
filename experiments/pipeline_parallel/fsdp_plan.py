"""Project FSDP pattern bindings onto pipeline-local model parts."""

from dataclasses import replace

from torch import nn

from experiments.pipeline_parallel.ownership import (
    PipelineOwnership,
    snapshot_pipeline_ownership,
)
from experiments.pipeline_parallel.parallel_config import ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fine_grained.fine_grained import (
    validate_fsdp_hook_target,
)
from llmtuner.parallel.fully_shard_parallel.plan import (
    _resolve_pattern_units,
    validate_fsdp_ignored_parameters,
)
from llmtuner.parallel.module_patterns import resolve_module_patterns
from llmtuner.parallel.plan import model_bindings_from_plan


def resolve_pipeline_fsdp_configs(
    model: nn.Module,
    parts: list[nn.Module],
    cfg: ParallelConfig,
    *,
    ownership: list[PipelineOwnership] | None = None,
) -> list[ParallelConfig]:
    """Bind a whole-model pattern plan to copied pipeline parts by registered FQN.

    Global patterns must match before selecting local units. Module and parameter
    overrides refer to the copied objects, while meshes and policies retain their
    identity. A part without a selected child still gets its root FSDP state.
    """
    bindings = (
        snapshot_pipeline_ownership(model, parts) if ownership is None else ownership
    )
    if len(bindings) != len(parts):
        raise ValueError("Pipeline ownership must describe every local part")
    validated_sources: set[int] = set()
    for part, binding in zip(parts, bindings, strict=True):
        # Parts normally share one source plan. Distinct supplied snapshots must
        # each be checked before borrowing any of their registered bindings.
        source_id = id(binding.global_plan)
        if source_id not in validated_sources:
            binding.global_plan.validate_binding(model)
            validated_sources.add(source_id)
        binding.local_plan.validate_binding(part)

    module_bindings = (
        model_bindings_from_plan(bindings[0].global_plan)["module_bindings"]
        if bindings
        else tuple(model.named_modules(remove_duplicate=False))
    )
    source_modules, unit_options, _ = _resolve_pattern_units(module_bindings, cfg)
    source_parameters = {
        parameter
        for module in source_modules.values()
        for parameter in module.parameters(recurse=False)
    }
    for name, options in unit_options.items():
        validate_fsdp_hook_target([source_modules[name]], options.get("hook_module"))
        validate_fsdp_ignored_parameters(
            options.get("ignored_params"), parameters=source_parameters, unit_name=name
        )
    source_paths = dict(module_bindings)
    ignored_names = resolve_module_patterns(
        list(source_paths),
        patterns=cfg.fsdp_config.fsdp_ignored_modules,
        plan_name="FSDP ignored",
    )

    configs = []
    for binding in bindings:
        local_options = {}
        for name, overrides in unit_options.items():
            unit_binding = binding.module_binding(source_modules[name])
            if unit_binding is None:
                continue
            local_name, _ = unit_binding
            options = overrides.copy()
            hook = options.get("hook_module")
            if hook is not None:
                hook_binding = binding.module_binding(hook)
                if hook_binding is None:
                    raise ValueError(
                        f"FSDP unit {name!r}: hook_module must belong to its "
                        "pipeline part"
                    )
                options["hook_module"] = hook_binding[1]
            exclusions = options.get("ignored_params")
            if exclusions is not None:
                options["ignored_params"] = binding.parameter_bindings(exclusions)
            local_options.setdefault(local_name, {}).update(options)
        local_ignored = [
            ignored_binding[0]
            for name in ignored_names
            if (ignored_binding := binding.module_binding(source_paths[name]))
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
