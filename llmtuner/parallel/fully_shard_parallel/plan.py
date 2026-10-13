"""Pattern selection and explicit prefetch plans for FSDP units."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

from llmtuner.config import ParallelConfig

from ..module_patterns import module_name_matches
from .fine_grained import get_fsdp_strategy


def set_fsdp_prefetch_windows(
    modules: list[nn.Module], *, forward_count: int, backward_count: int
) -> None:
    """Configure exact forward and backward prefetch windows."""
    if forward_count < 0 or backward_count < 0:
        raise ValueError("FSDP prefetch counts must be nonnegative")
    for index, module in enumerate(modules):
        forward = modules[index + 1 : index + 1 + forward_count]
        module.set_modules_to_forward_prefetch(forward)  # type: ignore[attr-defined]
        backward = list(reversed(modules[max(0, index - backward_count) : index]))
        module.set_modules_to_backward_prefetch(backward)  # type: ignore[attr-defined]


def _resolve_pattern_units(
    model: nn.Module, cfg: ParallelConfig
) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]], set[nn.Parameter]]:
    named = dict(model.named_modules())
    ignored_roots = {
        name
        for name in named
        if any(
            module_name_matches(pattern, name) for pattern in cfg.fsdp_ignored_modules
        )
    }
    for pattern in cfg.fsdp_ignored_modules:
        if not any(module_name_matches(pattern, name) for name in named):
            raise ValueError(f"FSDP ignored pattern {pattern!r} matched no module")

    # Ignoring a module excludes its complete subtree, matching the recursive
    # ignored-parameter collection below.
    ignored_names = {
        name
        for name in named
        if any(
            root == "" or name == root or name.startswith(f"{root}.")
            for root in ignored_roots
        )
    }

    selected: dict[str, dict[str, Any]] = {}
    for pattern, overrides in cfg.fsdp_apply_modules.items():
        matches = [
            name
            for name in named
            if name and name not in ignored_names and module_name_matches(pattern, name)
        ]
        if not matches:
            raise ValueError(f"FSDP apply pattern {pattern!r} matched no module")
        for name in matches:
            selected.setdefault(name, {}).update(overrides)
    if not selected:
        raise ValueError("fsdp_apply_modules selected no FSDP unit")

    ignored_params = {
        parameter
        for name in ignored_names
        for parameter in named[name].parameters(recurse=True)
    }
    return named, selected, ignored_params


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
    named, selected, ignored_params = _resolve_pattern_units(model, cfg)
    # The entry module imports this plan; defer runtime-policy imports.
    from .fully_shard_parallel import (
        build_fsdp_placement_fn,
        disable_fsdp_gradient_division,
        enable_fsdp_symm_mem,
        fsdp_shard_size,
        get_fsdp_reshard_after_forward_policy,
    )

    expert_placements: dict[nn.Parameter, Shard] = {}
    if cfg.ep > 1:
        if edp_mesh is None:
            raise ValueError("pattern FSDP with EP requires a sparse FSDP mesh")
        from ...models.common.moe.experts import GroupedExperts

        degree = fsdp_shard_size(edp_mesh)
        expert_placements = {
            parameter: Shard(1 if degree > module.num_experts else 0)
            for module in model.modules()
            if isinstance(module, GroupedExperts)
            for parameter in module.parameters(recurse=False)
            if parameter not in ignored_params
        }

    shard_fn = get_fsdp_strategy(cfg.fsdp_implementation).get_fully_shard_fn()
    common: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
        "ignored_params": ignored_params,
        "reshard_after_forward": get_fsdp_reshard_after_forward_policy(
            cfg.fsdp_reshard_after_forward, cfg.pp > 1
        ),
    }
    if cfg.fsdp_cpu_offload:
        common["offload_policy"] = CPUOffloadPolicy(
            pin_memory=cfg.fsdp_cpu_offload_pin_memory
        )

    unit_names = sorted(selected, key=lambda item: item.count("."), reverse=True)
    for name in [*unit_names, ""]:
        options = common | selected.get(name, {})
        if expert_placements and "shard_placement_fn" not in options:
            assert edp_mesh is not None
            options["shard_placement_fn"] = build_fsdp_placement_fn(
                expert_placements, dp_mesh=options["mesh"], edp_mesh=edp_mesh
            )
        shard_fn(named[name], **options)
    set_fsdp_prefetch_windows(
        [module for name, module in named.items() if name in selected],
        forward_count=cfg.fsdp_forward_prefetch_count,
        backward_count=cfg.fsdp_backward_prefetch_count,
    )

    enable_fsdp_symm_mem(
        model, cfg.fsdp_symm_mem_scope if cfg.enable_fsdp_symm_mem else None
    )
    disable_fsdp_gradient_division(model)


__all__ = ["apply_fsdp_by_pattern", "set_fsdp_prefetch_windows"]
