"""Pattern selection and explicit prefetch plans for FSDP units."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

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
        if forward:
            module.set_modules_to_forward_prefetch(forward)  # type: ignore[attr-defined]
        backward = list(reversed(modules[max(0, index - backward_count) : index]))
        if backward:
            module.set_modules_to_backward_prefetch(backward)  # type: ignore[attr-defined]


def _resolve_pattern_units(
    model: nn.Module, cfg: ParallelConfig
) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]], set[nn.Parameter]]:
    named = dict(model.named_modules())
    ignored_names = {
        name
        for name in named
        if any(
            module_name_matches(pattern, name) for pattern in cfg.fsdp_ignored_modules
        )
    }
    for pattern in cfg.fsdp_ignored_modules:
        if not any(module_name_matches(pattern, name) for name in named):
            raise ValueError(f"FSDP ignored pattern {pattern!r} matched no module")

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
) -> None:
    """Apply independently configured FSDP units selected by module patterns."""
    named, selected, ignored_params = _resolve_pattern_units(model, cfg)
    shard_fn = get_fsdp_strategy(cfg.fsdp_implementation).get_fully_shard_fn()
    common: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
        "ignored_params": ignored_params,
    }
    if cfg.fsdp_cpu_offload:
        common["offload_policy"] = CPUOffloadPolicy(
            pin_memory=cfg.fsdp_cpu_offload_pin_memory
        )

    for name in sorted(selected, key=lambda item: item.count("."), reverse=True):
        shard_fn(named[name], **(common | selected[name]))
    shard_fn(model, **common)
    set_fsdp_prefetch_windows(
        [module for name, module in named.items() if name in selected],
        forward_count=cfg.fsdp_forward_prefetch_count,
        backward_count=cfg.fsdp_backward_prefetch_count,
    )

    # Local import avoids a module cycle: the entry module owns FSDP runtime
    # policies, while this module owns plan resolution.
    from .fully_shard_parallel import disable_fsdp_gradient_division

    disable_fsdp_gradient_division(model)


__all__ = ["apply_fsdp_by_pattern", "set_fsdp_prefetch_windows"]
