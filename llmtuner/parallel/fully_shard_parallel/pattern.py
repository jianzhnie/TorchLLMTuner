"""Execute prevalidated pattern-selected FSDP units and configure prefetch.

Parsing, shared ownership and capability checks live in plan.py. Only this
module wraps the resolved units and installs runtime FSDP policies.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

from ...accelerator.collectives import synchronize_local_errors
from ...config.parallel_config import ParallelConfig
from .plan import FSDPPatternPlan, resolve_fsdp_pattern_plan
from .policies import (
    build_fsdp_placement_fn,
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    fsdp_shard_size,
)

__all__ = [
    "apply_fsdp_by_pattern",
    "execute_fsdp_pattern_plan",
    "set_fsdp_prefetch_windows",
]


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


def apply_fsdp_by_pattern(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    cfg: ParallelConfig,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    edp_mesh: DeviceMesh | None = None,
    error_group: dist.ProcessGroup | None = None,
) -> None:
    """Apply independently configured FSDP units selected by module patterns.

    An explicit error_group coordinates local selection/override failures before
    sharding. Every member must call in the same order; assembly supplies the
    stage-local loss group, including dense and expert communication peers.
    Standalone callers without a group retain local-only validation.
    """
    with (
        synchronize_local_errors("FSDP pattern validation", group=error_group)
        if error_group is not None
        else nullcontext()
    ):
        resolved = resolve_fsdp_pattern_plan(
            model,
            dp_mesh=dp_mesh,
            cfg=cfg,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            edp_mesh=edp_mesh,
        )
    execute_fsdp_pattern_plan(model, resolved, error_group=error_group)


def execute_fsdp_pattern_plan(
    model: nn.Module,
    resolved: FSDPPatternPlan,
    *,
    error_group: dist.ProcessGroup | None = None,
) -> None:
    """Consume the exact resolved bindings without rereading patterns or config.

    Every participant validates before any wrapping begins. Replaced modules,
    changed parameter geometry or forward bindings invalidate the plan; runtime
    collective failures still require checkpoint recovery, not local rollback.
    """
    with (
        synchronize_local_errors("FSDP pattern binding validation", group=error_group)
        if error_group is not None
        else nullcontext()
    ):
        resolved.bindings.validate_binding(model)
    named_modules = resolved.named_modules
    expert_placements = dict(resolved.expert_placements)
    expert_counts = resolved.expert_counts
    shard_fn = resolved.shard_fn
    registration_order = {name: index for index, name in enumerate(named_modules)}
    prefetch_units = []
    for names, planned_options in resolved.units:
        options = planned_options.copy()
        if resolved.ep_degree == 1 and expert_counts:
            degree = fsdp_shard_size(options["mesh"])
            expert_placements = {
                parameter: Shard(1)
                for parameter, count in expert_counts.items()
                if degree > count
            }
        if (
            expert_placements or (resolved.ep_degree == 1 and expert_counts)
        ) and options.get("shard_placement_fn") is None:
            if resolved.ep_degree > 1:
                assert resolved.edp_mesh is not None
                options["shard_placement_fn"] = build_fsdp_placement_fn(
                    expert_placements,
                    dp_mesh=options["mesh"],
                    edp_mesh=resolved.edp_mesh,
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
        forward_count=resolved.prefetch[0],
        backward_count=resolved.prefetch[1],
    )

    enable_fsdp_symm_mem(model, resolved.symm_mem_scope)
    disable_fsdp_gradient_division(model)
