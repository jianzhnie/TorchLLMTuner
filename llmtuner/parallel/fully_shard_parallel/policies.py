"""Shared FSDP placement, resharding, and communication policies.

Both decoder assembly and pattern-based assembly consume these policies;
this module must not import either assembly entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

try:
    from torch.distributed.fsdp._fully_shard._fsdp_common import (
        FSDPMeshInfo,
        ShardPlacementResult,
    )
    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
except ImportError as error:
    # Older PyTorch builds use nested expert units instead of mesh-aware placement.
    _placement_import_error = error
else:
    _placement_import_error = None


def iter_fsdp_modules(model: nn.Module) -> Iterator[FSDPModule]:
    """Yield every ``FSDPModule`` under ``model`` (``ReplicateModule`` included)."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            yield module


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Use summed gradients; Trainer scales loss by the global valid-token count.

    ReplicateModule inherits FSDPModule and follows the same policy.
    """
    for module in iter_fsdp_modules(model):
        module.set_gradient_divide_factor(1.0)


def fsdp_shard_size(dp_mesh: DeviceMesh) -> int:
    """How many ranks FSDP shards a parameter's dim 0 over, on ``dp_mesh``.

    FSDP cuts a parameter's dim 0 only over its shard axes: ``dp_shard``,
    plus ``cp`` when CP is on (``resolve_fsdp_mesh`` folds the two into
    ``dp_shard_cp``). ``dp_replicate`` replicates, so it must not inflate the
    degree compared against ``num_experts`` -- under HSDP (dp_replicate > 1)
    the raw mesh size would over-count and mis-pick ``Shard(1)`` for the
    expert weights.
    """
    degree = dp_mesh.size()
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        degree //= dp_mesh["dp_replicate"].size()
    return degree


def build_fsdp_placement_fn(
    expert_placements: dict[nn.Parameter, Shard],
    *,
    dp_mesh: DeviceMesh,
    edp_mesh: DeviceMesh,
) -> Callable[[nn.Parameter], Any]:
    """Keep EP-owned parameters on their sparse FSDP mesh inside any unit."""
    if _placement_import_error is not None:
        raise ImportError(
            "Mixed dense/expert FSDP placement requires compatible private FSDP APIs"
        ) from _placement_import_error

    edp_mesh_info = _get_mesh_info(edp_mesh, None)
    dp_mesh_info = _get_mesh_info(dp_mesh, None)
    assert isinstance(edp_mesh_info, FSDPMeshInfo)
    assert isinstance(dp_mesh_info, FSDPMeshInfo)

    def placement(param: nn.Parameter) -> ShardPlacementResult:
        expert_placement = expert_placements.get(param)
        if expert_placement is not None:
            return ShardPlacementResult(
                placement=expert_placement, mesh_info=edp_mesh_info
            )
        return ShardPlacementResult(placement=Shard(0), mesh_info=dp_mesh_info)

    return placement


def enable_fsdp_symm_mem(model: nn.Module, scope: str | None = "all") -> None:
    """Enable symmetric-memory communication for the FSDP modules ``scope`` selects.

    ``None`` disables it. ``"all"`` covers every FSDP module; ``"dense"``
    skips states owning any ``moe_enabled`` subtree, including grouped dense/MoE
    consumers and root-managed MoE children. Settings apply once per state, so
    attention parameters sharing an MoE state are skipped with its experts.
    Symmetric memory is not always beneficial for expert communication.
    """
    if scope is None:
        return
    if scope not in ("all", "dense"):
        raise ValueError(
            f"enable_fsdp_symm_mem scope must be one of 'all', 'dense', None; "
            f"got {scope!r}"
        )
    states: dict[int, FSDPModule] = {}
    sparse_states: set[int] = set()
    pending = [(model, None, False)]
    visited = set()
    while pending:
        module, owner, in_moe = pending.pop()
        if isinstance(module, FSDPModule):
            owner = id(module._get_fsdp_state())
            states.setdefault(owner, module)
        in_moe = in_moe or getattr(module, "moe_enabled", False)
        key = (id(module), owner, in_moe)
        if key in visited:
            continue
        visited.add(key)
        if in_moe and owner is not None:
            sparse_states.add(owner)
        pending.extend((child, owner, in_moe) for child in module.children())
    # Grouped modules share communication settings. Decide the complete scope
    # before configuring anything, including MoE children managed by a root state.
    for owner, module in states.items():
        if scope == "dense" and owner in sparse_states:
            continue
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm()


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )
