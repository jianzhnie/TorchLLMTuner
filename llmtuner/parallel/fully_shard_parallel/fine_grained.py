"""PyTorch 2.14 fine-grained FSDP initialization and strategy selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from llmtuner.errors import EnvironmentUnsupportedError

if TYPE_CHECKING:
    from torch.distributed.fsdp._fully_shard._fsdp_api import DataParallelMeshDims
    from torch.distributed.fsdp._fully_shard._fsdp_common import ShardPlacementFnResult


def validate_fsdp_hook_target(
    modules: list[nn.Module], hook_module: nn.Module | None
) -> None:
    """Require relocated hooks to enclose every managed module's execution."""
    if hook_module is None:
        return
    if not isinstance(hook_module, nn.Module):
        raise TypeError("hook_module must be a torch.nn.Module or None")
    descendants = set(hook_module.modules())
    if any(module not in descendants for module in modules):
        raise ValueError("hook_module must contain all managed FSDP modules")
    if isinstance(hook_module, nn.ModuleList | nn.ModuleDict):
        raise ValueError("hook_module must implement forward, not be a container")


def fully_shard_fine_grained(
    module: nn.Module | list[nn.Module],
    *,
    hook_module: nn.Module | None = None,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
) -> nn.Module | list[nn.Module]:
    """Load the PyTorch 2.14 custom engine when this strategy is applied.

    Apply bottom-up, including the root. A relocated hook target must enclose
    every managed module. Native strategy selection does not load this engine.
    """
    validate_fsdp_hook_target(
        [module] if isinstance(module, nn.Module) else module, hook_module
    )
    try:
        from .fine_grained_state import apply_fine_grained_fsdp
    except ImportError as error:
        raise EnvironmentUnsupportedError(
            "custom FSDP requires PyTorch 2.14 compatible private FSDP APIs; "
            "install a compatible PyTorch build or select fsdp_implementation='native'"
        ) from error
    return apply_fine_grained_fsdp(
        module,
        hook_module=hook_module,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=ignored_params,
        dp_mesh_dims=dp_mesh_dims,
    )


class NativeFSDPStrategy:
    """Strategy facade matching FSDPTurbo's strategy selection contract."""

    @staticmethod
    def get_fully_shard_fn() -> Callable[..., nn.Module | list[nn.Module]]:
        return fully_shard


class CustomFSDPStrategy(NativeFSDPStrategy):
    """Fine-grained hook/context strategy using PyTorch 2.14 collectives."""

    @staticmethod
    def get_fully_shard_fn() -> Callable[..., nn.Module | list[nn.Module]]:
        return fully_shard_fine_grained


def get_fsdp_strategy(implementation: str) -> NativeFSDPStrategy:
    if implementation == "native":
        return NativeFSDPStrategy()
    if implementation == "custom":
        return CustomFSDPStrategy()
    raise ValueError(f"unknown FSDP implementation {implementation!r}")


__all__ = [
    "CustomFSDPStrategy",
    "NativeFSDPStrategy",
    "fully_shard_fine_grained",
    "get_fsdp_strategy",
]
