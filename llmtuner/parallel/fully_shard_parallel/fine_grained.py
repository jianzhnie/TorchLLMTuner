"""PyTorch 2.14 compatible fine-grained FSDP adapter.

FSDPTurbo's original implementation subclasses private FSDP state classes. Those
classes changed between the PyTorch releases supported by this project. This
adapter keeps the public fine-grained contract while delegating parameter and
communication state to the installed PyTorch 2.14 implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch.distributed.fsdp import fully_shard


def fully_shard_fine_grained(
    module: nn.Module,
    *,
    hook_module: nn.Module | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Apply one independent FSDP unit using the PyTorch 2.14 state engine.

    ``hook_module`` is retained for FSDPTurbo plan compatibility. Native
    PyTorch 2.14 registers hooks on the managed module, so a distinct hook
    target is rejected instead of silently changing overlap semantics.
    """
    if hook_module is not None and hook_module is not module:
        raise ValueError(
            "PyTorch 2.14 native FSDP cannot attach a unit to a separate "
            "hook_module; select the managed module as the hook target"
        )
    return fully_shard(module, **kwargs)


class NativeFSDPStrategy:
    """Strategy facade matching FSDPTurbo's strategy selection contract."""

    supports_hook_module = False

    @staticmethod
    def get_fully_shard_fn() -> Callable[..., nn.Module]:
        return fully_shard


class CustomFSDPStrategy(NativeFSDPStrategy):
    """Fine-grained plan strategy backed by PyTorch 2.14 public FSDP2."""

    supports_hook_module = True

    @staticmethod
    def get_fully_shard_fn() -> Callable[..., nn.Module]:
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
