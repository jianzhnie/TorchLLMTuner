"""Inspect registered model state without executing checkpoint hooks."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor, nn


def named_persistent_buffers(model: nn.Module) -> Iterator[tuple[str, Tensor]]:
    """Preserve every registered alias while omitting runtime-only buffers."""
    for path, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module._buffers.items():
            if buffer is not None and name not in module._non_persistent_buffers_set:
                yield f"{path}.{name}" if path else name, buffer


def named_extra_state_owners(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Name extra-state providers without calling their getters or save hooks."""
    for path, module in model.named_modules(remove_duplicate=False):
        if type(module).get_extra_state is not nn.Module.get_extra_state:
            yield f"{path}._extra_state" if path else "_extra_state", module


def registered_state_paths(model: nn.Module) -> tuple[str, ...]:
    """State registrations that require ownership before pipeline partitioning."""
    return tuple(
        name
        for registrations in (
            model.named_parameters(remove_duplicate=False),
            named_persistent_buffers(model),
            named_extra_state_owners(model),
        )
        for name, _ in registrations
    )
