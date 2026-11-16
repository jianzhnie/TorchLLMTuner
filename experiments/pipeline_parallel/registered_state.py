"""Collect registered state paths for experimental pipeline ownership."""

from torch import nn

from llmtuner.models.registered_state import (
    named_extra_state_owners,
    named_persistent_buffers,
)


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
