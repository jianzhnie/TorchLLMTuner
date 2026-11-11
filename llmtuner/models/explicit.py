"""Opt-in model construction from the single architecture registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .architecture import registered_model_architectures

if TYPE_CHECKING:
    from torch import nn


def build_explicit_model(config: Any) -> nn.Module:
    """Construct an explicitly enabled architecture without HF auto resolution.

    Task selection belongs to the registration: composite models keep their
    conditional-generation class rather than being unwrapped as text models.
    """
    architectures = {
        entry.model_type: entry
        for entry in registered_model_architectures()
        if entry.explicit
    }
    architecture = architectures.get(config.model_type)
    if architecture is None:
        supported = ", ".join(sorted(architectures))
        raise ValueError(
            f"No explicit model for {config.model_type!r}; choose one of {supported}"
        )
    return architecture.model_class_loader()(config)


__all__ = ["build_explicit_model"]
