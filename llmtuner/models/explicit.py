"""Compatibility API for opt-in construction from the single model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .registry import build_registered_model

if TYPE_CHECKING:
    from torch import nn


def build_explicit_model(config: Any) -> nn.Module:
    """Preserve opt-in construction while sharing the production constructor."""
    return build_registered_model(config, explicit_only=True)


__all__ = ["build_explicit_model"]
