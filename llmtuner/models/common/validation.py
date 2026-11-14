"""Temporary text-attention policy shared by explicit decoder families."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def temporary_attention_mask(config: Any, *, mask_type: str | None) -> Iterator[None]:
    """Override the model-selected config and restore even an absent attribute.

    The model owns selection of this config. Nested contexts also support PP
    parts borrowing the same configuration object, restoring in reverse order.
    """
    had_mask = hasattr(config, "attn_mask_type")
    original = getattr(config, "attn_mask_type", None)
    try:
        if mask_type is not None:
            config.attn_mask_type = mask_type
        yield
    finally:
        if had_mask:
            config.attn_mask_type = original
        elif hasattr(config, "attn_mask_type"):
            delattr(config, "attn_mask_type")
