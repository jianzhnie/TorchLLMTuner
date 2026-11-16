"""Attention masks and temporary mask policy for explicit decoder models."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

__all__ = ["build_dense_attention_mask", "temporary_attention_mask"]


def build_dense_attention_mask(
    positions: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_causal: bool,
) -> torch.Tensor:
    """Build an additive mask [B, 1, T, T], with zero for allowed keys.

    Positions have shape [T] (one row) or [B, T] (independent rows).
    Positions reset to zero at packed document boundaries. Block-causal masks
    restrict attention to the same document; otherwise only causality applies.
    """
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    elif positions.ndim != 2:
        raise ValueError("positions must have shape [T] or [B, T]")
    num_tokens = positions.shape[-1]
    idx = torch.arange(num_tokens, device=positions.device)
    allowed = idx[:, None] >= idx[None, :]
    if block_causal:
        doc_ids = torch.cumsum((positions == 0).int(), dim=-1)
        allowed = allowed & (doc_ids[..., :, None] == doc_ids[..., None, :])
    mask = torch.zeros(
        (positions.shape[0], num_tokens, num_tokens),
        device=positions.device,
        dtype=dtype,
    )
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(1)


@contextmanager
def temporary_attention_mask(config: Any, *, mask_type: str | None) -> Iterator[None]:
    """Override the model-selected config and restore even an absent attribute.

    The model selects the configuration. Nested contexts restore shared
    configuration objects in reverse entry order.
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
