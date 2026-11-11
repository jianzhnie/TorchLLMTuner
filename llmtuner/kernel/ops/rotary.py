"""Rotary position embedding kernels for half-split and interleaved layouts."""

from __future__ import annotations

import torch

from ..registry import dispatch_kernel


def rotary_mul(
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    rotary_mode: str | None = None,
    use_eager: bool = False,
) -> torch.Tensor:
    """Apply half-split (default) or interleaved RoPE along an even final axis.

    Factors must broadcast to the input shape on the same device. Dtype
    conversion remains the caller's policy. use_eager selects reference torch
    math without moving tensors; otherwise dispatch follows hidden_states.
    """
    return dispatch_kernel(
        "rotary_mul",
        hidden_states,
        cosine,
        sine,
        rotary_mode=rotary_mode,
        device_type="cpu" if use_eager else None,
    )
