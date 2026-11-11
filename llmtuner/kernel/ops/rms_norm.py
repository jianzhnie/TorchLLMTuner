"""RMS normalization with device-dispatched implementations."""

from __future__ import annotations

import torch

from ..registry import dispatch_kernel


def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    use_eager: bool = False,
) -> torch.Tensor:
    """Normalize the final dimension using the selected backend."""
    return dispatch_kernel(
        "rms_norm",
        hidden_states,
        weight,
        epsilon,
        device_type="cpu" if use_eager else None,
    )
