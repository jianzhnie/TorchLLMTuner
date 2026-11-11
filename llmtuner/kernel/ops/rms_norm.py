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
    """Normalize the nonempty final axis using a matching 1-D weight.

    epsilon must be finite and nonnegative. Input and weight share a device;
    dtype conversion remains the caller's responsibility. use_eager explicitly
    selects reference torch math on the existing device, without data transfer.
    """
    return dispatch_kernel(
        "rms_norm",
        hidden_states,
        weight,
        epsilon,
        device_type="cpu" if use_eager else None,
    )
