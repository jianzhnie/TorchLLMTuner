"""MoE token movement kernels."""

from __future__ import annotations

import torch

from ..registry import dispatch_kernel


def moe_token_permute(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    use_eager: bool = False,
) -> torch.Tensor:
    """Gather [tokens, hidden] rows in stable expert-major assignment order.

    expert_ids is an int32/int64 [tokens, top_k] tensor on the same device.
    Return [tokens * top_k, hidden], repeating rows for multiple assignments.
    Gradients accumulate into the original token rows. use_eager selects torch
    math on the existing device; no tensor transfer or fallback is implicit.
    """
    return dispatch_kernel(
        "moe_token_permute",
        hidden_states,
        expert_ids,
        device_type="cpu" if use_eager else None,
    )
