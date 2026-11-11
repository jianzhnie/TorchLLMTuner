"""Shared input contracts for reference and device kernels.

Validation inspects metadata only, except packed offsets, which must be host
integers for vendor attention APIs. Converting device offsets synchronizes them;
callers can pass host lists to avoid that transfer on repeated invocations.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import torch

SequenceOffsets: TypeAlias = torch.Tensor | list[int] | None


def sequence_offsets(values: SequenceOffsets, size: int) -> list[int]:
    """Normalize leading-zero cumulative offsets; allow empty documents."""
    if values is None:
        return [0, size]
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError("Sequence offsets must be a 1-D int32/int64 tensor")
        result = values.tolist()
    elif isinstance(values, list):
        result = list(values)
    else:
        raise TypeError("Sequence offsets must be a list, tensor, or None")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ValueError("Sequence offsets must contain integers")
    if len(result) < 2 or result[0] != 0 or result[-1] != size:
        raise ValueError("Sequence offsets must start at 0 and end at the token count")
    if any(left > right for left, right in zip(result[:-1], result[1:], strict=True)):
        raise ValueError("Sequence offsets must be nondecreasing")
    return result


def require_same_device(*tensors: torch.Tensor) -> None:
    """Reject implicit movement between devices, including different NPU ranks."""
    if len({tensor.device for tensor in tensors}) > 1:
        raise ValueError("Kernel inputs must be on the same device")


def attention_offsets(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    q_cu_seqlens: SequenceOffsets,
    k_cu_seqlens: SequenceOffsets,
) -> tuple[list[int], list[int]]:
    """Validate non-causal TND multi-head attention and document boundaries."""
    if any(tensor.ndim != 3 for tensor in (query, key, value)):
        raise ValueError("Packed attention expects TND tensors")
    require_same_device(query, key, value)
    if any(not tensor.is_floating_point() for tensor in (query, key, value)):
        raise ValueError("Attention inputs must be floating-point tensors")
    if not query.dtype == key.dtype == value.dtype:
        raise ValueError("Attention inputs must have the same dtype")
    if isinstance(head_num, bool) or not isinstance(head_num, int) or head_num < 1:
        raise ValueError("head_num must be a positive integer")
    if any(tensor.shape[1] != head_num for tensor in (query, key, value)):
        raise ValueError("Packed attention head counts are inconsistent")
    if key.shape != value.shape or query.shape[2] != key.shape[2]:
        raise ValueError("Packed attention Q/K/V dimensions are inconsistent")
    if query.shape[2] == 0 or not math.isfinite(scale):
        raise ValueError("Attention requires a nonzero head dimension and finite scale")
    q_offsets = sequence_offsets(q_cu_seqlens, query.shape[0])
    k_offsets = sequence_offsets(k_cu_seqlens, key.shape[0])
    if len(q_offsets) != len(k_offsets):
        raise ValueError("Packed attention document counts are inconsistent")
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True
    ):
        if q_start != q_end and k_start == k_end:
            raise ValueError("A nonempty query document requires nonempty keys")
    return q_offsets, k_offsets


def rotary_inputs(
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    rotary_mode: str | None,
) -> None:
    """Require an even rotary width and broadcasting that preserves input shape."""
    require_same_device(hidden_states, cosine, sine)
    if rotary_mode not in (None, "half", "interleave"):
        raise ValueError(f"Unsupported rotary mode {rotary_mode!r}")
    if hidden_states.ndim == 0 or hidden_states.shape[-1] % 2:
        raise ValueError("RoPE requires an even final dimension")
    for factor in (cosine, sine):
        if (
            torch.broadcast_shapes(hidden_states.shape, factor.shape)
            != hidden_states.shape
        ):
            raise ValueError("RoPE factors must broadcast to the input shape")


def rms_norm_inputs(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    """Validate final-axis normalization without imposing a compute dtype policy."""
    require_same_device(hidden_states, weight)
    if weight.ndim != 1 or hidden_states.ndim == 0:
        raise ValueError("RMSNorm requires a 1-D weight and a non-scalar input")
    if hidden_states.shape[-1] != weight.shape[0] or weight.numel() == 0:
        raise ValueError("RMSNorm weight must match the nonzero final input dimension")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("RMSNorm epsilon must be finite and nonnegative")


def moe_permutation_inputs(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor
) -> None:
    """Validate token-major data and a nonempty top-k dimension, metadata only."""
    require_same_device(hidden_states, expert_ids)
    if hidden_states.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("MoE permutation expects [tokens, hidden] and [tokens, top_k]")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoE expert IDs must use int32 or int64")
    if expert_ids.shape[0] != hidden_states.shape[0] or expert_ids.shape[1] == 0:
        raise ValueError("MoE token counts must match and top_k must be positive")
