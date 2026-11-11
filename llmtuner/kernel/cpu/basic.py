"""Reference CPU implementations for the public kernel contracts."""

from __future__ import annotations

import torch

from ..registry import register_kernel
from ..validation import (
    SequenceOffsets,
    attention_offsets,
    moe_permutation_inputs,
    rms_norm_inputs,
    rotary_inputs,
)


@register_kernel("rms_norm", "cpu")
def rms_norm_cpu(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Use PyTorch final-axis RMSNorm without changing the caller's dtype policy."""
    rms_norm_inputs(hidden_states, weight, epsilon)
    return torch.nn.functional.rms_norm(
        hidden_states, (weight.numel(),), weight, epsilon
    )


@register_kernel("rotary_mul", "cpu")
def rotary_mul_cpu(
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    rotary_mode: str | None = None,
) -> torch.Tensor:
    """Apply either rotary layout with independent per-coordinate factors."""
    rotary_inputs(hidden_states, cosine, sine, rotary_mode)
    if rotary_mode in (None, "half"):
        first, second = hidden_states.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    else:
        even = hidden_states[..., 0::2]
        odd = hidden_states[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return hidden_states * cosine + rotated * sine


@register_kernel("moe_token_permute", "cpu")
def moe_token_permute_cpu(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    """Gather token assignments sorted stably by expert, retaining autograd."""
    moe_permutation_inputs(hidden_states, expert_ids)
    top_k = expert_ids.shape[1]
    assignments = torch.argsort(expert_ids.reshape(-1), stable=True)
    return hidden_states[assignments // top_k]


@register_kernel("fusion_attention", "cpu")
def fusion_attention_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    q_cu_seqlens: SequenceOffsets,
    k_cu_seqlens: SequenceOffsets,
) -> torch.Tensor:
    """Compute each document independently, using FP32 softmax as before.

    Avoid a global Tq-by-Tk mask and cross-document score matrix. Empty query
    documents are supported and keep zero-gradient links to all three inputs.
    """
    q_offsets, k_offsets = attention_offsets(
        query,
        key,
        value,
        head_num=head_num,
        scale=scale,
        q_cu_seqlens=q_cu_seqlens,
        k_cu_seqlens=k_cu_seqlens,
    )
    outputs = []
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True
    ):
        q = query[q_start:q_end].transpose(0, 1)
        k = key[k_start:k_end].transpose(0, 1)
        v = value[k_start:k_end].transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * scale
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        outputs.append((probabilities @ v).transpose(0, 1))
    return torch.cat(outputs, dim=0)
