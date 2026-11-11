"""Ascend implementations isolated from model and dispatch code."""

from __future__ import annotations

import torch

try:
    import torch_npu
except ImportError as error:
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR: ImportError | None = error
else:
    TORCH_NPU_IMPORT_ERROR = None

from ..registry import register_kernel
from ..validation import (
    SequenceOffsets,
    attention_offsets,
    moe_permutation_inputs,
    require_same_device,
    rms_norm_inputs,
    rotary_inputs,
)


def _require_npu(*tensors: torch.Tensor) -> None:
    """Keep optional dependency and device failures explicit, without fallback."""
    if torch_npu is None:
        raise ImportError(
            "Ascend kernels require torch-npu"
        ) from TORCH_NPU_IMPORT_ERROR
    require_same_device(*tensors)
    if any(tensor.device.type != "npu" for tensor in tensors):
        raise RuntimeError("Ascend kernel requires NPU tensors")


@register_kernel("rms_norm", "npu")
def rms_norm_npu(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Return CANN RMSNorm output; discard its internal reciprocal RMS result."""
    _require_npu(hidden_states, weight)
    rms_norm_inputs(hidden_states, weight, epsilon)
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


@register_kernel("rotary_mul", "npu")
def rotary_mul_npu(
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    rotary_mode: str | None = None,
) -> torch.Tensor:
    """Use CANN RoPE with an explicit, validated coordinate layout."""
    _require_npu(hidden_states, cosine, sine)
    rotary_inputs(hidden_states, cosine, sine, rotary_mode)
    options = {} if rotary_mode is None else {"rotary_mode": rotary_mode}
    return torch_npu.npu_rotary_mul(hidden_states, cosine, sine, **options)


@register_kernel("moe_token_permute", "npu")
def moe_token_permute_npu(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    """Return stable expert-major tokens; permutation metadata stays internal."""
    _require_npu(hidden_states, expert_ids)
    moe_permutation_inputs(hidden_states, expert_ids)
    return torch_npu.npu_moe_token_permute(hidden_states, expert_ids)[0]


@register_kernel("fusion_attention", "npu")
def fusion_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    q_cu_seqlens: SequenceOffsets,
    k_cu_seqlens: SequenceOffsets,
) -> torch.Tensor:
    """Adapt leading-zero packed offsets to CANN document end positions."""
    _require_npu(query, key, value)
    q_offsets, k_offsets = attention_offsets(
        query,
        key,
        value,
        head_num=head_num,
        scale=scale,
        q_cu_seqlens=q_cu_seqlens,
        k_cu_seqlens=k_cu_seqlens,
    )
    # CANN TND expects document end positions, without FlashAttention's
    # leading zero. Input tensors remain on their original device.
    return torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=head_num,
        pse=None,
        atten_mask=None,
        scale=scale,
        keep_prob=1,
        input_layout="TND",
        actual_seq_qlen=q_offsets[1:],
        actual_seq_kvlen=k_offsets[1:],
    )[0]
