"""Small Ascend operator bindings used by model-owned numerical policies."""

from __future__ import annotations

import torch

try:
    import torch_npu
except ImportError as error:
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = error

__all__ = ["fusion_attention", "moe_token_permute", "rms_norm", "rotary_mul"]


def _require_npu(*tensors: torch.Tensor) -> None:
    if torch_npu is None:
        raise ImportError(
            "NPU kernels require a working torch-npu installation"
        ) from TORCH_NPU_IMPORT_ERROR
    if any(tensor.device.type != "npu" for tensor in tensors):
        raise RuntimeError("Ascend operator requires NPU tensors")


def rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    _require_npu(hidden_states, weight)
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


def rotary_mul(
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    rotary_mode: str | None = None,
) -> torch.Tensor:
    _require_npu(hidden_states, cosine, sine)
    options = {} if rotary_mode is None else {"rotary_mode": rotary_mode}
    return torch_npu.npu_rotary_mul(hidden_states, cosine, sine, **options)


def fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    q_cu_seqlens: torch.Tensor | list[int] | None,
    k_cu_seqlens: torch.Tensor | list[int] | None,
) -> torch.Tensor:
    _require_npu(query, key, value)
    q_lengths = (
        q_cu_seqlens.tolist()
        if isinstance(q_cu_seqlens, torch.Tensor)
        else q_cu_seqlens
    )
    k_lengths = (
        k_cu_seqlens.tolist()
        if isinstance(k_cu_seqlens, torch.Tensor)
        else k_cu_seqlens
    )
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
        actual_seq_qlen=q_lengths,
        actual_seq_kvlen=k_lengths,
    )[0]


def moe_token_permute(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    _require_npu(hidden_states, expert_ids)
    return torch_npu.npu_moe_token_permute(hidden_states, expert_ids)[0]
