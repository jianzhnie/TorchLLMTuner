"""Accelerator-only non-causal variable-length attention selection.

Inputs are packed [tokens, heads, head_dim]. The caller owns projection and
output reshaping. This API never moves tensors or selects a CPU fallback.
"""

import torch

from ...accelerator.device import is_device_type_available
from ...errors import EnvironmentUnsupportedError, UnsupportedCombinationError
from ..validation import require_same_device
from .attention import fusion_attention

FLASH_ATTN_IMPORT_ERROR: ImportError | None = None
flash_attn_varlen_func = None
if is_device_type_available("cuda"):
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as error:
        FLASH_ATTN_IMPORT_ERROR = error


def varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None,
    k_cu_seqlens: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    """Run Ascend fusion attention or CUDA FlashAttention with no dropout.

    CUDA keeps the vendor's offset/maximum-length contract. Ascend delegates
    geometry and offset validation to the existing fusion_attention API.
    Its deterministic mode is not established, so an explicit request is rejected.
    """
    if query.device.type == "npu":
        if deterministic:
            raise UnsupportedCombinationError(
                "Ascend variable-length attention has no validated deterministic "
                "mode; deterministic=True is unsupported"
            )
        if any(tensor.ndim != 3 for tensor in (query, key, value)):
            raise ValueError("Packed attention expects TND tensors")
        if query.shape[-1] == 0:
            raise ValueError("Attention requires a nonzero head dimension")
        output = fusion_attention(
            query,
            key,
            value,
            head_num=query.shape[1],
            scale=query.shape[-1] ** -0.5,
            q_cu_seqlens=q_cu_seqlens,
            k_cu_seqlens=k_cu_seqlens,
        )
    elif query.device.type == "cuda":
        if flash_attn_varlen_func is None:
            raise EnvironmentUnsupportedError(
                "CUDA variable-length attention requires flash_attn"
            ) from FLASH_ATTN_IMPORT_ERROR
        if any(tensor.ndim != 3 for tensor in (query, key, value)):
            raise ValueError("Packed attention expects TND tensors")
        require_same_device(query, key, value)
        if not query.dtype == key.dtype == value.dtype:
            raise ValueError("Attention inputs must have the same dtype")
        output = flash_attn_varlen_func(
            query,
            key,
            value,
            q_cu_seqlens,
            k_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            causal=False,
            deterministic=deterministic,
        )
    else:
        raise EnvironmentUnsupportedError(
            "Accelerator variable-length attention does not support "
            f"{query.device.type}"
        )
    return output[0] if isinstance(output, tuple) else output
