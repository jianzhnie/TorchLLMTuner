"""Variable-length non-causal attention kernels."""

from __future__ import annotations

import torch

from ..registry import dispatch_kernel
from ..validation import SequenceOffsets


def fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    q_cu_seqlens: SequenceOffsets,
    k_cu_seqlens: SequenceOffsets,
    use_eager: bool = False,
) -> torch.Tensor:
    """Non-causal MHA for [tokens, heads, head_dim] inputs of equal dtype.

    Q/K may have different token counts but must describe the same documents.
    Offsets are nondecreasing integers, starting at zero and ending at the
    respective token count; None denotes a single document. Empty queries are
    allowed, but nonempty queries require keys. There is no dropout or mask.
    use_eager selects torch reference math on the existing device, without
    transferring tensors. Default dispatch follows query.device.
    """
    return dispatch_kernel(
        "fusion_attention",
        query,
        key,
        value,
        head_num=head_num,
        scale=scale,
        q_cu_seqlens=q_cu_seqlens,
        k_cu_seqlens=k_cu_seqlens,
        device_type="cpu" if use_eager else None,
    )
