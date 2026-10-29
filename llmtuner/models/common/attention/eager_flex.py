"""Explicit eager Flex execution without process-wide PyTorch patches."""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask

from llmtuner.errors import EnvironmentUnsupportedError, UnsupportedCombinationError


def _identity(score, batch, head, query, key):
    return score


def eager_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Run the eager Flex HOP with GQA and an explicit boolean mask contract."""
    if torch.compiler.is_compiling():
        raise UnsupportedCombinationError(
            "eager Flex is not a model-compile backend; select the native backend"
        )
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("eager Flex requires four-dimensional Q/K/V tensors")
    if any(tensor.device != query.device for tensor in (key, value)):
        raise ValueError("eager Flex requires Q/K/V on the same device")
    if any(tensor.dtype != query.dtype for tensor in (key, value)):
        raise ValueError("eager Flex requires identical Q/K/V dtypes")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("eager Flex supports float32, float16, and bfloat16")
    if query.device.type not in ("cpu", "npu"):
        raise UnsupportedCombinationError("eager Flex currently supports CPU and NPU")
    if (
        query.shape[0] != key.shape[0]
        or key.shape[:3] != value.shape[:3]
        or query.shape[-1] != key.shape[-1]
        or query.shape[1] == 0
        or key.shape[1] == 0
        or query.shape[1] % key.shape[1]
        or query.shape[-2] == 0
        or key.shape[-2] == 0
    ):
        raise ValueError("eager Flex requires compatible batch, GQA heads and lengths")
    if not isinstance(block_mask, BlockMask):
        raise TypeError("eager Flex requires an explicit BlockMask")
    if tuple(block_mask.seq_lengths) != (query.shape[-2], key.shape[-2]):
        raise ValueError("eager Flex BlockMask lengths must match Q/K lengths")
    if block_mask.kv_num_blocks.device != query.device:
        raise ValueError("eager Flex BlockMask must share the Q/K/V device")
    if block_mask.q_indices is None and torch.is_grad_enabled():
        raise ValueError("eager Flex backward requires BlockMask query indices")
    try:
        from torch._higher_order_ops.flex_attention import flex_attention
        from torch.nn.attention.flex_attention import _apply_kernel_options
    except ImportError as error:
        raise EnvironmentUnsupportedError(
            "eager Flex requires PyTorch Flex HOP and kernel-option APIs "
            "(validated with torch 2.12)"
        ) from error
    options = _apply_kernel_options(query, key, value, False, None)
    output, _, _ = flex_attention(
        query,
        key,
        value,
        _identity,
        block_mask.as_tuple(),
        query.shape[-1] ** -0.5 if scale is None else scale,
        options,
    )
    return output
