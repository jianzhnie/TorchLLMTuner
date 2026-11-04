"""NPU grouped GEMM with runtime pointer alignment for compiled execution.

Dynamo cannot trace Tensor.storage_offset(), and folding the check at trace time
misses later inputs with the same shape/strides but different offsets. Keep the
check inside an opaque functional op. Aligned tensors pass through unchanged;
only unaligned operands get new storage, preserving their supported strides.
Callers still validate dtype, dimensions and stride requirements.
"""

from typing import Any

import torch

__all__ = ["aligned_npu_grouped_mm"]


def _aligned_operand(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.data_ptr() % 16 == 0:
        return tensor
    # A contiguous clone of a transposed operand can introduce an unsupported
    # row stride (e.g. a contracting token dimension of 5). Keep the layout.
    aligned = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    aligned.copy_(tensor)
    return aligned


@torch.library.custom_op("llmtuner::aligned_npu_grouped_mm", mutates_args=())
def aligned_npu_grouped_mm(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Apply native grouped GEMM with aligned NPU operand storage.

    The fresh result never aliases inputs. The same op handles the 2D/3D,
    2D/2D and 3D/2D matrix modes used by native grouped GEMM's derivatives.
    """
    # EP's received token count can be unbacked during fullgraph tracing.
    # Handle empty contractions at runtime, including the weight-gradient
    # (2D/2D) and its (3D/2D) derivative modes, without a Python shape guard
    # in the traced caller. Native NPU grouped GEMM rejects empty operands.
    if left.numel() == 0 or right.numel() == 0:
        if left.ndim == 2 and right.ndim == 3:
            shape = (left.shape[0], right.shape[-1])
        elif left.ndim == 2 and right.ndim == 2:
            shape = (offsets.numel(), left.shape[0], right.shape[-1])
        elif left.ndim == 3 and right.ndim == 2:
            shape = (left.shape[1], right.shape[-1])
        else:
            raise ValueError("Unsupported grouped GEMM matrix dimensions")
        return left.new_zeros(shape)
    return torch._grouped_mm(
        _aligned_operand(left), _aligned_operand(right), offs=offsets
    )


@aligned_npu_grouped_mm.register_fake
def _fake_grouped_mm(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    # Reuse native shape/stride inference, including expert/contracting layouts.
    return torch._grouped_mm(left, right, offs=offsets)


def _setup_context(
    ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    ctx.save_for_backward(*inputs)


def _backward(
    ctx: Any, gradient: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
    left, right, offsets = ctx.saved_tensors
    # Sum/expand backward can supply zero strides that grouped GEMM rejects.
    gradient = gradient.contiguous()
    grad_left = (
        aligned_npu_grouped_mm(gradient, right.transpose(-2, -1), offsets)
        if ctx.needs_input_grad[0]
        else None
    )
    grad_right = (
        aligned_npu_grouped_mm(left.transpose(-2, -1), gradient, offsets)
        if ctx.needs_input_grad[1]
        else None
    )
    return grad_left, grad_right, None


aligned_npu_grouped_mm.register_autograd(_backward, setup_context=_setup_context)
