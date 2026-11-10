"""NPU grouped GEMM with runtime pointer alignment for compiled execution.

Dynamo cannot trace Tensor.storage_offset(), and folding the check at trace time
misses later inputs with the same shape/strides but different offsets. Keep the
check inside an opaque functional op. Aligned tensors pass through unchanged;
only unaligned operands get new storage, preserving their supported strides.
Callers still validate dtype, dimensions and stride requirements.
"""

from typing import Any

import torch

__all__ = ["aligned_npu_grouped_mm", "npu_grouped_mm_layout_supported"]


def npu_grouped_mm_layout_supported(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Metadata contract for the BF16 grouped kernel and its backward layouts.

    Pointer alignment is handled separately inside aligned_npu_grouped_mm.
    The output width becomes a row stride in backward, so checking operands
    alone is insufficient. This predicate performs no device access.
    """
    return (
        left.dtype == right.dtype == torch.bfloat16
        and right.shape[-1] % 8 == 0
        and all(
            stride > 0 and (stride == 1 or stride % 8 == 0)
            for stride in (*left.stride(), *right.stride())
        )
    )


def _validate_metadata(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> None:
    """Check shape contracts even when empty operands bypass the native kernel.

    Keep this metadata-only: reading offset values here would synchronize NPU
    execution and cannot run with FakeTensors or unbacked token dimensions.
    """
    if (left.ndim, right.ndim) not in ((2, 3), (2, 2), (3, 2)):
        raise ValueError("Unsupported grouped GEMM matrix dimensions")
    if offsets.ndim != 1:
        raise ValueError("Grouped GEMM offsets must be one-dimensional")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError("Grouped GEMM offsets must use int32 or int64")
    if left.dtype != right.dtype:
        raise TypeError("Grouped GEMM operands must have the same dtype")
    if left.device != right.device or left.device != offsets.device:
        raise ValueError("Grouped GEMM operands and offsets must share a device")
    torch._check(
        left.shape[-1] == right.shape[-2],
        lambda: "Grouped GEMM contracting dimensions must match",
    )
    if left.ndim == 3 or right.ndim == 3:
        experts = left.shape[0] if left.ndim == 3 else right.shape[0]
        torch._check(
            offsets.numel() == experts,
            lambda: "Grouped GEMM requires one offset per expert",
        )


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


def _output_shape(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> tuple:
    if left.ndim == 2 and right.ndim == 3:
        return (left.shape[0], right.shape[-1])
    if left.ndim == 2 and right.ndim == 2:
        return (offsets.numel(), left.shape[0], right.shape[-1])
    return (left.shape[1], right.shape[-1])


@torch.library.custom_op("llmtuner::aligned_npu_grouped_mm", mutates_args=())
def aligned_npu_grouped_mm(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Apply native grouped GEMM with aligned NPU operand storage.

    The fresh result never aliases inputs. The same op handles the 2D/3D,
    2D/2D and 3D/2D matrix modes used by native grouped GEMM's derivatives.
    """
    _validate_metadata(left, right, offsets)
    # EP's received token count can be unbacked during fullgraph tracing.
    # Handle empty contractions at runtime, including the weight-gradient
    # (2D/2D) and its (3D/2D) derivative modes, without a Python shape guard
    # in the traced caller. Native NPU grouped GEMM rejects empty operands.
    if left.numel() == 0 or right.numel() == 0:
        return left.new_zeros(_output_shape(left, right, offsets))
    return torch._grouped_mm(
        _aligned_operand(left), _aligned_operand(right), offs=offsets
    )


@aligned_npu_grouped_mm.register_fake
def _fake_grouped_mm(
    left: torch.Tensor, right: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    # Native Meta rejects empty contraction strides which our runtime handles
    # without calling the kernel. Infer our own contiguous NPU output contract;
    # do not branch on a possibly unbacked received-token dimension here.
    _validate_metadata(left, right, offsets)
    return left.new_empty(_output_shape(left, right, offsets))


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
