"""Precision-controlled linear projections for decoder heads and MoE routers.

Semantics vendored from torchtitan's ``CastLinear``: the vocabulary logits are
scored in a configured dtype regardless of the dtype the rest of the model
runs in, because bf16 logits lose the precision the loss (and RL logprob/KL
math downstream of it) needs. Explicit models choose their output projection
at construction through ``build_lm_head``. Direct weight and bias registration
preserves checkpoint names and model-owned embedding/output weight tying.

RouterGateLinear also controls backward GEMM precision. Its custom autograd
registration uses the SPMD adapter without creating a distributed process group.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ...accelerator.spmd_backend import spmd

__all__ = ["CastLinear", "RouterGateLinear", "build_lm_head", "weights_are_tied"]


def weights_are_tied(head: nn.Module | None, embedding: nn.Module | None) -> bool:
    """Report live weight identity, including modules without a weight."""
    weight = getattr(head, "weight", None)
    return weight is not None and weight is getattr(embedding, "weight", None)


def build_lm_head(
    hidden_size: int, vocab_size: int, *, compute_dtype: str | None = None
) -> nn.Linear:
    """Build the declared output projection with one normal initialization.

    The default is an ordinary Linear. CastLinear uses the same constructor
    and parameter layout; model post_init and weight tying still own the
    initialization lifecycle. Reject unknown policies before allocating storage.
    """
    if compute_dtype is None:
        return nn.Linear(hidden_size, vocab_size, bias=False)
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if not isinstance(compute_dtype, str) or compute_dtype not in dtypes:
        raise ValueError(
            "lm_head compute_dtype must be float32, float16, bfloat16 or None, "
            f"got {compute_dtype!r}"
        )
    return CastLinear(
        hidden_size, vocab_size, bias=False, compute_dtype=dtypes[compute_dtype]
    )


class CastLinear(nn.Linear):
    """An ``nn.Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before ``F.linear``
    and the output is returned in that dtype. The stored parameters keep their
    original dtype -- autograd casts the incoming gradients back through the
    forward casts, so the weight gradient lands in the parameter's own dtype
    and the optimizer never sees an upcast copy. The cast cannot be cached
    across steps because the optimizer updates the weight every step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        compute_dtype: torch.dtype,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if compute_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "compute_dtype must be float16, bfloat16, float32 or float64, "
                f"got {compute_dtype!r}"
            )
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.compute_dtype = compute_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # An outer autocast context must not override the configured matmul dtype.
        with torch.autocast(device_type=input.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.to(self.compute_dtype)
            return F.linear(
                input.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias
            )


@spmd.register_local_autograd_function
class RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor) -> torch.Tensor:
        use_cuda_bf16_forward = (
            input_TD.device.type == "cuda"
            and input_TD.dtype is torch.bfloat16
            and weight_ED.dtype is torch.bfloat16
        )
        with torch.autocast(device_type=input_TD.device.type, enabled=False):
            if use_cuda_bf16_forward:
                input_forward_TD = input_TD
                weight_forward_ED = weight_ED
                # CUDA supports BF16 matmul with FP32 accumulation and output via
                # out_dtype. The portable path below promotes the operands because
                # this mixed input/output dtype is not supported by all devices.
                output_TE = torch.mm(
                    input_forward_TD, weight_forward_ED.T, out_dtype=torch.float32
                )
            else:
                input_forward_TD = input_TD.float()
                weight_forward_ED = weight_ED.float()
                output_TE = torch.mm(input_forward_TD, weight_forward_ED.T)
        ctx.save_for_backward(input_forward_TD, weight_forward_ED)
        ctx.input_dtype = input_TD.dtype
        ctx.weight_dtype = weight_ED.dtype
        return output_TE

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE: torch.Tensor):
        input_forward_TD, weight_forward_ED = ctx.saved_tensors
        grad_output_fp32_TE = grad_output_TE.float()

        with torch.autocast(device_type=grad_output_TE.device.type, enabled=False):
            grad_input_TD = None
            if ctx.needs_input_grad[0]:
                grad_input_TD = torch.mm(
                    grad_output_fp32_TE, weight_forward_ED.float()
                ).to(ctx.input_dtype)

            grad_weight_ED = None
            if ctx.needs_input_grad[1]:
                grad_weight_ED = torch.mm(
                    grad_output_fp32_TE.T, input_forward_TD.float()
                ).to(ctx.weight_dtype)
        return grad_input_TD, grad_weight_ED


class RouterGateLinear(nn.Linear):
    """The router's projection: one score per expert, always in fp32.

    A plain ``nn.Linear`` would return the input dtype, and a bf16 score can
    reorder a top-k on close calls, so both the forward GEMM and the backward
    GEMMs run in fp32 whatever the model's dtype is. The custom Function (rather
    than casting the operands inline) is what pins the backward as well; on CUDA
    with bf16 operands the forward keeps bf16 inputs and asks for an fp32
    accumulation, while every other device promotes the operands first.

    Subclasses ``nn.Linear`` rather than a bare ``nn.Module`` so the module
    signature is the one the rest of the stack expects: ``weight`` of shape
    ``(E, D)``, and a ``bias`` attribute that exists (as ``None``) even when the
    projection is built without one.

    Args:
        dim: model dimension (D).
        num_experts: number of experts (E).
        bias: add a bias to the scores. Defaults to False, matching the
            bias-free projection the HF MoE blocks ship.
    """

    def __init__(self, dim: int, num_experts: int, *, bias: bool = False) -> None:
        super().__init__(dim, num_experts, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_TE = RouterGateLinearFunction.apply(input, self.weight)
        if self.bias is not None:
            output_TE = output_TE + self.bias.float()
        return output_TE
