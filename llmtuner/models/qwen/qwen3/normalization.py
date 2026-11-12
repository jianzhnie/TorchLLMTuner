"""Qwen3 norm construction with explicit device-kernel selection."""

import torch
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from ....kernel.ops.rms_norm import rms_norm
from ....kernel.validation import rms_norm_inputs


def validate_norm_backend(config: PretrainedConfig) -> None:
    """Reject unavailable policies before constructing model parameters."""
    backend = getattr(config, "norm_backend", "native")
    if backend not in ("native", "torch", "npu"):
        raise ValueError(
            "Explicit Qwen3 supports norm_backend native/torch/npu; "
            f"{backend!r} needs its model-owned normalization implementation."
        )


class Qwen3KernelRMSNorm(Qwen3MoeRMSNorm):
    """Keep Qwen3 parameter layout while delegating the kernel to its owner."""

    def __init__(self, hidden_size: int, eps: float, *, backend: str) -> None:
        if backend not in ("torch", "npu"):
            raise ValueError("Kernel RMSNorm backend must be torch or npu")
        super().__init__(hidden_size, eps=eps)
        self.backend = backend

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.backend == "torch" and hidden_states.dtype != self.weight.dtype:
            # Qwen3 casts normalized values before the affine multiply. The
            # generic fused op casts its final result to input dtype instead;
            # preserve the model's promotion order for mixed storage dtypes.
            rms_norm_inputs(hidden_states, self.weight, self.variance_epsilon)
            values = hidden_states.float()
            values = values * torch.rsqrt(
                values.square().mean(-1, keepdim=True) + self.variance_epsilon
            )
            return self.weight * values.to(hidden_states.dtype)
        if self.backend == "npu":
            if hidden_states.device.type != "npu":
                raise RuntimeError("norm_backend=npu requires NPU input tensors")
            # The fused output has the input dtype. Mixed affine/input dtypes
            # need an explicitly validated promotion policy, not an implicit cast.
            if hidden_states.dtype != self.weight.dtype:
                raise TypeError(
                    "NPU Qwen3 RMSNorm requires matching input/weight dtype"
                )
        return rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            use_eager=self.backend == "torch",
        )


def build_rms_norm(hidden_size: int, eps: float, *, backend: str) -> Qwen3MoeRMSNorm:
    """Leave native construction untouched; opt in without a forward patch."""
    if backend == "native":
        return Qwen3MoeRMSNorm(hidden_size, eps=eps)
    return Qwen3KernelRMSNorm(hidden_size, eps, backend=backend)
