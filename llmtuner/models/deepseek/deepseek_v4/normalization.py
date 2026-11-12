# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 normalization with explicit native cast boundaries."""

import torch
from torch import nn
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference


class DeepseekV4RMSNorm(reference.DeepseekV4RMSNorm):
    """Normalize in FP32, cast back, then multiply the affine weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * values.to(dtype)


class DeepseekV4UnweightedRMSNorm(reference.DeepseekV4UnweightedRMSNorm):
    """Cast the FP32 reciprocal RMS before multiplying the original input."""

    def __init__(self, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reciprocal = torch.rsqrt(
            hidden_states.float().square().mean(-1, keepdim=True) + self.eps
        )
        return hidden_states * reciprocal.to(hidden_states.dtype)
