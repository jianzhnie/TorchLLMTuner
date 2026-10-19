"""Gated activations for feed-forward layers and MoE experts.

Adapted from torchtitan ``models/common/activation.py``. ActivationFn defines
the shared ``(gate, up)`` interface; modules are constructed directly without
upstream model-config factories.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ActivationFn", "SwiGLU", "SituGLU"]


class ActivationFn(nn.Module, ABC):
    """A gated activation: ``(gate, up)`` in, one tensor out.

    Both inputs have the same shape, and so does the result, so it can be fed
    straight into the down projection.
    """

    @abstractmethod
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Combine the gate and up halves of a fused projection."""
        raise NotImplementedError


class SwiGLU(ActivationFn):
    """``silu(gate) * up`` -- the standard gated activation."""

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


class SituGLU(ActivationFn):
    """Kimi's bounded gate, with an optional bounded up projection.

    Matches FSDPTurbo ops/cpu/situ_glu.py at 0af4eca. Compute in FP32
    before restoring the projection dtype; no optional kernel is imported.
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("SituGLU beta must be finite and positive")
        if linear_beta is not None and (
            not math.isfinite(linear_beta) or linear_beta < 0
        ):
            raise ValueError("SituGLU linear_beta must be finite and nonnegative")
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        activation = self.beta * torch.tanh(gate / self.beta) * gate.sigmoid()
        if self.linear_beta not in (None, 0.0):
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (activation * up).to(dtype)
