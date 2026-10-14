"""Gated activations for feed-forward layers and MoE experts.

Adapted from torchtitan ``models/common/activation.py``. ActivationFn defines
the shared ``(gate, up)`` interface; modules are constructed directly without
upstream model-config factories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ActivationFn", "SwiGLU"]


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
