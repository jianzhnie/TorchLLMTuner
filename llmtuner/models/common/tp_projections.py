"""Model components that fold the TP collectives into their GEMMs.

Vendored from torchtitan ``models/common/async_linear.py`` (renamed from
``dist_gemm.py`` upstream in 9e159aed7).
These replace the stock QKV, output and SwiGLU projections. They move the TP
collective inside the GEMM, using the autograd Functions in
``llmtuner/parallel/tensor_parallel/linear_ops.py`` (which holds the collective+GEMM
math itself; this file is only the wiring and the fallbacks).

What changed from upstream:

* Modules are constructed directly. QKV and FFN inherit their base
  constructors; ReduceScatterLinear takes feature sizes followed by a keyword
  bias option. Upstream's nested ``Config`` subclasses are omitted.
* ``TensorParallelFeedForward`` subclasses :class:`~llmtuner.models.common.feed_forward.
  FeedForward`, as it does upstream: the fused and unfused paths share the weight
  layout (``w13`` holding the interleaved gate and up) and the activation split,
  so only the two GEMMs differ. It overrides ``forward`` and falls back to the
  inherited implementation when TP is off.
* ``torch_remat`` is gone. Upstream wraps each projection in
  ``remat.region(..., recompute=...)``, which only steers activation
  checkpointing; calling the projection directly is the same arithmetic.
* The TP group is read per forward through
  ``llmtuner.accelerator.spmd.spmd_mesh_group``.

The fallback matters: when TP is not active there is no collective to fuse, so
each module runs its plain path. That keeps a TP=1 run working, but it also means
a misconfiguration looks like success -- hence the warning rather than silence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmtuner.accelerator.spmd import spmd_mesh_group
from llmtuner.models.common.attention.qkv import QKVLinear
from llmtuner.models.common.feed_forward import FeedForward
from llmtuner.parallel.tensor_parallel.linear_ops import (
    AllGatherLinear,
    LinearReduceScatter,
)

from ...utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "AllGatherFusedQKVLinear",
    "TensorParallelFeedForward",
    "ReduceScatterLinear",
]

# Shape suffix legend, scoped to this file:
#   T = tokens, D = model dimension, N = output features, F = ffn hidden dim,
#   H = head dimension.

_WARNED_NO_TP = False


def warn_once_no_tp_overlap() -> None:
    """Say so when the dist-GEMM modules were selected but TP is not on.

    Otherwise the fallback is indistinguishable from the feature working: the run
    succeeds, the loss looks fine, and nothing ran fused. TP=1 remains runnable
    through the standard projection, with a warning identifying the fallback.
    """
    global _WARNED_NO_TP
    if not _WARNED_NO_TP:
        _WARNED_NO_TP = True
        logger.warning(
            "Tensor-parallel projection selected but tensor parallelism is not "
            "active; running the standard projection without collective "
            "overlap."
        )


class AllGatherFusedQKVLinear(QKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard.

    Otherwise identical to :class:`QKVLinear`: same weights, same checkpoint
    keys, same output split. Only the projection step differs, which is why it
    overrides ``_project`` rather than ``forward``.
    """

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            warn_once_no_tp_overlap()
            return super()._project(x)

        return AllGatherLinear.apply(
            x,
            self.wqkv.weight,
            self.wqkv.bias,
            tp_group,
            tp_group.group_name,
        )


class ReduceScatterLinear(nn.Linear):
    """A rowwise linear whose matmul is fused with the TP reduce-scatter.

    Constructed from full feature dimensions, unlike the HF sharding wrapper
    in ``parallel/tensor_parallel/tensor_parallel.py``, which receives already
    sharded weights. This module delegates fused communication and GEMMs to
    the ``LinearReduceScatter`` autograd Function.

    Args:
        in_features: input feature count (the weight's second dim).
        out_features: output feature count (the weight's first dim).
        bias: whether to allocate a bias parameter. False stores ``bias=None``;
            the constructor option itself must be a boolean.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ) -> None:
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive integer, got {value!r}")
        if not isinstance(bias, bool):
            raise TypeError(f"bias must be bool, got {type(bias).__name__}")
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            warn_once_no_tp_overlap()
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return LinearReduceScatter.apply(
            x,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


class TensorParallelFeedForward(FeedForward):
    """SwiGLU feed-forward with both TP collectives folded into its GEMMs.

    The fused ``w13`` projection consumes an all-gather of the sequence shard;
    ``w2`` is row-parallel and reduce-scatters back to a sequence shard.

    ``w13`` holds the gate and up projections *interleaved* along its output dim
    -- ``[g0, u0, g1, u1, ...]`` -- so ``w1`` and ``w3`` can be recovered by
    unflattening on the last axis. That is the layout an HF-style checkpoint
    expects, and the pairing ``(g_i, u_i)`` is what the activation consumes.
    Both of those are inherited from :class:`FeedForward`; this class replaces
    only the two projections' execution.

    Args:
        w13: the fused gate-and-up projection, ``dim -> 2 * hidden_dim``.
        w2: the down projection, ``hidden_dim -> dim``.
        activation_fn: the gated activation; defaults to SwiGLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            warn_once_no_tp_overlap()
            return super().forward(x)

        gate_up_TF = AllGatherLinear.apply(
            x,
            self.w13.weight,
            self.w13.bias,
            tp_group,
            tp_group.group_name,
        )
        out_TD = LinearReduceScatter.apply(
            self.activation_fn(*self.split_gate_up(gate_up_TF)),
            self.w2.weight,
            self.w2.bias,
            tp_group,
            tp_group.group_name,
        )
        return out_TD
