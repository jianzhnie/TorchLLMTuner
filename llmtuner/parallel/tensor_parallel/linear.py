"""TP projection execution and storage-free initialization coordinates.

These modules consume an already selected process group and sharding degree.
They do not inspect models, resolve patterns or choose parallel strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from ...accelerator.linear_collectives import (
    AllGatherLinear,
    LinearReduceScatter,
    all_gather_linear,
    linear_reduce_scatter,
)


@dataclass(frozen=True)
class TPInitializationSpec:
    """Storage-free original HF initialization and TP ownership coordinates."""

    module_type: type[nn.Module]
    global_shape: tuple[int, ...]
    dim: int
    size: int
    rank: int
    packed: bool = False


def shard_weight(
    weight: torch.Tensor, dim: int, *, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Cut ``weight`` into ``tp_size`` pieces along ``dim``; keep this rank's.

    A plain slice with no collective: every rank starts from the same full weight
    (the model is built identically everywhere), so each just drops the rest.
    """
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError(f"tp_size must be a positive integer, got {tp_size!r}")
    if (
        isinstance(tp_rank, bool)
        or not isinstance(tp_rank, int)
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError(
            f"tp_rank must be an integer in [0, {tp_size}), got {tp_rank!r}"
        )
    if weight.shape[dim] % tp_size != 0:
        raise ValueError(
            f"weight dim {dim} (size {weight.shape[dim]}) is not divisible by "
            f"tp_size={tp_size}"
        )
    return torch.chunk(weight.detach(), tp_size, dim=dim)[tp_rank].contiguous()


class ShardedLinear(nn.Module):
    """Initialize a weight shard; subclasses define activation communication.

    Feature counts describe the original projection, not the local shard.
    Projections without collectives retain no process-group or runtime flags.
    """

    shard_dim: int
    uses_collectives = True

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        tp_size: int,
        tp_rank: int,
        group: dist.ProcessGroup,
        use_symm_mem: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self._tp_init_specs = {
            "weight": TPInitializationSpec(
                nn.Linear, tuple(weight.shape), self.shard_dim, tp_size, tp_rank
            )
        }
        self.weight = nn.Parameter(
            shard_weight(weight, self.shard_dim, tp_size=tp_size, tp_rank=tp_rank),
            requires_grad=weight.requires_grad
            if isinstance(weight, nn.Parameter)
            else True,
        )
        if self.uses_collectives:
            self.group = group
            self.tp_size = tp_size
            self.use_symm_mem = use_symm_mem


class ColumnParallelLinear(ShardedLinear):
    """Column-parallel projection: output features split across TP ranks.

    Stores the weight as ``[out_features / tp, in_features]`` -- HF's own
    ``[out, in]`` layout cut on dim 0, which is the ``w_shard_n = [N / R, K]``
    layout ``AllGatherLinear`` contracts for (the op self-transposes).

    forward all-gathers the sequence shard and leaves the activation feature-
    sharded; backward is the dual (reduce-scatter of the input gradient, local
    weight gradient).
    """

    shard_dim = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The collective primitives are strictly 2D (sequence-major rows), but
        # the HF decoder feeds [B, T, K] hidden states: fold the leading dims
        # and restore them after, with the row count multiplied by tp_size --
        # that is the all-gathered sequence length.
        if x.ndim not in (2, 3):
            raise ValueError("TP projection input must be [T, K] or [1, T, K]")
        if x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(
                "TP realizers gather/reduce rows in rank order; a batched "
                f"[B, T, K] input with B={x.shape[0]} would interleave batches "
                "across ranks after the fold. Flatten to [T, K] or keep B=1 "
                "(the HF wrapper always runs B=1)."
            )
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = AllGatherLinear.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = all_gather_linear(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] * self.tp_size, y_2d.shape[-1])


class RowParallelLinear(ShardedLinear):
    """Row-parallel projection: input features split across TP ranks.

    Stores the weight as ``[out_features, in_features / tp]`` -- HF's layout cut
    on dim 1, which is what ``LinearReduceScatter`` expects.

    forward multiplies the feature-sharded activation and reduce-scatters the
    partial sums back to a sequence shard; backward is the dual (all-gather of the
    output gradient, local weight gradient).
    """

    shard_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror of ColumnParallelLinear.forward: fold to 2D for the collective, then
        # restore the leading dims with the row count divided by tp_size --
        # that is the reduce-scattered sequence shard.
        if x.ndim not in (2, 3):
            raise ValueError("TP projection input must be [T, K] or [1, T, K]")
        if x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(
                "TP realizers gather/reduce rows in rank order; a batched "
                f"[B, T, K] input with B={x.shape[0]} would interleave batches "
                "across ranks after the fold. Flatten to [T, K] or keep B=1 "
                "(the HF wrapper always runs B=1)."
            )
        lead = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if self.use_symm_mem:
            y_2d = LinearReduceScatter.apply(
                x_2d, self.weight, None, self.group, self.group.group_name
            )
        else:
            y_2d = linear_reduce_scatter(x_2d, self.weight, self.group)
        return y_2d.reshape(*lead[:-1], lead[-1] // self.tp_size, y_2d.shape[-1])


class ColwiseLinearNoGather(ShardedLinear):
    """Column-parallel projection without the fused sequence all-gather.

    Same weight shard as :class:`ColumnParallelLinear` (``[out / tp, in]``) but a plain
    local GEMM, for sites whose input is already full-sequence: the attention
    boundary gather (``GatherSequenceFirst``) runs upstream, because HF
    attention derives q/k/v shapes from ``hidden_states`` and cannot absorb a
    projection whose output is physically longer than its input.

    The backward is exact without any collective here: the rowwise o_proj's
    backward all-gather reassembles the full-sequence, total-loss output
    gradient before attention's backward runs, so the local ``dy.T @ x`` is the
    complete weight gradient for this rank's feature shard.
    """

    shard_dim = 0
    uses_collectives = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)
