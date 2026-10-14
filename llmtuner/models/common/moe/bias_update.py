"""MoE load-balancing hooks: the per-step expert-bias updates.

Sign-based and quantile-histogram bias updates share one policy-selection
entry point, ``register_moe_balancing_hook``, and token-statistic reductions.
Both are optimizer step pre-hooks over the MoE blocks ``collect_moe_layers``
finds; the blocks themselves live in ``layer.py`` and the routers in
``routing.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from llmtuner.accelerator.collectives import all_reduce

from .layer import MoE, collect_moe_layers
from .routing import QuantileBalancedTopKRouter

if TYPE_CHECKING:
    from ....parallel.topology import ParallelTopology


def _reduce_token_statistics(
    statistics: Sequence[torch.Tensor], parallel_dims: ParallelTopology | None
) -> tuple[torch.Tensor, ...]:
    """Sum token-partial counts over DP, CP, and TP when EP uses its tokens."""
    flattened = torch.cat([tensor.reshape(-1) for tensor in statistics])
    axes = ["dp", "cp"] + (["tp"] if parallel_dims and parallel_dims.ep_enabled else [])
    for axis in axes:
        mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh(axis)
        if mesh is None:
            continue
        all_reduce(flattened, group=mesh.get_group())
    return tuple(
        reduced.reshape(tensor.shape)
        for reduced, tensor in zip(
            flattened.split([tensor.numel() for tensor in statistics]),
            statistics,
            strict=True,
        )
    )


def update_expert_bias(
    moe_layers: list[MoE],
    *,
    parallel_dims: ParallelTopology | None = None,
) -> None:
    """Turn every MoE's accumulated token counts into one bias update.

    ``tokens_per_expert_E`` counted only the tokens *this* rank saw, so it is
    summed over every axis that shards one token stream before it means
    anything. Three collectives, for three different reasons:

    * ``dp`` -- each data-parallel rank processes a different slice of the
      batch, so the counts are partial over it. This is the one that matters
      for a plain FSDP run: without it every rank would balance its own shard
      and the biases would drift apart.
    * ``cp`` -- the sequence is split across the context-parallel ranks, so the
      tokens are partial there too.
    * ``tp`` -- only when EP is on: EP borrows ranks from TP, so the token
      stream is sharded over TP as well. Without EP every TP rank already sees
      the same full stream, so summing would multiply the counts by ``tp``
      (harmless for a sign-based step, but wrong for anything else reading it).

    Pipeline parallelism needs no collective here: it is a layer split, not a
    token split, so each stage's blocks see their own whole token stream.

    The expert dimension is deliberately *not* reduced. ``tokens_per_expert_E``
    is the global per-expert count -- the routing map spans all ``E`` experts on
    every EP rank, before the dispatcher narrows anything to a local shard -- so
    an EP all-reduce would double-count.

    Because every rank ends up with identical counts, the identical update is
    applied identically and ``expert_bias_E`` stays replicated, which is what
    the forward assumes.
    """
    if not moe_layers:
        return
    counts = _reduce_token_statistics(
        [moe.tokens_per_expert_E for moe in moe_layers], parallel_dims
    )

    for moe, counts_E in zip(moe_layers, counts, strict=True):
        moe.tokens_per_expert_E.copy_(counts_E)
        moe.update_expert_bias()


@torch.no_grad()
def update_quantile_expert_bias(
    moe_layers: list[MoE],
    *,
    parallel_dims: ParallelTopology | None = None,
) -> None:
    """Reduce the quantile histograms and write the next expert biases.

    The histograms count the same tokens ``tokens_per_expert_E`` counts, so
    they are summed over exactly the same axes (dp, cp, and tp only when EP
    shards the token stream over it) -- see ``update_expert_bias`` for why
    those axes and no others. Every rank then holds the identical global
    histogram, computes the identical estimate, and ``expert_bias_E`` stays
    replicated, which is what the forward assumes.

    After the update each layer's histogram and token counter are drained:
    both are per-step scratch, and the counter is otherwise never consumed
    (the sign-based ``update_expert_bias`` that drains it is disabled for
    these layers).
    """
    if not moe_layers:
        return
    histograms = [
        moe.router.quantile_balancer.required_bias_histogram_EB for moe in moe_layers
    ]
    reduced_histograms = _reduce_token_statistics(histograms, parallel_dims)

    for moe, histogram_EB in zip(moe_layers, reduced_histograms, strict=True):
        quantile_balancer = moe.router.quantile_balancer
        moe.expert_bias_E.copy_(
            quantile_balancer.estimate_expert_bias(histogram_EB, moe.expert_bias_E)
        )
        quantile_balancer.required_bias_histogram_EB.zero_()
        moe.tokens_per_expert_E.zero_()


def register_moe_balancing_hook(
    optimizer: torch.optim.Optimizer,
    *,
    model_parts: Sequence[nn.Module],
    parallel_dims: ParallelTopology | None,
) -> None:
    """Validate one balancing policy and register its optimizer step pre-hook.

    Traverse model parts once. Quantile routing must cover every MoE layer;
    otherwise sign-based coefficients must be set on every layer or none.
    Dense models and models without balancing need no hook or collective.
    """
    layers: list[MoE] = []
    seen: set[int] = set()
    for part in model_parts:
        for moe in collect_moe_layers(part):
            if id(moe) not in seen:
                layers.append(moe)
                seen.add(id(moe))
    if not layers:
        return
    quantile_layers = [
        moe for moe in layers if isinstance(moe.router, QuantileBalancedTopKRouter)
    ]
    if quantile_layers:
        if len(quantile_layers) != len(layers) or any(
            moe.load_balance_coeff is not None for moe in layers
        ):
            raise ValueError(
                "Quantile-balanced routing is mutually exclusive with the "
                "sign-based load-balancing bias: every MoE layer must use a "
                "QuantileBalancedTopKRouter with load_balance_coeff=None, or "
                "none may."
            )
        update_bias = update_quantile_expert_bias
    else:
        enabled = layers[0].load_balance_coeff is not None
        if any((moe.load_balance_coeff is not None) != enabled for moe in layers):
            raise ValueError(
                "MoE load_balance_coeff must be configured consistently across "
                "all MoE layers. Either set it for every MoE layer or leave it "
                "unset for all MoE layers."
            )
        if not enabled:
            return
        update_bias = update_expert_bias
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: update_bias(layers, parallel_dims=parallel_dims)
    )
