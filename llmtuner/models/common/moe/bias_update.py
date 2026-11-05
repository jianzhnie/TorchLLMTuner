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


@torch.no_grad()
def reset_routing_statistics(model_parts: Sequence[nn.Module]) -> None:
    """Discard observations before starting a new accumulation window.

    A failed step may not have reached the optimizer hooks that normally drain
    counters and histograms. These are non-persistent scratch buffers; learned
    router weights and the persistent expert bias must remain unchanged.
    """
    seen: set[int] = set()
    for part in model_parts:
        for module in part.modules():
            if not isinstance(module, MoE) or id(module) in seen:
                continue
            seen.add(id(module))
            module.tokens_per_expert_E.zero_()
            if isinstance(module.router, QuantileBalancedTopKRouter):
                module.router.quantile_balancer.required_bias_histogram_EB.zero_()


def _reduce_token_statistics(
    statistics: Sequence[torch.Tensor], parallel_dims: ParallelTopology | None
) -> tuple[torch.Tensor, ...]:
    """Sum token-partial counts over DP, CP, and TP when EP uses its tokens."""
    # Shared counters have identical ownership on every rank. Reduce each
    # object once, then restore the caller's layer order without copying.
    unique = list({id(tensor): tensor for tensor in statistics}.values())
    flattened = torch.cat([tensor.reshape(-1) for tensor in unique])
    axes = ["dp", "cp"] + (["tp"] if parallel_dims and parallel_dims.ep_enabled else [])
    for axis in axes:
        mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh(axis)
        if mesh is None:
            continue
        all_reduce(flattened, group=mesh.get_group())
    reduced_by_id = {
        id(tensor): reduced.reshape(tensor.shape)
        for reduced, tensor in zip(
            flattened.split([tensor.numel() for tensor in unique]),
            unique,
            strict=True,
        )
    }
    return tuple(reduced_by_id[id(tensor)] for tensor in statistics)


def _combine_shared_statistics(
    indices: list[int],
    statistics: Sequence[torch.Tensor],
    reduced: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Count each original buffer once, even when multiple layers share it."""
    contributions: dict[int, torch.Tensor] = {}
    for index in indices:
        contributions.setdefault(id(statistics[index]), reduced[index])
    values = list(contributions.values())
    return values[0] if len(values) == 1 else torch.stack(values).sum(0)


@torch.no_grad()
def collect_routing_load_metrics(
    model_parts: Sequence[nn.Module],
    parallel_dims: ParallelTopology | None,
    device: torch.device,
) -> dict[str, float]:
    """Observe native MoE counters before optimizer hooks drain them.

    Counts use the same ownership reductions as balancing. Scalar aggregation
    across WORLD handles PP stages with differing numbers of local MoE layers.
    Repeated data replicas cancel in averages; inactive layers are not counted.
    """
    layers = list(
        {
            id(layer): layer
            for part in model_parts
            for layer in collect_moe_layers(part)
        }.values()
    )
    totals = torch.zeros(4, device=device, dtype=torch.float32)
    if layers:
        counts = _reduce_token_statistics(
            [layer.tokens_per_expert_E.detach().clone() for layer in layers],
            parallel_dims,
        )
        for values in counts:
            mean = values.float().mean()
            active = (mean > 0).float()
            totals[0] += active
            totals[1] += active * (values == 0).float().mean()
            totals[2] += (
                active * values.float().std(unbiased=False) / mean.clamp_min(1e-12)
            )
            totals[3] = torch.maximum(
                totals[3], active * values.max() / mean.clamp_min(1e-12)
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals[:3])
        torch.distributed.all_reduce(totals[3:], op=torch.distributed.ReduceOp.MAX)
    count, empty, cv, maximum = totals.tolist()
    if not count:
        return {}
    return {
        "moe/empty_expert_fraction": empty / count,
        "moe/load_cv": cv / count,
        "moe/max_to_mean_load": maximum,
    }


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
    # A shared bias has one update owner, even across distinct MoE modules.
    # Validate ownership before communication or any mutation.
    unique_layers = list({id(moe): moe for moe in moe_layers}.values())
    groups: dict[int, list[int]] = {}
    for index, moe in enumerate(unique_layers):
        key = id(moe.expert_bias_E) if moe.expert_bias_E is not None else id(moe)
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        coefficient = unique_layers[indices[0]].load_balance_coeff
        if any(unique_layers[i].load_balance_coeff != coefficient for i in indices):
            raise ValueError("Shared expert bias requires identical load_balance_coeff")

    statistics = [moe.tokens_per_expert_E for moe in unique_layers]
    counts = _reduce_token_statistics(statistics, parallel_dims)
    for indices in groups.values():
        owner = unique_layers[indices[0]]
        combined = _combine_shared_statistics(indices, statistics, counts)
        owner.tokens_per_expert_E.copy_(combined)
        owner.update_expert_bias()
        if owner.load_balance_coeff is not None and owner.expert_bias_E is not None:
            for index in indices[1:]:
                unique_layers[index].tokens_per_expert_E.zero_()


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
    unique_layers = list({id(moe): moe for moe in moe_layers}.values())
    groups: dict[int, list[int]] = {}
    for index, moe in enumerate(unique_layers):
        groups.setdefault(id(moe.expert_bias_E), []).append(index)
    for indices in groups.values():
        owner = unique_layers[indices[0]].router.quantile_balancer
        policy = (owner.num_experts, owner.top_k, owner.num_bins)
        for index in indices[1:]:
            other = unique_layers[index].router.quantile_balancer
            if (other.num_experts, other.top_k, other.num_bins) != policy:
                raise ValueError(
                    "Shared quantile bias requires identical histogram policy"
                )
    histograms = [
        moe.router.quantile_balancer.required_bias_histogram_EB for moe in unique_layers
    ]
    reduced_histograms = _reduce_token_statistics(histograms, parallel_dims)
    for indices in groups.values():
        owner = unique_layers[indices[0]]
        combined = _combine_shared_statistics(indices, histograms, reduced_histograms)
        owner.expert_bias_E.copy_(
            owner.router.quantile_balancer.estimate_expert_bias(
                combined, owner.expert_bias_E
            )
        )
        for index in indices:
            histograms[index].zero_()
            unique_layers[index].tokens_per_expert_E.zero_()


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
