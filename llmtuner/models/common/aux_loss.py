"""Auxiliary-loss gradient injection and metrics, adapted from torchtitan.

``AuxLoss`` scales gradients by the configured coefficient and the step's
global routed-token count, including masked prompts and excluding padding.
Forward calls accumulate local metrics; an optimizer pre-hook snapshots these
on each instance. Collection sums over each group's mesh and PP stages,
then divides by the current runtime's instance count to report the mean over layers.

Full/selective activation checkpoint replay suppresses metric writes. Region
checkpointing may retain the accumulation and injection through torch_remat.
Upstream sources and adaptations are in ``docs/llmtuner_upstream_map.md``.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import all_reduce

from llmtuner.accelerator.device import device_type

if TYPE_CHECKING:
    # Keep model import independent of the parallel package's initialization.
    from llmtuner.parallel.topology import ParallelTopology

__all__ = [
    "AuxLoss",
    "collect_aux_loss_metrics",
    "register_aux_loss_metrics_hook",
]

# Non-reentrant activation checkpointing replays the forward during backward.
# Its recompute context suppresses metric side effects, while the injection
# remains in the graph so the router gradient is still computed.
in_checkpoint_recompute: ContextVar[bool] = ContextVar(
    "in_checkpoint_recompute", default=False
)


class AuxLossInjection(torch.autograd.Function):
    """Identity forward that injects an aux-loss gradient on the backward pass.

    The carrier passes through untouched; its only job is to give the aux
    loss a differentiable path into the graph. On the way back, the aux loss
    contributes exactly 1.0 to its own gradient -- the ``coeff`` and the
    ``1/denominator`` scaling were already folded into the loss value, so the
    injection itself stays a constant.
    """

    @staticmethod
    def forward(ctx, carrier, aux_loss):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def backward(ctx, grad_carrier):  # pyrefly: ignore[bad-override]
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


class AuxLoss(nn.Module):
    """Base class: subclasses call ``inject()`` each forward.

    Each instance accumulates its scaled value in the ``instance_acc`` buffer; a
    step pre-hook rolls those into the ``group_acc`` registers, which
    ``collect_aux_loss_metrics`` reduces for logging.

    Args:
        coeff: scales the loss's gradient contribution.
        reduce_mesh: mesh the per-step metric is summed over -- ``"batch"``
            (dp) for cp-identical losses like the microbatch-wise load-balance
            loss, ``"loss"`` (dp, cp and tp: llmtuner's TP is sequence-parallel,
            so a rank's loss covers only its ``T / tp`` token shard) for
            per-token-additive losses whose rank-local values add up across
            coordinates.

    Normalization: ``denominator`` is the step's global routed-token count, set
    by the trainer via ``set_step_denominator`` before the first forward.
    """

    # Standalone calls without model_parts use this construction registry.
    # The trainer counts and collects its actual instances across PP stages.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Global routed-token count of the current step, set before the first
    # forward. Unlike CE's denominator, this includes masked prompt tokens.
    _step_denominator: ClassVar[torch.Tensor | None] = None

    # Per metric group (``(reduce_mesh, metric_name)``): this rank's total value
    # for the current step, rolled up from the per-instance ``instance_acc``
    # buffers by ``rollup_aux_loss_metrics`` at each optimizer step pre-hook and
    # reduced by ``collect_aux_loss_metrics`` at log time. Cleared at the next
    # pre-hook.
    group_acc: ClassVar[dict[tuple[str, str], torch.Tensor]] = {}

    def __init__(self, *, coeff: float, reduce_mesh: str = "batch") -> None:
        super().__init__()
        self.coeff = coeff
        self.reduce_mesh = reduce_mesh
        # Runtime scratch is attached to the actual loss instance. Values are
        # assigned after materialization and are never checkpoint state.
        self._runtime_denominator: torch.Tensor | None = None
        self._step_metric: torch.Tensor | None = None
        # Per-instance accumulator: the sum of this loss instance's scaled
        # per-microbatch values over the current training step. Filled in the
        # forward; rolled into ``group_acc`` and zeroed by ``rollup_aux_loss_metrics``
        # at each optimizer step pre-hook. Non-persistent: it is scratch, and
        # a checkpoint resuming mid-step should not restore a partial sum.
        self.register_buffer(
            "instance_acc", torch.zeros((), dtype=torch.float32), persistent=False
        )
        AuxLoss._group_counts[(self.reduce_mesh, self.metric_name)] += 1

    @classmethod
    def has_pending_counts(cls) -> bool:
        """Whether any aux-loss group is registered (any MoE model built)."""
        return bool(cls._group_counts)

    @property
    def metric_name(self) -> str:
        """The class name in snake_case, e.g. ``MicrobatchWiseLoadBalanceLoss``
        -> ``microbatch_wise_load_balance_loss``."""
        return re.sub(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            "_",
            type(self).__name__,
        ).lower()

    @classmethod
    def set_step_denominator(
        cls,
        denominator: torch.Tensor,
        *,
        model_parts: Sequence[nn.Module] | None = None,
    ) -> None:
        """Set the current step's global routed-token count.

        The trainer calls this once per step with the DP-summed count of
        non-padding input tokens that pass through the router and its local
        model_parts. This also discards unfinished metric scratch from an
        earlier failed step. Standalone calls set the class default instead.
        """
        if model_parts is None:
            cls._step_denominator = denominator
            return
        for part in model_parts:
            for module in part.modules():
                if isinstance(module, AuxLoss):
                    module._runtime_denominator = denominator
                    module._step_metric = None
                    # A failed iteration may never reach the optimizer hook.
                    # A new runtime step must discard its partial metrics.
                    module.instance_acc.zero_()

    def inject(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        """Inject the aux-loss gradient on ``carrier`` and accumulate the metric.

        Args:
            raw_sum: unnormalized per-microbatch loss value (differentiable).
            carrier: the tensor whose backward path carries the gradient.

        Returns:
            ``carrier`` unchanged.

        Raises:
            ValueError: if no step denominator has been set, which would make
                the loss's scale undefined.
        """
        denominator = self._runtime_denominator
        if denominator is None:
            denominator = AuxLoss._step_denominator
        if denominator is None:
            raise ValueError(
                "AuxLoss.set_step_denominator() must be called with the "
                "step's global routed-token count before the first forward."
            )
        # clamp_min(1): an all-padding step has zero routed tokens, and inf
        # here would poison the router gradients it is injected into
        # (upstream's guard, same line).
        metric_region = getattr(self, "_metric_region_fn", None)
        if metric_region is not None:
            result = metric_region(raw_sum, carrier=carrier, denominator=denominator)
            needs_tensor = getattr(self, "_metric_region_needs_tensor", None)
            if needs_tensor is not None:
                needs_tensor(result)
            return result
        return self._accumulate_and_inject(
            raw_sum, carrier=carrier, denominator=denominator
        )

    def _accumulate_and_inject(
        self,
        raw_sum: torch.Tensor,
        *,
        carrier: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / denominator.clamp_min(1)
        injected = raw_sum * (self.coeff * scale)
        # Accumulate the metric in the forward. The no_grad mask keeps the
        # buffer out of the autograd graph.
        if not in_checkpoint_recompute.get():
            with torch.no_grad():
                self.instance_acc.add_(raw_sum * scale)
        return AuxLossInjection.apply(carrier, injected)


def rollup_aux_loss_metrics(model_parts: Sequence[nn.Module]) -> None:
    """Snapshot per-instance metrics, populate standalone groups, and clear scratch.

    Optimizer step pre-hook. Each metric group is summed once per model part
    that holds an instance, so a group spread across pipeline stages ends up
    with one register holding the step's total.
    """
    AuxLoss.group_acc.clear()
    seen = set()
    for part in model_parts:
        for module in part.modules():
            if isinstance(module, AuxLoss) and id(module) not in seen:
                seen.add(id(module))
                module._step_metric = module.instance_acc.detach().clone()
                key = (module.reduce_mesh, module.metric_name)
                if key not in AuxLoss.group_acc:
                    AuxLoss.group_acc[key] = torch.zeros_like(module.instance_acc)
                AuxLoss.group_acc[key] += module.instance_acc
                module.instance_acc.zero_()


def collect_aux_loss_metrics(
    parallel_dims: ParallelTopology,
    *,
    model_parts: Sequence[nn.Module] | None = None,
) -> dict[str, float]:
    """Reduce the current step's ``group_acc`` registers for logging.

    Returns ``{metric_name}/mean`` per group, or ``{}`` when no auxiliary loss
    is configured. All ranks call this at log time. When model_parts are
    supplied, use only their snapshots and agree on metric keys across PP stages,
    including stages with no local loss. Standalone calls without parts retain
    the constructor registry; the trainer always supplies its current parts.
    """
    pp_mesh = parallel_dims.get_optional_mesh("pp")
    counts = dict(AuxLoss._group_counts)
    runtime_totals: dict[tuple[str, str], torch.Tensor] = {}
    if model_parts is not None:
        instances = {
            id(module): module
            for part in model_parts
            for module in part.modules()
            if isinstance(module, AuxLoss)
        }
        counts = Counter(
            (module.reduce_mesh, module.metric_name) for module in instances.values()
        )
        for module in instances.values():
            key = (module.reduce_mesh, module.metric_name)
            value = module._step_metric
            if value is None:
                continue
            if key not in runtime_totals:
                runtime_totals[key] = value.clone()
            else:
                runtime_totals[key].add_(value)
        if pp_mesh is not None:
            # Stage-local groups may differ, including a stage with none.
            # Agree on keys before issuing the per-group tensor collectives.
            stage_keys = [None] * pp_mesh.size()
            dist.all_gather_object(
                stage_keys, sorted(counts), group=pp_mesh.get_group()
            )
            counts = {key: counts.get(key, 0) for keys in stage_keys for key in keys}

    metrics = {}
    for key, count in sorted(counts.items()):
        mesh_name, tag = key
        total = (
            runtime_totals.get(key)
            if model_parts is not None
            else AuxLoss.group_acc.get(key)
        )
        if total is None:
            # A stage with no local instance still joins every collective.
            total = torch.zeros((), dtype=torch.float32, device=device_type)
        reduce_mesh = parallel_dims.get_optional_mesh(mesh_name)
        for mesh in (reduce_mesh, pp_mesh):
            if mesh is None:
                continue
            # Sum: each coordinate contributes its own data, and every layer
            # lives on exactly one pipeline stage, so summing over the reduce
            # mesh and the stages counts every layer once. Dividing by the
            # current runtime instance count below then
            # gives the mean over all layers.
            total = all_reduce(total, reduceOp="sum", group=mesh)
        if model_parts is not None:
            # Constructor counts include previously built models in the same
            # process. Count the current runtime's distinct local instances,
            # then combine pipeline stages (data peers replicate each layer).
            layer_count = torch.tensor(count, dtype=torch.int64, device=total.device)
            if pp_mesh is not None:
                layer_count = all_reduce(layer_count, reduceOp="sum", group=pp_mesh)
            count = int(layer_count.item())
            if count == 0:
                continue
        metrics[f"{tag}/mean"] = float(total.item()) / count
    return metrics


def register_aux_loss_metrics_hook(
    optimizer: torch.optim.Optimizer,
    *,
    model_parts: Sequence[nn.Module],
) -> None:
    """Register the step pre-hook that rolls ``instance_acc`` into ``group_acc``.

    This roll-up is local; distributed metric reduction happens later in
    ``collect_aux_loss_metrics``.
    """
    tracked_parts = tuple(model_parts)
    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: rollup_aux_loss_metrics(tracked_parts)
    )
