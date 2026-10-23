"""Warmup-stable-decay schedules for the trainer's inner optimizers.

Adapted from torchtitan ``components/optimizer/lr_scheduler.py``. Each inner
optimizer owns one LambdaLR, sharing the WSD factor but retaining its own base
learning rates. The container steps them together and checkpoints their common
last_epoch; restore recomputes rates from optimizer initial_lr values.

Configuration lives in ``config/lr_scheduler_config.py``. This module owns the
curve, scheduler lifecycle, and learning-rate metrics.
"""

from __future__ import annotations

import functools
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ...config.field_checks_config import require_integer_at_least
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ...config.lr_scheduler_config import LRSchedulerConfig

logger = get_logger(__name__)

__all__ = ["build_lr_scheduler", "LRSchedulersContainer"]


def wsd_factor(
    current_step: int,
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    decay_type: str,
    min_lr_factor: float,
) -> float:
    """The multiplicative factor to apply to the base lr at ``current_step``.

    Warmup starts at ``1 / warmup_steps``; the stable phase uses 1. Decay
    approaches ``min_lr_factor`` within the configured schedule.
    """
    warmup_stable_steps = warmup_steps + stable_steps
    if current_step < warmup_steps:
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        assert warmup_steps != 0, "warmup_steps must not be zero to reach this branch"
        return float(current_step / warmup_steps)
    if current_step < warmup_stable_steps:
        return 1.0

    # 0-indexed step, hence + 1 adjustments
    current_step += 1
    assert decay_steps != 0, "decay_steps must not be zero to reach this branch"
    progress = float(current_step - warmup_stable_steps) / decay_steps

    if decay_type == "linear":
        factor = 1 - progress
    elif decay_type == "sqrt":
        factor = 1 - math.sqrt(progress)
    elif decay_type == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    return min_lr_factor + (1 - min_lr_factor) * factor


class LRSchedulersContainer(Stateful):
    """One ``LambdaLR`` per inner optimizer, stepped together.

    The training loop drives this like a single scheduler: ``step`` advances
    them all, ``get_metrics`` reports the lr of each.

    All of them share one lambda. They can still diverge, because a ``LambdaLR``
    scales its own optimizer's ``base_lrs`` -- so a run whose param groups carry
    different base learning rates gets the right curve per group without a
    second lambda.

    Args:
        optimizers: one scheduler is built per entry, in order.
        lr_lambda: maps ``last_epoch`` to the multiplicative lr factor.
        total_steps: the schedule's length, resolved by :func:`build_lr_scheduler`
            from the config and run length. Restore rejects checkpoint progress
            beyond this length.
    """

    def __init__(
        self,
        optimizers: list[Optimizer],
        *,
        lr_lambda: Callable[[int], float],
        total_steps: int,
    ) -> None:
        if not optimizers:
            raise ValueError(
                "LRSchedulersContainer needs at least one optimizer to schedule."
            )
        if len({id(optimizer) for optimizer in optimizers}) != len(optimizers):
            raise ValueError("Each optimizer must appear only once in the scheduler")
        self.total_steps = total_steps
        require_integer_at_least(self, "total_steps", group="lr_scheduler")
        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> Iterator[LambdaLR]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Advance the schedule by one step.

        Called *after* ``optimizer.step()``, so the lr the optimizer just used is
        the one this computes from the previous ``last_epoch`` -- which is why
        the first training step runs at ``lambda(0)``.
        """
        # Independent stepping of an exposed inner scheduler breaks the common
        # progress contract. Reject it before advancing any optimizer's rate.
        last_epoch = self.state_dict()["last_epoch"]
        if last_epoch >= self.total_steps:
            raise ValueError(
                "Cannot advance lr_scheduler beyond total_steps "
                f"({self.total_steps}); extend the schedule before continuing"
            )
        for scheduler in self.schedulers:
            scheduler.step()

    def get_metrics(self) -> dict[str, float]:
        """The current lr of each optimizer, keyed so several cannot collide.

        Upstream's keying: ``lr/<OptName>`` when there is only one scheduler of
        that optimizer type; a per-type index disambiguates several (PP gives
        every pipeline stage its own optimizer of the same class), and a
        param-group index goes last when one optimizer has several groups.
        """
        metrics: dict[str, float] = {}
        optimizer_counts = Counter(
            type(scheduler.optimizer).__name__ for scheduler in self.schedulers
        )
        optimizer_indices: defaultdict[str, int] = defaultdict(int)
        for scheduler in self.schedulers:
            optimizer_name = type(scheduler.optimizer).__name__
            optimizer_index = optimizer_indices[optimizer_name]
            optimizer_indices[optimizer_name] += 1
            last_lrs = scheduler.get_last_lr()
            prefix = f"lr/{optimizer_name}"
            if optimizer_counts[optimizer_name] > 1:
                prefix = f"{prefix}/{optimizer_index}"
            for group_index, value in enumerate(last_lrs):
                key = f"{prefix}/{group_index}" if len(last_lrs) > 1 else prefix
                metrics[key] = float(value)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        # One integer, not one per scheduler: every scheduler is stepped
        # together from the same lambda, so they share a step count by
        # construction. Storing it once is what lets a checkpoint survive
        # resharding to a different number of optimizers.
        last_epoch = self.schedulers[0].last_epoch
        if any(scheduler.last_epoch != last_epoch for scheduler in self.schedulers):
            raise ValueError(
                "Cannot checkpoint schedulers with different last_epoch values; "
                "advance them through LRSchedulersContainer.step() together"
            )
        return {"last_epoch": last_epoch}

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate saved progress without reading or changing optimizer groups."""
        if not isinstance(state_dict, dict):
            raise ValueError("scheduler checkpoint state must be a dictionary")
        if not state_dict:
            return
        last_epoch = state_dict["last_epoch"]
        if (
            not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < 0
        ):
            raise ValueError("scheduler last_epoch must be a nonnegative integer")
        if last_epoch > self.total_steps:
            raise ValueError(
                f"scheduler last_epoch ({last_epoch}) exceeds total_steps "
                f"({self.total_steps}); restore with a schedule covering the "
                "checkpoint's progress"
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ``last_epoch`` on every scheduler.

        ``LambdaLR`` is stateless apart from that -- the lr is a pure function of
        ``(last_epoch, base_lr)`` -- so ``_last_lr`` is recomputed rather than
        stored, per scheduler, from its optimizer groups' restored
        ``initial_lr`` values.
        """
        self.validate_state_dict(state_dict)
        if not state_dict:
            return
        last_epoch = state_dict["last_epoch"]
        restored = []
        for scheduler in self.schedulers:
            base_lrs = [
                group["initial_lr"] for group in scheduler.optimizer.param_groups
            ]
            last_lrs = [
                base_lr * lr_lambda(last_epoch)
                for base_lr, lr_lambda in zip(
                    base_lrs, scheduler.lr_lambdas, strict=True
                )
            ]
            restored.append((scheduler, base_lrs, last_lrs))
        for scheduler, base_lrs, last_lrs in restored:
            scheduler.base_lrs = base_lrs
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = last_lrs
            for group, learning_rate in zip(
                scheduler.optimizer.param_groups, last_lrs, strict=True
            ):
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"].fill_(learning_rate)
                else:
                    group["lr"] = learning_rate


def build_lr_scheduler(
    config: LRSchedulerConfig,
    *,
    optimizers: list[Optimizer],
    training_steps: int,
) -> LRSchedulersContainer:
    """Build the schedule a :class:`~llmtuner.config.LRSchedulerConfig`
    describes.

    ``training_steps`` is the run's actual length; the config's ``total_steps``
    overrides it for the curve only. The two are validated against each other
    rather than clamped: a schedule shorter than the run would put the last steps
    past its end, where the decay factor runs off the bottom of the curve and
    turns the learning rate negative -- which ascends the loss instead of
    failing.
    """
    if (
        not isinstance(training_steps, int)
        or isinstance(training_steps, bool)
        or training_steps < 1
    ):
        raise ValueError("training_steps must be a positive integer")
    total_steps = (
        config.total_steps if config.total_steps is not None else training_steps
    )
    if total_steps < training_steps:
        raise ValueError(
            f"lr_scheduler.total_steps ({total_steps}) is shorter than the run "
            f"({training_steps} steps). The decay would run past its end and "
            "produce a negative learning rate. Raise total_steps, or drop it "
            "to use the run length."
        )

    warmup_steps = config.warmup_steps
    if warmup_steps > total_steps:
        logger.warning(
            "lr_scheduler.warmup_steps (%d) exceeds total_steps (%d); "
            "clamping the warmup to the whole schedule.",
            warmup_steps,
            total_steps,
        )
        warmup_steps = total_steps

    decay_steps = round(total_steps * config.decay_ratio)
    if warmup_steps + decay_steps > total_steps:
        logger.warning(
            "lr_scheduler warmup (%d) + decay (%d) exceed total_steps (%d); "
            "shortening the decay to %d.",
            warmup_steps,
            decay_steps,
            total_steps,
            total_steps - warmup_steps,
        )
        decay_steps = total_steps - warmup_steps
    # The "+ 1" is a virtual final step. Without it the last real step would
    # land exactly at the end of the decay, where the factor is 0 (linear) --
    # an lr of zero on the final update. With no decay phase it makes the
    # stable region one step longer than the run, which is the point: every
    # real step falls inside it and the factor is a constant 1.0.
    stable_steps = total_steps + 1 - warmup_steps - decay_steps

    lr_lambda = functools.partial(
        wsd_factor,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        decay_type=config.decay_type,
        min_lr_factor=config.min_lr_factor,
    )
    return LRSchedulersContainer(
        optimizers, lr_lambda=lr_lambda, total_steps=total_steps
    )
