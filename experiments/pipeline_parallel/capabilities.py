"""Probe the optional native PP microbatch drivers used by this experiment."""

from __future__ import annotations

import inspect
from functools import cache

try:
    import torch.distributed.pipelining.schedules as pipeline_schedules
except ImportError:
    pipeline_schedules = None

__all__ = ["has_microbatch_drivers"]


@cache
def has_microbatch_drivers() -> bool:
    """Whether torch's pipeline schedules carry the microbatch eval driver.

    Attribute presence is not enough: torch 2.9 shipped ``eval(*args, target,
    losses, **kwargs)`` -- the microbatch keyword form (``arg_mbs`` /
    ``kwarg_mbs`` / ``target_mbs``) would be swallowed by ``**kwargs`` and
    misrouted into the model kwargs. Probe the signature instead. The training
    seam's ``_step_microbatches(..., return_outputs=...)`` arrived in the same
    window, so both are checked.
    """
    schedule = getattr(pipeline_schedules, "_PipelineSchedule", None)
    evaluate = getattr(schedule, "eval", None)
    if evaluate is None:
        return False
    eval_params = inspect.signature(evaluate).parameters
    if "arg_mbs" not in eval_params:
        return False
    step = getattr(schedule, "_step_microbatches", None)
    if step is None:
        return False
    return "return_outputs" in inspect.signature(step).parameters
