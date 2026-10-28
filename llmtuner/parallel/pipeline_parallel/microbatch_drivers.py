"""Drive a torch pipeline schedule one already-split microbatch set at a time.

torch 2.10 added ``_PipelineSchedule.eval(arg_mbs=...)`` and
``_step_microbatches(..., return_outputs=...)``. torch 2.9 (the Ascend /
``torch_npu`` stack) has neither: its ``step``/``eval`` take *whole-batch*
inputs and declare ``**kwargs``, so microbatch keywords would be swallowed and
forwarded into the model call rather than routed to the schedule. That is a
silent misroute, not a failure, which is why the capability exists and why this
module exists next to it.

What ``step()`` does before handing off to ``_step_microbatches`` is a handful
of lines over ``schedule._stage(s)`` -- set ``has_backward``, clear the stages'
runtime state -- and that sequence is the same in 2.9 and 2.10. Reproducing it
lets the older build run PP instead of refusing to start.

Neither driver returns outputs, matching the ``return_outputs=False`` the 2.10
driver is called with: llmtuner's train and validation paths consume the
per-microbatch losses, never the last stage's activations.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

__all__ = ["run_eval_microbatches", "run_microbatches", "schedule_stages"]


@contextlib.contextmanager
def _fork_rng_with_stage_devices() -> Iterator[None]:
    """Work around torch 2.12 forking pipeline-init RNG on the wrong device.

    ``_initialize_pp_stages`` calls ``torch.random.fork_rng(devices=...)``
    without ``device_type``, so the ``"cuda"`` default resolves to
    ``torch.cuda`` and raises on builds without CUDA whenever a stage lives on
    another accelerator. Derive the type from the devices handed in instead.
    """
    real_fork_rng = torch.random.fork_rng

    @contextlib.contextmanager
    def shim(devices: Any = None, **kwargs: Any) -> Iterator[None]:
        if "device_type" not in kwargs and devices:
            types = {torch.device(device).type for device in devices}
            if len(types) == 1:
                kwargs["device_type"] = types.pop()
        with real_fork_rng(devices=devices, **kwargs) as ctx:
            yield ctx

    torch.random.fork_rng = shim
    try:
        yield
    finally:
        torch.random.fork_rng = real_fork_rng


def schedule_stages(schedule: Any) -> list[Any]:
    """The stage objects a schedule drives.

    Interleaved schedules hold several (``_stages``); the plain ones hold one
    (``_stage``). Single source for that lookup, so the two drivers below and
    the newer-torch call site cannot disagree about which stages to prime.
    """
    stages = getattr(schedule, "_stages", None)
    if stages is None:
        stages = [schedule._stage]
    return list(stages)


def run_microbatches(
    schedule: Any,
    *,
    args: list[Any] | None,
    kwargs: list[dict[str, Any]],
    targets: list[Any] | None,
    losses: list[Any] | None,
) -> None:
    """Run one training microbatch set on a schedule without microbatch drivers.

    Mirrors 2.9's ``step()`` setup exactly: the schedule's ``_has_backward``
    picks the stages' mode, and it is left as the caller set it (the training
    path runs with backward enabled).
    """
    for stage in schedule_stages(schedule):
        stage.has_backward = schedule._has_backward
        stage.clear_runtime_states()
    with _fork_rng_with_stage_devices():
        schedule._step_microbatches(args, kwargs, targets, losses)


def run_eval_microbatches(
    schedule: Any,
    *,
    args: list[Any] | None,
    kwargs: list[dict[str, Any]],
    targets: list[Any] | None,
    losses: list[Any] | None,
) -> None:
    """Run one forward-only microbatch set on a schedule without eval drivers.

    Mirrors 2.9's ``eval()``: ``_has_backward`` decides the mode, so it is
    forced off for the duration and restored afterwards, and the stages are
    primed with the same flag so the schedule never takes a backward step.
    """
    previous = schedule._has_backward
    schedule._has_backward = False
    try:
        for stage in schedule_stages(schedule):
            stage.has_backward = False
            stage.clear_runtime_states()
        with _fork_rng_with_stage_devices():
            schedule._step_microbatches(args, kwargs, targets, losses)
    finally:
        schedule._has_backward = previous
