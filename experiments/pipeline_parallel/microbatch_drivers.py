"""Adapt pre-split microbatches to the installed Torch pipeline schedule API.

Own version-specific calls, loss normalization state and stage initialization
here so training orchestration does not depend on private schedule fields.
Both drivers collect losses without returning final-stage activations.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from functools import wraps
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any
from weakref import WeakSet

import torch

from experiment.pipeline_parallel.capabilities import (
    has as capability,
)
from llmtuner.components.loss_contract import validate_normalization_tokens
from llmtuner.errors import EnvironmentUnsupportedError

__all__ = ["run_eval_microbatches", "run_microbatches", "schedule_stages"]

_adapted_rng_methods: WeakSet[FunctionType] = WeakSet()


def bind_stage_rng(schedule: Any) -> None:
    """Give upstream metadata inference a private device-aware torch namespace.

    Preserve upstream control flow without changing process-global fork_rng.
    Older schedules without this initialization method require no adapter.
    """
    method = getattr(schedule, "_initialize_pp_stages", None)
    if method is None:
        return
    function = getattr(method, "__func__", None)
    if isinstance(function, FunctionType) and function in _adapted_rng_methods:
        return
    if not isinstance(function, FunctionType):
        raise EnvironmentUnsupportedError(
            "Pipeline RNG adapter requires a Python metadata initialization method"
        )
    if "torch" not in function.__code__.co_names:
        return
    original = torch.random.fork_rng

    @contextlib.contextmanager
    def fork_rng(devices: Any = None, **kwargs: Any) -> Iterator[None]:
        if "device_type" not in kwargs and devices:
            types = {torch.device(device).type for device in devices}
            if len(types) != 1:
                raise ValueError("Pipeline metadata requires one accelerator type")
            kwargs["device_type"] = types.pop()
        with original(devices=devices, **kwargs):
            yield

    namespace = dict(function.__globals__)
    torch_namespace = dict(vars(torch))
    random_namespace = dict(vars(torch.random))
    random_namespace["fork_rng"] = fork_rng
    torch_namespace["random"] = SimpleNamespace(**random_namespace)
    namespace["torch"] = SimpleNamespace(**torch_namespace)
    bound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    wraps(function)(bound)
    _adapted_rng_methods.add(bound)
    schedule._initialize_pp_stages = MethodType(bound, schedule)


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


def validate_pipeline_schedule(schedule: Any, stages: list[Any]) -> None:
    """Check a factory's local driver contract without invoking model execution."""
    actual = schedule_stages(schedule)
    if len(actual) != len(stages) or any(
        left is not right for left, right in zip(actual, stages, strict=True)
    ):
        raise ValueError("Pipeline schedule must retain the assembled stage order")
    if getattr(schedule, "_has_backward", None) is not True:
        raise ValueError("Training pipeline schedule must enable backward")
    methods = ["_step_microbatches"]
    if capability("pipelining_microbatch_drivers"):
        methods.append("eval")
    for name in methods:
        if not callable(getattr(schedule, name, None)):
            raise TypeError(f"Pipeline schedule requires callable {name}")
    loss = getattr(schedule, "_loss_fn", None)
    if not callable(loss):
        raise TypeError(
            "Pipeline schedule requires a callable normalization-aware loss"
        )
    validate_normalization_tokens(getattr(loss, "global_valid_tokens", None))


@contextlib.contextmanager
def _loss_normalization(
    schedule: Any, normalization_tokens: torch.Tensor
) -> Iterator[None]:
    """Keep a step's denominator from leaking into later calls or retaining it."""
    validate_normalization_tokens(normalization_tokens)
    loss = schedule._loss_fn
    previous = loss.global_valid_tokens
    loss.global_valid_tokens = normalization_tokens
    try:
        yield
    finally:
        loss.global_valid_tokens = previous


def run_microbatches(
    schedule: Any,
    *,
    args: list[Any] | None,
    kwargs: list[dict[str, Any]],
    targets: list[Any] | None,
    losses: list[Any] | None,
    normalization_tokens: torch.Tensor,
) -> None:
    """Run one training microbatch set with the installed schedule driver.

    Mirrors 2.9's ``step()`` setup exactly: the schedule's ``_has_backward``
    picks the stages' mode, and it is left as the caller set it (the training
    path runs with backward enabled).
    """
    with _loss_normalization(schedule, normalization_tokens):
        for stage in schedule_stages(schedule):
            stage.has_backward = schedule._has_backward
            stage.clear_runtime_states()
        bind_stage_rng(schedule)
        if capability("pipelining_microbatch_drivers"):
            schedule._step_microbatches(
                args, kwargs, targets, losses, return_outputs=False
            )
        else:
            schedule._step_microbatches(args, kwargs, targets, losses)


def run_eval_microbatches(
    schedule: Any,
    *,
    args: list[Any] | None,
    kwargs: list[dict[str, Any]],
    targets: list[Any] | None,
    losses: list[Any] | None,
    normalization_tokens: torch.Tensor,
) -> None:
    """Run one forward-only microbatch set with the installed schedule driver.

    Mirrors 2.9's ``eval()``: ``_has_backward`` decides the mode, so it is
    forced off for the duration and restored afterwards, and the stages are
    primed with the same flag so the schedule never takes a backward step.
    """
    with _loss_normalization(schedule, normalization_tokens):
        stages = schedule_stages(schedule)
        stage_modes = [stage.has_backward for stage in stages]
        previous = schedule._has_backward
        try:
            if capability("pipelining_microbatch_drivers"):
                schedule.eval(
                    arg_mbs=args, kwarg_mbs=kwargs, target_mbs=targets, losses=losses
                )
                return
            schedule._has_backward = False
            for stage in stages:
                stage.has_backward = False
                stage.clear_runtime_states()
            bind_stage_rng(schedule)
            schedule._step_microbatches(args, kwargs, targets, losses)
        finally:
            schedule._has_backward = previous
            for stage, mode in zip(stages, stage_modes, strict=True):
                stage.has_backward = mode
