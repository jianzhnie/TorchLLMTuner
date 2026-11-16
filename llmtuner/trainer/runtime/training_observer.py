"""Synchronized training-only observations, independent of log-window cadence.

All ranks call begin/end for every successful optimizer step. Timing includes
loader and training work, but excludes reporting, checkpointing and validation.
Device fences and small reductions are intentional measurement overhead.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ...accelerator.collectives import synchronize_local_errors
from ...components.metrics import compute_utilization, model_tflops
from ...models.common.moe.experts import DispatchObservation
from ...utils.logging import get_logger

logger = get_logger(__name__)


def percentile(values: list[float], fraction: float) -> float:
    return _sorted_percentile(sorted(values), fraction)


def _sorted_percentile(ordered: list[float], fraction: float) -> float:
    """Interpolate a percentile from an already sorted observation window."""
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class DispatchTimer:
    """Optional current-stream events; covers forward/replay dispatch only."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []
        self._failed_record_streams: set[Any] = set()
        self.cpu_seconds = 0.0

    @contextmanager
    def measure(self, device: torch.device) -> Iterator[None]:
        if device.type == "cpu":
            started = time.perf_counter()
            try:
                yield
            finally:
                self.cpu_seconds += time.perf_counter() - started
        else:
            backend = torch.get_device_module(device)
            stream = backend.current_stream(device)
            start, end = (
                backend.Event(enable_timing=True),
                backend.Event(enable_timing=True),
            )
            try:
                start.record(stream)
            except BaseException:
                self._failed_record_streams.add(stream)
                raise
            try:
                yield
            finally:
                dispatch_error = sys.exc_info()[1]
                try:
                    end.record(stream)
                except BaseException:
                    # No end event can fence the submitted work. Borrow the
                    # originating stream until close successfully drains it.
                    self._failed_record_streams.add(stream)
                    if dispatch_error is None:
                        raise
                    try:
                        logger.exception("Dispatch timing failed after a model error")
                    except BaseException:
                        pass  # Preserve the active model exception.
                else:
                    self.events.append((start, end))

    def consume_seconds(self) -> float:
        result = self.cpu_seconds
        for start, end in self.events:
            end.synchronize()
            result += start.elapsed_time(end) / 1000
        self.events.clear()
        self.cpu_seconds = 0.0
        return result

    def close(self) -> None:
        """Drain pending observations, retaining failed event waits for retry."""
        failed = []
        first_error: BaseException | None = None
        for stream in tuple(self._failed_record_streams):
            try:
                stream.synchronize()
                self._failed_record_streams.remove(stream)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        for start, end in self.events:
            try:
                end.synchronize()
            except BaseException as error:
                failed.append((start, end))
                if first_error is None:
                    first_error = error
        self.events = failed
        self.cpu_seconds = 0.0
        if first_error is not None:
            raise first_error


@dataclass(slots=True)
class DispatchTimerResources:
    """Own installed model bindings and drain them before detaching observers."""

    bindings: tuple[tuple[DispatchObservation, DispatchTimer], ...]

    @property
    def timers(self) -> tuple[DispatchTimer, ...]:
        return tuple(dict.fromkeys(timer for _, timer in self.bindings))

    def close(self) -> None:
        first_error: BaseException | None = None
        failed = set()
        for timer in reversed(self.timers):
            try:
                timer.close()
            except BaseException as error:
                failed.add(timer)
                if first_error is None:
                    first_error = error
        pending = []
        for boundary, timer in self.bindings:
            if timer in failed:
                pending.append((boundary, timer))
            elif boundary.observer is timer:
                boundary.observer = None
        self.bindings = tuple(pending)
        if first_error is not None:
            raise first_error


def install_dispatch_timers(
    model_parts: Iterable[torch.nn.Module],
) -> DispatchTimerResources:
    """Acquire exclusive ownership of declared model observation boundaries.

    Models without a forward observation site do not declare this capability.
    Validate every declaration before mutation so custom observers are never
    silently replaced and partially installed timers cannot escape on failure.
    Existing owners must close before another installation can acquire a boundary.
    """
    targets = {}
    for part in model_parts:
        for module in part.modules():
            boundary = getattr(module, "dispatch_observation", None)
            if boundary is None:
                continue
            if not isinstance(boundary, DispatchObservation):
                raise TypeError("dispatch_observation must be DispatchObservation")
            targets[id(boundary)] = boundary
    bindings = []
    for boundary in targets.values():
        if boundary.observer is not None:
            raise ValueError(
                "Model dispatch boundary already has an observer; "
                "close its owner before installing dispatch timers"
            )
        bindings.append((boundary, DispatchTimer()))
    for boundary, timer in bindings:
        boundary.observer = timer
    return DispatchTimerResources(tuple(bindings))


class TrainingObserver:
    """Bounded rolling statistics; warmup restarts on independent resume."""

    def __init__(
        self,
        metrics: Any,
        device: torch.device,
        *,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        # This group must span all ranks represented by the global token
        # counters and peak-FLOPs denominator, including TP/CP replicas.
        self.group = group
        self.metrics = metrics
        self.device = device
        self.completed = 0
        self.samples: deque[tuple[float, int]] = deque(
            maxlen=metrics.config.steady_window_steps
        )

    def synchronize(self) -> None:
        if self.device.type != "cpu":
            torch.get_device_module(self.device).synchronize(self.device)

    def begin(self) -> None:
        with synchronize_local_errors("Training observation start", group=self.group):
            self.synchronize()
            self.tokens = self.metrics.ntokens_since_last_log
            self.valid = self.metrics.valid_tokens_since_last_log
            self.loading_index = len(self.metrics.data_loading_times)
            self.started = time.perf_counter()

    def end(self) -> dict[str, float]:
        with synchronize_local_errors(
            "Training observation snapshot", group=self.group
        ):
            self.synchronize()
            elapsed = time.perf_counter() - self.started
            times = torch.tensor(
                [elapsed, sum(self.metrics.data_loading_times[self.loading_index :])],
                device=self.device,
                dtype=torch.float32,
            )
            counts = torch.tensor(
                [
                    self.metrics.ntokens_since_last_log - self.tokens,
                    self.metrics.valid_tokens_since_last_log - self.valid,
                ],
                device=self.device,
                dtype=torch.int64,
            )
        world = 1
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size(self.group)
            dist.all_reduce(times, op=dist.ReduceOp.MAX, group=self.group)
            dist.all_reduce(counts, group=self.group)
        seconds, loading = times.tolist()
        replication = self.metrics._non_data_parallel_size
        tokens, valid = (int(value) // replication for value in counts.tolist())
        tps = tokens / seconds
        result = {
            "train/slowest_rank_step_seconds": seconds,
            "train/global_tokens": float(tokens),
            "train/global_valid_targets": float(valid),
            "train/global_tps": tps,
            "train/valid_target_tps_per_device": valid / seconds / world,
            "train/valid_target_fraction": valid / tokens if tokens else 0.0,
            "train/data_loading_rank_max_seconds": loading,
            "train/data_loading_rank_max_percent": 100 * loading / seconds,
        }
        result.update(self._throughput_metrics("train", tps=tps, world=world))
        result.update(
            self.record_steady_step(seconds=seconds, tokens=tokens, world=world)
        )
        self.metrics.latest_training_observations = result
        return result

    def _throughput_metrics(
        self, prefix: str, *, tps: float, world: int
    ) -> dict[str, float]:
        """Report per-device throughput and utilization from global token rates."""
        result = {
            f"{prefix}/tps_per_device": tps / world,
            f"{prefix}/tflops_per_device": model_tflops(
                self.metrics.num_flops_per_token, tps / world
            ),
        }
        compute = compute_utilization(
            tokens_per_second=tps,
            flops_per_token=self.metrics.num_flops_per_token,
            peak_flops_per_device=self.metrics.gpu_peak_flops,
            world_size=world,
        )
        if compute["utilization_percent"] is not None:
            result[f"{prefix}/mfu_percent"] = compute["utilization_percent"]
        return result

    def record_steady_step(
        self, *, seconds: float, tokens: int, world: int
    ) -> dict[str, float]:
        """Record a completed step and summarize the bounded post-warmup window."""
        self.completed += 1
        if self.completed > self.metrics.config.metrics_warmup_steps:
            self.samples.append((seconds, tokens))
        if not self.samples:
            return {}

        durations = [sample[0] for sample in self.samples]
        steady_tps = sum(sample[1] for sample in self.samples) / sum(durations)
        durations.sort()
        result = {
            "steady/samples": float(len(self.samples)),
            "steady/step_p50_seconds": _sorted_percentile(durations, 0.5),
            "steady/step_p90_seconds": _sorted_percentile(durations, 0.9),
        }
        result.update(
            self._throughput_metrics("steady", tps=steady_tps, world=world)
        )
        return result
