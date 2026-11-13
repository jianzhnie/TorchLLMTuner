"""Synchronized training-only observations, independent of log-window cadence.

All ranks call begin/end for every successful optimizer step. Timing includes
loader and training work, but excludes reporting, checkpointing and validation.
Device fences and small reductions are intentional measurement overhead.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from ...accelerator.collectives import synchronize_local_errors
from ...components.metrics import compute_utilization, model_tflops
from ...models.common.moe.experts import RoutedExperts


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class DispatchTimer:
    """Optional current-stream events; covers forward/replay dispatch only."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []
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
            start, end = (
                backend.Event(enable_timing=True),
                backend.Event(enable_timing=True),
            )
            start.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((start, end))

    def consume_seconds(self) -> float:
        result = self.cpu_seconds
        for start, end in self.events:
            end.synchronize()
            result += start.elapsed_time(end) / 1000
        self.events.clear()
        self.cpu_seconds = 0.0
        return result


def install_dispatch_timers(
    model_parts: Iterable[torch.nn.Module],
) -> list[DispatchTimer]:
    """Observe each native dispatch boundary once, preserving existing timers.

    Dedicated fused implementations do not execute this boundary. Do not attach
    a timer to subclasses and report a misleading zero. Validate ownership before
    installation so a custom observer is never silently replaced.
    """
    targets = dict.fromkeys(
        module
        for part in model_parts
        for module in part.modules()
        if type(module) is RoutedExperts
    )
    bindings = []
    for module in targets:
        observer = module.dispatch_observer
        if observer is not None and not isinstance(observer, DispatchTimer):
            raise ValueError("Native expert dispatch already has a different observer")
        bindings.append((module, observer if observer is not None else DispatchTimer()))
    for module, timer in bindings:
        module.dispatch_observer = timer
    return list(dict.fromkeys(timer for _, timer in bindings))


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
        # counters and peak-FLOPs denominator, including PP/TP replicas.
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
        compute = compute_utilization(
            tokens_per_second=tps,
            flops_per_token=self.metrics.num_flops_per_token,
            peak_flops_per_device=self.metrics.gpu_peak_flops,
            world_size=world,
        )
        result = {
            "train/slowest_rank_step_seconds": seconds,
            "train/global_tokens": float(tokens),
            "train/global_valid_targets": float(valid),
            "train/global_tps": tps,
            "train/tps_per_device": tps / world,
            "train/valid_target_tps_per_device": valid / seconds / world,
            "train/valid_target_fraction": valid / tokens if tokens else 0.0,
            "train/data_loading_rank_max_seconds": loading,
            "train/data_loading_rank_max_percent": 100 * loading / seconds,
            "train/tflops_per_device": model_tflops(
                self.metrics.num_flops_per_token, tps / world
            ),
        }
        if compute["utilization_percent"] is not None:
            result["train/mfu_percent"] = compute["utilization_percent"]
        result.update(
            self.record_steady_step(seconds=seconds, tokens=tokens, world=world)
        )
        self.metrics.latest_training_observations = result
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
        result = {
            "steady/samples": float(len(self.samples)),
            "steady/step_p50_seconds": percentile(durations, 0.5),
            "steady/step_p90_seconds": percentile(durations, 0.9),
            "steady/tps_per_device": steady_tps / world,
            "steady/tflops_per_device": model_tflops(
                self.metrics.num_flops_per_token, steady_tps / world
            ),
        }
        steady_compute = compute_utilization(
            tokens_per_second=steady_tps,
            flops_per_token=self.metrics.num_flops_per_token,
            peak_flops_per_device=self.metrics.gpu_peak_flops,
            world_size=world,
        )
        if steady_compute["utilization_percent"] is not None:
            result["steady/mfu_percent"] = steady_compute["utilization_percent"]
        return result
