"""Training traces and allocator snapshots, adapted from torchtitan.

Enter ``Profiler`` around the training loop and call ``step`` after each
iteration. It owns trace collection, periodic snapshots, and an OOM snapshot
on exit. Trace activities follow the resolved device: CPU plus CUDA, XPU or
NPU when selected; other devices collect CPU traces. Allocator history uses
``utils/monitoring`` and unavailable history disables snapshot output.

Upstream sources and adaptation details are in ``docs/llmtuner_upstream_map.md``.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakSet

import torch

if TYPE_CHECKING:
    from ..config.observability_config import ProfilerConfig

from ..accelerator.device import device_type
from ..utils.logging import get_distributed_rank, get_logger
from ..utils.monitoring import (
    read_memory_snapshot,
    record_memory_history,
    stop_memory_history,
)

logger = get_logger(__name__)

__all__ = ["MemoryProfiler", "Profiler"]

# File layout within the directories selected by ProfilerConfig.
PROFILE_ITER_DIR = "iteration_{step}"
PROFILE_FILE = "rank{rank}_trace.json.gz"  # .../{PROFILE_FILE}

MEMORY_STEP_DIR = "step_{step:012d}"
MEMORY_EXIT_DIR = "step_{step:012d}_exit"  # the OOM variant of the same
MEMORY_FILE = "{rank:06d}_step_{step}.pickle"

# A snapshot needs only the Python frames to be useful, and protocol 4 is what
# pytorch.org/memory_viz reads. See MemoryProfiler.step.
_SNAPSHOT_PICKLE_PROTOCOL = 4


class MemoryProfiler:
    """Writes periodic allocator snapshots while training runs.

    Built by :meth:`Profiler.build_memory_profiler` when snapshots are enabled.
    Call :meth:`step` once per iteration; the frequency is checked here rather
    than by the caller, so the loop does not have to know the policy.
    """

    _history_owners: ClassVar[WeakSet[MemoryProfiler]] = WeakSet()
    _history_lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        *,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        leaf_folder: str,
        rank: int,
        max_entries: int,
    ) -> None:
        for name, value, minimum in (
            ("step_num", step_num, 0),
            ("freq", freq, 1),
            ("rank", rank, 0),
            ("max_entries", max_entries, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self._max_entries = max_entries
        with self._history_lock:
            if any(owner._max_entries != max_entries for owner in self._history_owners):
                raise ValueError(
                    "Overlapping memory profilers require identical max_entries"
                )
            self._records_history = bool(self._history_owners) or record_memory_history(
                max_entries=max_entries
            )
            if self._records_history:
                self._history_owners.add(self)
        if not self._records_history:
            # Not an error: the device has no allocator history to keep. Said
            # once, at the point the run asked for snapshots, rather than
            # silently writing empty files for the rest of the run.
            logger.warning(
                "Memory snapshots are enabled but the %s allocator keeps no "
                "memory history; no snapshots will be written.",
                device_type,
            )
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def step(self, *, exit_ctx: bool = False) -> None:
        """Write a snapshot if this iteration calls for one.

        ``exit_ctx`` forces one regardless of the frequency, and names it for
        the *previous* step: the snapshot is taken while unwinding an exception,
        so the step counter has not advanced past the one that failed.

        On a device whose allocator keeps no history (CPU) there is nothing to
        write: returning here is what keeps a pickled ``None`` out of the
        snapshot directory on the OOM path, where the frequency check below is
        deliberately bypassed. Upstream writes the file unconditionally.
        """
        self.step_num += 1
        if not self._records_history:
            return
        if not exit_ctx and self.step_num % self.freq != 0:
            return

        if exit_ctx:
            curr_step = self.step_num - 1
            dir_name = MEMORY_EXIT_DIR.format(step=curr_step)
        else:
            curr_step = self.step_num
            dir_name = MEMORY_STEP_DIR.format(step=curr_step)

        directory = os.path.join(self._snapshot_dir, dir_name, self._leaf_folder)
        os.makedirs(directory, exist_ok=True)

        logger.info("Dumping memory snapshot at step %d", curr_step)
        begin = time.monotonic()
        output_file = os.path.join(
            directory, MEMORY_FILE.format(rank=self._rank, step=curr_step)
        )
        temporary_path = None
        try:
            with NamedTemporaryFile(
                dir=directory, suffix=".tmp", delete=False
            ) as output:
                temporary_path = output.name
                # Protocol 4 is understood by memory_viz's JS parser.
                pickle.dump(
                    read_memory_snapshot(), output, protocol=_SNAPSHOT_PICKLE_PROTOCOL
                )
            os.replace(temporary_path, output_file)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        logger.info(
            "Finished dumping memory snapshot in %.2f seconds", time.monotonic() - begin
        )

    def close(self) -> None:
        """Stop the allocator history started by this profiler, once."""
        with self._history_lock:
            if self._records_history:
                if not any(owner is not self for owner in self._history_owners):
                    stop_memory_history()
                self._history_owners.discard(self)
                self._records_history = False


class Profiler:
    """Owns the trace and memory-snapshot lifecycle for a training run.

    Example::

        with Profiler(config, global_step=step, base_folder=folder) as prof:
            for step in training_loop:
                ...
                prof.step()

    Args:
        config: a :class:`ProfilerConfig` instance.
        global_step: the step profiling begins at. When resuming from a
            checkpoint this is the loaded step, so trace directories are named
            for where the run actually is (``iteration_100``, not
            ``iteration_0``) and the snapshot frequency stays aligned with it.
        base_folder: root directory for both trace and snapshot output.
    """

    def __init__(
        self,
        config: ProfilerConfig,
        *,
        global_step: int = 0,
        base_folder: str = "",
    ) -> None:
        self._config = config
        self._global_step = global_step
        self._base_folder = base_folder
        self.torch_profiler: torch.profiler.profile | None = None
        self.memory_profiler: MemoryProfiler | None = None
        self._entered = False
        self._used = False

    def __enter__(self) -> Profiler:
        if self._entered:
            raise RuntimeError("Profiler is already active")
        if self._used:
            raise RuntimeError("Profiler has finished; create a new instance")
        self._entered = True
        try:
            self.torch_profiler = self.build_torch_profiler()
            self.memory_profiler = self.build_memory_profiler()
        except BaseException:
            # __exit__ is not called by `with` when __enter__ fails.
            self.__exit__(*sys.exc_info())
            raise
        self._used = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._entered = False
        first_error: BaseException | None = None
        trace, self.torch_profiler = self.torch_profiler, None
        memory, self.memory_profiler = self.memory_profiler, None
        if trace is not None:
            try:
                trace.__exit__(exc_type, exc_val, exc_tb)
            except BaseException as error:
                first_error = error
                try:
                    logger.exception("Failed to stop trace profiler")
                except BaseException:
                    pass  # Preserve the primary error and continue resource cleanup.
        if memory is not None and caused_by_oom(exc_val):
            try:
                # Attempt the OOM snapshot even when trace export failed.
                memory.step(exit_ctx=True)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                try:
                    logger.exception("Failed to write OOM memory snapshot")
                except BaseException:
                    pass  # Preserve the primary error and continue resource cleanup.
        if memory is not None:
            try:
                memory.close()
            except BaseException as error:
                # A failed backend shutdown may leave allocator recording active.
                # Retain its owner so callers can retry exit after the failure.
                self.memory_profiler = memory
                if first_error is None:
                    first_error = error
                try:
                    logger.exception("Failed to stop allocator memory history")
                except BaseException:
                    pass  # Preserve the primary error and continue resource cleanup.
        if exc_val is None and first_error is not None:
            raise first_error
        return False

    def step(self) -> None:
        """Advance every active profiler by one training iteration."""
        if self.torch_profiler is not None:
            self.torch_profiler.step()
        if self.memory_profiler is not None:
            self.memory_profiler.step()

    def build_torch_profiler(self) -> torch.profiler.profile | None:
        """Create, start, and return the torch profiler, or ``None`` if off.

        Entered here rather than by the caller so the returned handle is
        already collecting; :meth:`__exit__` is what stops it.
        """
        cfg = self._config
        if not cfg.enable_profiling:
            return None

        trace_dir = os.path.join(self._base_folder, cfg.save_traces_folder)
        rank = get_distributed_rank()
        start_step = self._global_step
        last_trace_step: int | None = None

        def trace_handler(prof):
            nonlocal last_trace_step
            global_step = start_step + prof.step_num
            directory_name = PROFILE_ITER_DIR.format(step=global_step)
            if global_step == last_trace_step:
                # Exiting an active cycle can export again without step_num
                # advancing. Keep both the completed cycle and its exit tail.
                directory_name += "_exit"
            directory = os.path.join(trace_dir, directory_name)
            os.makedirs(directory, exist_ok=True)

            logger.info("Dumping profiler traces at step %d", global_step)
            begin = time.monotonic()
            filename = PROFILE_FILE.format(rank=rank)
            if device_type == "npu":
                filename = filename.removesuffix(".gz")
            prof.export_chrome_trace(os.path.join(directory, filename))
            last_trace_step = global_step
            logger.info(
                "Finished dumping profiler traces in %.2f seconds",
                time.monotonic() - begin,
            )

        logger.info("Profiling active. Traces will be saved at %s", trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        # Left out of the schedule rather than passed as None, because
        # ``torch.profiler.schedule`` treats an explicit None differently from
        # an absent argument.
        optional_params = {
            key: value
            for key, value in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if value is not None
        }

        # What remains of the cycle after warmup and active. The config's
        # validation guarantees this is not negative.
        wait = cfg.profile_freq - (cfg.profiler_active + cfg.profiler_warmup)
        on_trace_ready = trace_handler
        profiler_api = torch.profiler
        if device_type == "npu":
            from torch_npu import profiler as profiler_api

            raw_trace_handler = profiler_api.tensorboard_trace_handler(
                os.path.join(trace_dir, "raw"), analyse_flag=False
            )

            def on_trace_ready(prof):
                raw_trace_handler(prof)
                trace_handler(prof)

        activities = [profiler_api.ProfilerActivity.CPU]
        if device_type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif device_type == "xpu":
            # Upstream's second branch. The device is resolved once in
            # ``accelerator/device.py``, where ``xpu`` is one of five
            # accelerators, so following the resolved device is what keeps a
            # trace from being CPU-only on a device the run actually uses.
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif device_type == "npu":
            activities.append(profiler_api.ProfilerActivity.NPU)

        schedule = profiler_api.schedule(
            wait=wait,
            warmup=cfg.profiler_warmup,
            active=cfg.profiler_active,
            **optional_params,
        )
        torch_profiler = profiler_api.profile(
            activities=activities,
            schedule=lambda step: schedule(start_step + step),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
        )
        try:
            torch_profiler.__enter__()
        except BaseException:
            # Starting a trace can acquire resources before raising. The
            # outer context has not received this handle yet, so it cannot
            # release it. Preserve the original failure if stopping also fails.
            try:
                torch_profiler.__exit__(*sys.exc_info())
            except BaseException:
                try:
                    logger.exception("Failed to stop partially started trace profiler")
                except BaseException:
                    pass  # Preserve the primary error and continue resource cleanup.
            raise
        return torch_profiler

    def build_memory_profiler(self) -> MemoryProfiler | None:
        """Create a :class:`MemoryProfiler`, or ``None`` if snapshots are off.

        Its constructor starts recording immediately; that is why this is built
        on entry rather than at the first dump.
        """
        cfg = self._config
        if not cfg.enable_memory_snapshot:
            return None

        freq = (
            cfg.profile_freq
            if cfg.memory_snapshot_freq is None
            else cfg.memory_snapshot_freq
        )
        if freq <= 0:
            raise ValueError(
                "Memory snapshot frequency must be greater than zero; set "
                "profiler.memory_snapshot_freq or profiler.profile_freq to a "
                "positive value."
            )

        snapshot_dir = os.path.join(self._base_folder, cfg.save_memory_snapshot_folder)
        os.makedirs(snapshot_dir, exist_ok=True)
        rank = get_distributed_rank()

        logger.info(
            "Memory profiler active. Snapshots will be saved at %s", snapshot_dir
        )
        return MemoryProfiler(
            step_num=self._global_step,
            freq=freq,
            snapshot_dir=snapshot_dir,
            leaf_folder="",
            rank=rank,
            max_entries=cfg.memory_snapshot_max_entries,
        )


def caused_by_oom(exc: BaseException | None) -> bool:
    """Whether an ``OutOfMemoryError`` appears anywhere in the cause chain.

    The chain matters rather than just the exception type: pipeline parallelism
    does not re-raise the OOM it catches, it wraps the failure in a plain
    ``RuntimeError`` to attach the stage's shapes, which hides the type.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
