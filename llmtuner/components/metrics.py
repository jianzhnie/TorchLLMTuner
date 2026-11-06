"""Training metric sinks, device-memory sampling, and console reporting.

Adapted from torchtitan's ``observability/metrics.py``. Compute arithmetic lives
in ``compute_metrics``; synchronized per-step and steady-window observations live
in ``training_observer``. This module retains legacy log-window metrics for
existing consumers and publishes the synchronized observations when available.

Logging performs no collectives. The trainer synchronizes pipeline loss metrics
before reporting; writer ownership and console rank filtering are independent.
Unknown device peaks suppress MFU, and empty loading windows report zero.
Each external sink owns its resources and propagates configuration/runtime
failures instead of silently disabling tracking.
"""

from __future__ import annotations

import os
import time
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from ..accelerator.device import device_module, device_type
from ..utils.logging import get_distributed_rank, get_logger
from ..utils.monitoring import (
    Color,
    NoColor,
    colors_enabled,
    get_device_capacity_bytes,
    get_device_name,
    get_peak_flops,
)
from .compute_metrics import compute_utilization, model_tflops, token_throughput

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as error:
    SummaryWriter = None
    TENSORBOARD_IMPORT_ERROR = error

try:
    import wandb
except ImportError as error:
    wandb = None
    WANDB_IMPORT_ERROR = error

if TYPE_CHECKING:
    # Annotation-only. ``parallel_dims`` imports the training config, which
    # imports the checkpointer, so importing it for real here would close a
    # cycle back into the components package that only exists to name a type.
    from ..config.observability_config import MetricsConfig
    from ..parallel.topology import ParallelTopology

# llmtuner configures handlers per module, rather than on the root logger the
# way torchtitan does, so this has to be get_logger for the metrics lines to
# reach the console at all.
logger = get_logger(__name__)

__all__ = [
    "BaseLogger",
    "DeviceMemStats",
    "DeviceMemoryMonitor",
    "LoggerContainer",
    "MetricsProcessor",
    "TensorBoardLogger",
    "WandBLogger",
    "build_device_memory_monitor",
    "get_metrics_rank",
]


DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    """Rank-local device allocator sampling over a logging window.

    The trainer resets peak counters after every log. Reserved memory includes
    cached allocator blocks retained from earlier windows; resetting counters
    does not release those blocks. Use active and reserved metrics separately
    when comparing tensor residency and allocator capacity.
    No rank reduction occurs here: memory peaks, retries and OOM counts describe
    this process's current device, even when step timing uses the slowest rank.
    """

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type
        self.device_name = get_device_name()
        self.device_capacity = get_device_capacity_bytes()
        self.device_capacity_gib = self._to_gib(self.device_capacity)
        self.reset_peak_stats()
        if device_type != "cpu":
            # Clear startup residue from the caching allocator so the first
            # window's max_reserved baseline is the training loop's own.
            device_module.empty_cache()

    @staticmethod
    def _to_gib(memory_in_bytes: float) -> float:
        # GiB (gibibyte) is 1024^3, GB is 1000^3. Device memory is reported in
        # the former.
        return memory_in_bytes / (1024 * 1024 * 1024)

    def _to_pct(self, memory: float) -> float:
        if self.device_capacity == 0:
            return 0.0
        return 100 * memory / self.device_capacity

    def _device_module(self):
        if self.device_type != "cpu":
            return device_module
        return None

    def get_peak_stats(self) -> DeviceMemStats:
        device_module = self._device_module()
        if device_module is None:
            # CPU has no allocator to ask, so every field is a real zero rather
            # than a placeholder: there genuinely is no device memory in use.
            return DeviceMemStats(0.0, 0.0, 0.0, 0.0, 0, 0)

        device_info = device_module.memory_stats()
        max_active = device_info.get("active_bytes.all.peak", -1)
        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                "%d %s memory allocation retries.",
                num_retries,
                self.device_type.upper(),
            )
        if num_ooms > 0:
            logger.warning(
                "%d %s OOM errors thrown.", num_ooms, self.device_type.upper()
            )

        return DeviceMemStats(
            self._to_gib(max_active),
            self._to_pct(max_active),
            self._to_gib(max_reserved),
            self._to_pct(max_reserved),
            num_retries,
            num_ooms,
        )

    def get_current_allocated_gib(self) -> float:
        """Current tensor allocation, distinct from cached reserved memory."""
        device_module = self._device_module()
        return (
            0.0
            if device_module is None
            else self._to_gib(device_module.memory_allocated())
        )

    def reset_peak_stats(self) -> None:
        device_module = self._device_module()
        if device_module is not None:
            device_module.reset_peak_memory_stats()


def build_device_memory_monitor() -> DeviceMemoryMonitor:
    """Build a monitor and say which device it is watching."""
    monitor = DeviceMemoryMonitor(device_type)
    if device_type != "cpu":
        logger.info(
            "%s capacity: %s with %.2fGiB memory",
            device_type.upper(),
            monitor.device_name,
            monitor.device_capacity_gib,
        )
    return monitor


class BaseLogger:
    """A sink that takes metrics. Logging nothing is a valid implementation."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Writes scalars to a TensorBoard event directory."""

    def __init__(self, log_dir: str, tag: str | None = None):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requires the tensorboard package"
            ) from TENSORBOARD_IMPORT_ERROR

        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info("TensorBoard logging enabled. Logs will be saved at %s", log_dir)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            tag = key if self.tag is None else f"{self.tag}/{key}"
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Streams metrics to Weights & Biases.

    Experiment identity comes from the environment, matching torchtitan. That is
    deliberate for a launcher-driven tool: ``WANDB_RUN_ID`` and ``WANDB_RESUME_FROM``
    are how a requeued job rejoins its own run, and reading them from the
    environment means a launcher can set them without touching the config.
    Each logger creates and owns a separate run; closing it leaves existing
    runs alive. This requires WandB support for ``reinit="create_new"``.
    """

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        if wandb is None:
            raise ImportError(
                "WandB logging requires the wandb package"
            ) from WANDB_IMPORT_ERROR

        self.tag = tag

        os.makedirs(log_dir, exist_ok=True)
        self.run = wandb.init(
            reinit="create_new",
            entity=os.getenv("WANDB_TEAM"),
            project=os.getenv("WANDB_PROJECT", "llmtuner"),
            name=os.getenv("WANDB_RUN_NAME"),
            id=os.getenv("WANDB_RUN_ID"),
            notes=os.getenv("WANDB_RUN_NOTES"),
            tags=[
                tag.strip()
                for tag in os.getenv("WANDB_RUN_TAGS", "").split(",")
                if tag.strip()
            ]
            or None,
            group=os.getenv("WANDB_RUN_GROUP"),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE"),
            resume_from=os.getenv("WANDB_RESUME_FROM"),
            fork_from=os.getenv("WANDB_FORK_FROM"),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        prefixed = {
            key if self.tag is None else f"{self.tag}/{key}": value
            for key, value in metrics.items()
        }
        # commit=True: without it wandb treats an explicit step as a partial
        # record and holds it open, so a metric logged once per step can be
        # merged into (or overwritten by) the next one.
        if self.run is None:
            raise RuntimeError("WandBLogger is closed")
        self.run.log(prefixed, step=step, commit=True)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


class LoggerContainer(BaseLogger):
    """Fans one ``log`` call out to every enabled sink."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    def close(self) -> None:
        loggers, self._loggers = self._loggers, []
        first_error: BaseException | None = None
        for logger_instance in loggers:
            try:
                logger_instance.close()
            except BaseException as error:
                # Keep failed owners reachable so Trainer.close can retry;
                # successfully closed sinks must not be closed a second time.
                self._loggers.append(logger_instance)
                if first_error is None:
                    first_error = error
                else:
                    logger.exception("Additional metrics logger cleanup failure")
        if first_error is not None:
            raise first_error


def is_v_schedule(pp_schedule: str) -> bool:
    """V schedules put the last stage, which computes the loss, on pp rank 0.

    Class-based like upstream (``_is_v_schedule``), not a string check: both
    ScheduleZBVZeroBubble and ScheduleDualPipeV pair stage 0 with stage N-1 on
    rank 0, and a string test would miss the second.
    """
    return get_schedule_class(pp_schedule) in (
        ScheduleZBVZeroBubble,
        ScheduleDualPipeV,
    )


def get_metrics_rank(*, parallel_dims: ParallelTopology, pp_schedule: str) -> int:
    """Choose one rank for external metric writers.

    Trainer metrics are synchronized across PP stages; selecting the last
    stage for ordinary schedules preserves the established writer location.
    V schedules select rank 0, which owns their last virtual stage.
    """
    if not parallel_dims.pp_enabled:
        return 0
    if is_v_schedule(pp_schedule):
        return 0
    # First rank of the last pipeline stage. Ranks are laid out
    # [dp_replicate, dp_shard, cp, tp] within a stage, so this is the first
    # rank of the final stage block.
    pp_size = parallel_dims.pp
    return (parallel_dims.world_size // pp_size) * (pp_size - 1)


@dataclass(kw_only=True, slots=True)
class DerivedMetrics:
    """The numbers a log call computes from its window, before naming."""

    tps: float
    tflops: float
    mfu: float | None
    time_end_to_end: float
    time_data_loading: float
    time_data_loading_pct: float


class MetricsProcessor:
    """Turns per-step counters into throughput, MFU, and memory, and reports them.

    Args:
        config: which sinks to build and how often to log.
        parallel_dims: the resolved parallelism, for the throughput divisor and
            the metrics rank. ``None`` in the single-process case.
        dump_folder: base directory for the TensorBoard event files.
        pp_schedule: the pipeline schedule name, which decides the metrics rank.
        num_flops_per_token: model FLOPs per token, used for tflops and MFU. The
            caller sets this once the model exists. ``0`` -- the value
            :func:`~llmtuner.models.hf.flops.flops_per_token` returns for a
            config whose geometry it cannot read -- suppresses MFU rather than
            reporting ``0.00%``; ``tflops`` is then ``0.0`` of its own accord.
        config_dict: the full job config, handed to wandb. Only wandb reads it.
        tag: optional override for ``config.tag``, applied to recorded keys.
            ``None`` uses the config value; the console line is not tagged.
    """

    def __init__(
        self,
        config: MetricsConfig,
        *,
        parallel_dims: ParallelTopology | None,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        num_flops_per_token: int = 0,
        num_parameters: int | None = None,
        num_trainable_parameters: int | None = None,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        if (
            isinstance(num_flops_per_token, bool)
            or not isinstance(num_flops_per_token, int)
            or num_flops_per_token < 0
        ):
            raise ValueError(
                "num_flops_per_token must be a non-negative integer, "
                f"got {num_flops_per_token!r}"
            )
        self.num_parameters = num_parameters
        self.num_trainable_parameters = num_trainable_parameters
        self.config = config
        self.parallel_dims = parallel_dims
        self.logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            config_dict=config_dict,
            tag=config.tag if tag is None else tag,
        )
        try:
            self.device_memory_monitor = build_device_memory_monitor()
            self.color = (
                Color()
                if colors_enabled(disable_color_printing=config.disable_color_printing)
                else NoColor()
            )
            self._pp_schedule = pp_schedule
            self.gpu_peak_flops = get_peak_flops(self.device_memory_monitor.device_name)
            self.num_flops_per_token = num_flops_per_token

            self.ntokens_since_last_log = 0
            self.valid_tokens_since_last_log = 0
            self._summary_windows = 0
            self._summary_loss = 0.0
            self._summary_steps = 0
            self._summary_seconds = 0.0
            self._summary_reported = False
            self.latest_training_observations: dict[str, float] = {}
            self._steady_summary_reported = False
            self.data_loading_times: list[float] = []
            self.time_last_log = time.perf_counter()
            self.step_last_log: int | None = None
            self.device_memory_monitor.reset_peak_stats()
        except BaseException:
            try:
                self.logger.close()
            except BaseException:
                logger.exception("Metrics logger cleanup failed during initialization")
            raise

    @property
    def _non_data_parallel_size(self) -> int:
        """Number of CP/TP/PP ranks sharing one data-parallel token stream.

        Fetch accounting records the full local batch before CP/TP partitioning.
        Divide by this degree to report per-device throughput. EP overlays the
        dense topology and must not introduce another divisor.
        """
        if self.parallel_dims is None:
            return 1
        return self.parallel_dims.non_data_parallel_size

    def should_log(self, step: int) -> bool:
        """Whether ``step`` is a logging step. The first step always is.

        Anchors ``step_last_log`` on the FIRST call after a fresh start or a
        resume -- even when this step does not log: ``time_end_to_end`` divides
        the measured wall time by ``step - step_last_log``, and anchoring late
        (at the first logging step) would divide a multi-step span by 1.
        """
        if self.step_last_log is None:
            self.step_last_log = step - 1
        return step == 1 or step % self.config.log_freq == 0

    def add_data_loading_time(self, seconds: float) -> None:
        """Record how long this step's data fetch blocked."""
        self.data_loading_times.append(seconds)

    def add_tokens(self, num_tokens: int) -> None:
        """Record tokens processed since the last log."""
        self.ntokens_since_last_log += num_tokens

    def add_valid_tokens(self, num_tokens: int) -> None:
        """Record supervised target tokens separately from padded model inputs."""
        self.valid_tokens_since_last_log += num_tokens

    def _build_metric_logger(
        self,
        *,
        config: MetricsConfig,
        parallel_dims: ParallelTopology | None,
        dump_folder: str,
        pp_schedule: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """Build the sink this rank should write to.

        Returns a no-op logger -- rather than ``None`` -- when nothing is
        enabled or this is not the metrics rank, so the caller has one code path.
        """
        if not (config.enable_tensorboard or config.enable_wandb):
            return BaseLogger()

        if not config.save_for_all_ranks:
            if parallel_dims is None:
                metrics_rank = 0
            else:
                metrics_rank = get_metrics_rank(
                    parallel_dims=parallel_dims, pp_schedule=pp_schedule
                )
            if get_distributed_rank() != metrics_rank:
                return BaseLogger()

        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )
        if config.save_for_all_ranks:
            base_log_dir = os.path.join(base_log_dir, f"rank_{get_distributed_rank()}")

        logger_container = LoggerContainer()

        try:
            if config.enable_wandb:
                try:
                    logger_container.add_logger(
                        WandBLogger(base_log_dir, config_dict=config_dict, tag=tag)
                    )
                except ImportError as error:
                    # Only the missing-package case: that is a dependency the user
                    # can install. A credential or project error is a
                    # misconfiguration they should see, not something to swallow.
                    raise ImportError(
                        "metrics.enable_wandb is set but wandb is not installed. "
                        "Install it with 'pip install wandb', or unset it."
                    ) from error

            if config.enable_tensorboard:
                logger_container.add_logger(TensorBoardLogger(base_log_dir, tag))

        except BaseException:
            try:
                logger_container.close()
            except BaseException:
                logger.exception("Metrics logger cleanup failed during initialization")
            raise

        return logger_container

    def _derive(self, step: int) -> DerivedMetrics:
        """Compute the windowed numbers that both log paths share.

        The keys these become carry no tag: a sensor is what the tag belongs to,
        and it applies it once on the way out. Applying it here as well would
        produce ``tag/tag/throughput(tps)`` in the recorded output.
        """
        time_delta = time.perf_counter() - self.time_last_log

        tps = token_throughput(
            self.ntokens_since_last_log,
            time_delta,
            replication_degree=self._non_data_parallel_size,
        )
        compute = compute_utilization(
            tokens_per_second=tps,
            flops_per_token=self.num_flops_per_token,
            peak_flops_per_device=self.gpu_peak_flops,
            world_size=1,
        )
        # Logs retain model TFLOPS even when the hardware peak is unknown.
        tflops = compute["tflops"]
        if tflops is None:
            tflops = model_tflops(self.num_flops_per_token, tps)
        mfu = compute["utilization_percent"]

        assert self.step_last_log is not None, "should_log must run before log"
        time_end_to_end = time_delta / (step - self.step_last_log)
        # Zero rather than a division: a window with no recorded fetch spent no
        # time fetching, which is the number. torchtitan reports the mean only,
        # so a caller that does not time its loader -- or a log before the first
        # training step -- divides by zero there.
        time_data_loading = (
            sum(self.data_loading_times) / len(self.data_loading_times)
            if self.data_loading_times
            else 0.0
        )
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        return DerivedMetrics(
            tps=tps,
            tflops=tflops,
            mfu=mfu,
            time_end_to_end=time_end_to_end,
            time_data_loading=time_data_loading,
            time_data_loading_pct=time_data_loading_pct,
        )

    def _end_window(self, step: int) -> None:
        """Reset the counters that the next window measures over."""
        self.ntokens_since_last_log = 0
        self.valid_tokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.step_last_log = step
        self.device_memory_monitor.reset_peak_stats()

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Report one training step.

        Synchronized training observations supply global token counts and
        slowest-rank timing. Allocator fields remain local to this logging rank;
        they are not world maxima and cannot establish all-rank memory bounds.

        Args:
            step: the current training step.
            global_avg_loss: total loss over total valid tokens, so it is
                independent of how the batch was split across ranks.
            global_max_loss: the worst rank's per-token loss. Equal to
                ``global_avg_loss`` when averaging over rank means rather than
                tokens -- see the note in the trainer.
            grad_norm: total gradient norm before clipping, returned by clipping.
            extra_metrics: additional numbers the caller wants recorded.
        """
        if self.step_last_log is None:
            # Defensive: should_log anchors on its first call; a caller that
            # went straight to log() gets the same anchor here.
            self.step_last_log = step - 1

        derived = self._derive(step)
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        valid_fraction = (
            self.valid_tokens_since_last_log / self.ntokens_since_last_log
            if self.ntokens_since_last_log
            else 0.0
        )
        valid_tps = derived.tps * valid_fraction
        allocated_gib = self.device_memory_monitor.get_current_allocated_gib()
        metrics: dict[str, Any] = {
            "throughput/valid_target_tps": valid_tps,
            "tokens/valid_target_fraction": valid_fraction,
            "memory/current_allocated(GiB)": allocated_gib,
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": derived.tps,
            "tflops": derived.tflops,
            "time_metrics/end_to_end(s)": derived.time_end_to_end,
            "time_metrics/data_loading(s)": derived.time_data_loading,
            "time_metrics/data_loading(%)": derived.time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        if self.num_parameters is not None:
            metrics["model/parameters"] = self.num_parameters
        if self.num_trainable_parameters is not None:
            metrics["model/trainable_parameters"] = self.num_trainable_parameters
        if derived.mfu is not None:
            metrics["mfu(%)"] = derived.mfu
        if extra_metrics:
            metrics.update(extra_metrics)

        # Synchronized per-step observations have a training-only scope. Keep
        # legacy window fields in the sink for compatibility, but prefer these
        # explicitly scoped values on the console.
        observed = extra_metrics or {}
        console_tps = observed.get("train/tps_per_device", derived.tps)
        console_time = observed.get(
            "train/slowest_rank_step_seconds", derived.time_end_to_end
        )
        valid_tps = observed.get("train/valid_target_tps_per_device", valid_tps)
        valid_fraction = observed.get("train/valid_target_fraction", valid_fraction)

        self.logger.log(metrics, step)

        self._summary_windows += 1
        self._summary_loss += global_avg_loss
        self._summary_steps += step - self.step_last_log
        self._summary_seconds += derived.time_end_to_end * (step - self.step_last_log)
        lr_str = (
            ", ".join(
                f"{key}={value:.3e}"
                for key, value in (extra_metrics or {}).items()
                if key.startswith("lr/")
            )
            or "lr=N/A"
        )
        color = self.color
        mfu = observed.get("train/mfu_percent", metrics.get("mfu(%)"))
        mfu_str = "N/A" if mfu is None else f"{mfu:.2f}%"
        parameter_str = (
            "N/A"
            if self.num_parameters is None
            else f"{self.num_parameters / 1e9:.6f}B"
        )
        trainable_str = (
            "N/A"
            if self.num_trainable_parameters is None
            else f"{self.num_trainable_parameters / 1e9:.6f}B"
        )
        details = [f"max_loss: {global_max_loss:.5f}"]
        loading_seconds = observed.get(
            "train/data_loading_rank_max_seconds", derived.time_data_loading
        )
        loading_percent = observed.get(
            "train/data_loading_rank_max_percent", derived.time_data_loading_pct
        )
        details.append(f"data: {loading_seconds:.3f}s({loading_percent:.1f}%)")
        if "train/global_valid_targets" in observed:
            details.append(
                f"global_tokens: {int(observed['train/global_tokens']):,} "
                f"valid_targets: {int(observed['train/global_valid_targets']):,}"
            )
        details.append(
            f"alloc_retries: {device_mem_stats.num_alloc_retries} "
            f"ooms: {device_mem_stats.num_ooms}"
        )
        if "steady/samples" in observed:
            details.append(
                f"steady[n={int(observed['steady/samples'])}]: "
                f"p50={observed['steady/step_p50_seconds']:.3f}s "
                f"p90={observed['steady/step_p90_seconds']:.3f}s "
                f"tps={observed['steady/tps_per_device']:.1f}/device"
            )
            if "steady/mfu_percent" in observed:
                details.append(f"steady_mfu: {observed['steady/mfu_percent']:.2f}%")
        details.extend(
            f"{key}={value:.4g}"
            for key, value in observed.items()
            if key.startswith(("aux_loss/", "moe/", "swap/", "checkpoint/"))
            or ("loss" in key.lower() and key.endswith("/mean"))
        )
        # A single pre-built message rather than a format string plus arguments:
        # the colour codes are already interpolated, so passing them as logging
        # arguments would leave a line no log handler could re-format.
        logger.info(
            f"{color.red}step: {step:2d}  "
            f"{color.green}loss: {global_avg_loss:8.5f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"step_time: {console_time:.3f}s  {lr_str}  "
            f"allocated: {allocated_gib:.2f}GiB  "
            f"peak_active: {device_mem_stats.max_active_gib:.2f}GiB  "
            f"valid_tps: {valid_tps:,.1f}/device (targets: {valid_fraction:.1%})  "
            f"params: {parameter_str} (trainable: {trainable_str})  "
            f"{color.blue}tps: {round(console_tps):,}/device  "
            f"{color.cyan}tflops: "
            f"{observed.get('train/tflops_per_device', derived.tflops):,.2f}/device  "
            f"{color.magenta}mfu: {mfu_str}{color.reset}  " + "  ".join(details)
        )

        self._end_window(step)

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ) -> None:
        """Report a validation pass.

        Uses the same window as ``log``: a validation pass that ran since the
        last log is inside this window's token count and elapsed time, so the
        throughput here is the one including validation.
        """
        if self.step_last_log is None:
            self.step_last_log = step - 1

        time_delta = time.perf_counter() - self.time_last_log
        device_mem_stats = self.device_memory_monitor.get_peak_stats()
        tps = token_throughput(
            self.ntokens_since_last_log,
            time_delta,
            replication_degree=self._non_data_parallel_size,
        )

        metrics: dict[str, Any] = {
            "validation_metrics/loss": loss,
            "validation_metrics/throughput(tps)": tps,
            "validation_metrics/memory/max_active(GiB)": (
                device_mem_stats.max_active_gib
            ),
            "validation_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
            "validation_metrics/memory/max_reserved(GiB)": (
                device_mem_stats.max_reserved_gib
            ),
            "validation_metrics/memory/max_reserved(%)": (
                device_mem_stats.max_reserved_pct
            ),
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.yellow}validate step: {step:2d}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        self._end_window(step)

    def log_resources(self, step: int, metrics: dict[str, float]) -> None:
        """Publish local counters after checkpoint/save boundaries as well."""
        if metrics:
            self.logger.log(metrics, step)
            logger.info(
                "Resource observations (rank-local): %s",
                "  ".join(f"{key}={value:.4g}" for key, value in metrics.items()),
            )

    def close(self) -> None:
        observed = self.latest_training_observations
        if not self._steady_summary_reported and "steady/samples" in observed:
            mfu = observed.get("steady/mfu_percent")
            logger.info(
                "Training steady summary (rolling, warmup excluded): "
                "%d samples, p50=%.4fs, p90=%.4fs, tps=%.1f/device, "
                "tflops=%.2f/device, mfu=%s",
                int(observed["steady/samples"]),
                observed["steady/step_p50_seconds"],
                observed["steady/step_p90_seconds"],
                observed["steady/tps_per_device"],
                observed["steady/tflops_per_device"],
                "N/A" if mfu is None else f"{mfu:.2f}%",
            )
            self._steady_summary_reported = True
        if not self._summary_reported and self._summary_windows:
            logger.info(
                "Training logged-window summary: %d steps, %d loss samples, "
                "sample_mean_loss=%.6f, mean_step_time=%.4fs "
                "(logged windows only; may include validation/startup)",
                self._summary_steps,
                self._summary_windows,
                self._summary_loss / self._summary_windows,
                self._summary_seconds / self._summary_steps,
            )
            self._summary_reported = True
        self.logger.close()
