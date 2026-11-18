"""Training metric processing, device-memory sampling, and console reporting.

Reuses production metric arithmetic, derived results and device-memory monitors.
Synchronized per-step and steady-window observations live in
``trainer.runtime.training_observer``. This module retains legacy log-window metrics for
existing consumers and publishes the synchronized observations when available.

Logging performs no collectives. The trainer synchronizes pipeline loss metrics
before reporting; writer ownership and console rank filtering are independent.
Unknown device peaks suppress MFU, and empty loading windows report zero.
External sinks live in ``utils.metric_loggers``. Each owns its resources and
propagates configuration/runtime failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from llmtuner.components.metrics import (
    DerivedMetrics as DerivedMetrics,
)
from llmtuner.components.metrics import (
    DeviceMemoryMonitor,
    DeviceMemStats,
    build_device_memory_monitor,
    compute_utilization,
    configured_tokens_per_step,
    model_tflops,
    token_throughput,
)
from llmtuner.components.metrics import MetricsProcessor as BaseMetricsProcessor
from llmtuner.utils import metric_loggers

if TYPE_CHECKING:
    # Annotation-only. ``parallel_dims`` imports the training config, which
    # imports the checkpointer, so importing it for real here would close a
    # cycle back into the components package that only exists to name a type.
    from experiments.pipeline_parallel.topology import ParallelTopology
    from llmtuner.config.observability_config import MetricsConfig

__all__ = [
    "DeviceMemStats",
    "DeviceMemoryMonitor",
    "MetricsProcessor",
    "build_device_memory_monitor",
    "get_metrics_rank",
    "compute_utilization",
    "configured_tokens_per_step",
    "model_tflops",
    "token_throughput",
]


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


class MetricsProcessor(BaseMetricsProcessor):
    """Use shared reporting and cleanup with the experimental PP writer rank."""

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
        self._pp_schedule = pp_schedule
        super().__init__(
            config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            num_flops_per_token=num_flops_per_token,
            num_parameters=num_parameters,
            num_trainable_parameters=num_trainable_parameters,
            config_dict=config_dict,
            tag=tag,
        )

    def _build_metric_logger(
        self,
        *,
        config: MetricsConfig,
        dump_folder: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> metric_loggers.BaseLogger:
        parallel_dims = self.parallel_dims
        metrics_rank = 0
        if (
            (config.enable_tensorboard or config.enable_wandb)
            and not config.save_for_all_ranks
            and parallel_dims is not None
        ):
            metrics_rank = get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=self._pp_schedule
            )
        return super()._build_metric_logger(
            config=config,
            dump_folder=dump_folder,
            config_dict=config_dict,
            tag=tag,
            metrics_rank=metrics_rank,
        )
