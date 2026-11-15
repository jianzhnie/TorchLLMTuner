"""Inject service factories without exposing Trainer to components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from torch import nn
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import _PipelineSchedule
    from torch.optim import Optimizer

    from ...components.checkpointer.core.base import BaseCheckpointManager
    from ...components.checkpointer.model_state import ModelStateDictAdapter
    from ...components.loss_contract import LossFunctions, LossSum
    from ...components.metrics import MetricsProcessor
    from ...components.optimizer.ema import EMA
    from ...components.optimizer.lr_scheduler import LRSchedulersContainer
    from ...components.optimizer.optimizer import OptimizersContainer
    from ...config.checkpoint_config import CheckpointConfig
    from ...config.llmtuner_config import LLMTunerConfig
    from ...config.lr_scheduler_config import LRSchedulerConfig
    from ...config.observability_config import MetricsConfig, ProfilerConfig
    from ...config.optimizer_config import OptimizerConfig
    from ...config.parallel_config import ParallelConfig
    from ...parallel.assembly import ModelPartLayout
    from ...parallel.topology import ParallelTopology


class PipelineScheduleFactory(Protocol):
    """Build a torch pipeline schedule with the assembled vocabulary loss."""

    def __call__(
        self,
        stages: list[PipelineStage],
        /,
        *,
        cfg: ParallelConfig,
        loss_sum: LossSum | None,
        tp_group: ProcessGroup | None = None,
        global_vocab_size: int | None = None,
    ) -> _PipelineSchedule: ...


class EMAFactory(Protocol):
    """Build model-state averaging over final local parameter ownership."""

    def __call__(
        self,
        *,
        model_parts: list[nn.Module],
        decay: float | None,
        half_life_fraction: float,
        start_step: int,
        step_bias: int,
        update_every_n_steps: int,
        buffer_patterns: Sequence[str],
        sharded_expert_parallel: bool,
        sharded_tensor_parallel: bool,
        runtime_layouts: tuple[ModelPartLayout, ...],
    ) -> EMA: ...


class MetricsFactory(Protocol):
    """Build reporting with explicit model counts and reduction topology."""

    def __call__(
        self,
        config: MetricsConfig,
        /,
        *,
        parallel_dims: ParallelTopology | None,
        dump_folder: str,
        num_parameters: int,
        num_trainable_parameters: int,
        pp_schedule: str,
        num_flops_per_token: int,
    ) -> MetricsProcessor: ...


class OptimizerFactory(Protocol):
    """Construct against final parameter ownership and borrowed process groups."""

    def __call__(
        self,
        config: OptimizerConfig,
        /,
        *,
        model_parts: list[nn.Module],
        pipeline_mesh: DeviceMesh | None,
        initialization_group: ProcessGroup | None,
        sharded_tensor_parallel: bool,
        sharded_expert_parallel: bool,
        runtime_layouts: tuple[ModelPartLayout, ...],
    ) -> OptimizersContainer: ...


class StepProfiler(Protocol):
    """The entered observation service required by the training loop."""

    def step(self) -> None: ...


class SchedulerFactory(Protocol):
    """Build schedules over the actual inner optimizers, without Trainer."""

    def __call__(
        self,
        config: LRSchedulerConfig,
        /,
        *,
        optimizers: list[Optimizer],
        training_steps: int,
    ) -> LRSchedulersContainer: ...


class CheckpointFactory(Protocol):
    """Construct persistence with explicit borrowed state and owned resources.

    Construction may create process groups and must obey the selected backend's
    collective contract. The factory cleans partial objects until it returns;
    Trainer owns closing the returned manager.
    """

    def __call__(
        self,
        config: CheckpointConfig,
        /,
        *,
        model_parts: list[nn.Module],
        runtime_layouts: tuple[ModelPartLayout, ...],
        optimizer: OptimizersContainer,
        lr_scheduler: LRSchedulersContainer,
        ema: EMA | None,
        states: dict[str, Any],
        folder: str,
        sd_adapter: ModelStateDictAdapter,
    ) -> BaseCheckpointManager: ...


class ProfilerFactory(Protocol):
    """Return a context owner; optional close() enables shutdown retries.

    The entered value may differ from the context owner and only needs step().
    Factories own partial construction and context owners clean failed entry.
    """

    def __call__(
        self,
        config: ProfilerConfig,
        /,
        *,
        global_step: int,
        base_folder: str,
    ) -> AbstractContextManager[StepProfiler]: ...


@dataclass(frozen=True, slots=True)
class TrainingBuilders:
    """Optional factories using the default component's constructor contract.

    None resolves the existing factory at assembly time, so registration and
    scoped overrides keep working. Scheduler receives the actual optimizers;
    EMA is constructed only when enabled. Returned optimizer containers must
    support the existing state, iteration, hook and update interfaces. Metrics
    factories must return a processor supporting reporting and close().

    loss receives the run configuration and returns LossFunctions. A pipeline
    schedule factory accepts stages, parallel cfg, loss_sum and vocabulary
    reduction keywords, using build_pipeline_schedule's calling convention.

    profiler accepts profiler config, global_step and base_folder, and returns
    a context manager whose entered value supports step(). checkpoint accepts
    the selected manager's constructor arguments and returns a manager with
    save/load/wait/close and requires_close_retry lifecycle semantics.
    """

    loss: Callable[[LLMTunerConfig], LossFunctions] | None = None
    pipeline_schedule: PipelineScheduleFactory | None = None
    optimizer: OptimizerFactory | None = None
    scheduler: SchedulerFactory | None = None
    ema: EMAFactory | None = None
    metrics: MetricsFactory | None = None
    profiler: ProfilerFactory | None = None
    checkpoint: CheckpointFactory | None = None

    def __post_init__(self) -> None:
        for name in (
            "loss",
            "pipeline_schedule",
            "optimizer",
            "scheduler",
            "ema",
            "metrics",
            "profiler",
            "checkpoint",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"Training builder {name!r} must be callable")
