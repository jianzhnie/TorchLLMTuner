"""Inject service factories without exposing Trainer to components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol

from llmtuner.trainer.runtime.builders import EMAFactory, SchedulerFactory

if TYPE_CHECKING:
    from torch import nn
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import _PipelineSchedule

    from experiments.pipeline_parallel.config import (
        LLMTunerConfig,
    )
    from experiments.pipeline_parallel.metrics import (
        MetricsProcessor,
    )
    from experiments.pipeline_parallel.optimizer import OptimizersContainer
    from experiments.pipeline_parallel.parallel_config import (
        ParallelConfig,
    )
    from experiments.pipeline_parallel.topology import ParallelTopology
    from llmtuner.components.checkpointer.core.base import BaseCheckpointManager
    from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
    from llmtuner.components.loss_contract import LossFunctions, LossSum
    from llmtuner.components.optimizer.ema import EMA
    from llmtuner.components.optimizer.lr_scheduler import LRSchedulersContainer
    from llmtuner.config.checkpoint_config import CheckpointConfig
    from llmtuner.config.observability_config import MetricsConfig
    from llmtuner.config.optimizer_config import OptimizerConfig
    from llmtuner.parallel.assembly import ModelPartLayout
    from llmtuner.trainer.runtime.builders import ProfilerFactory


class PipelineScheduleFactory(Protocol):
    """Build a training schedule retaining the supplied stages in order.

    Pass PipelineLoss (or an equivalent callable with mutable scalar tensor
    global_valid_tokens) as the schedule's loss_fn. The driver scopes this
    denominator to each invocation. The returned schedule must support the
    installed Torch microbatch API and have backward enabled. logits_width is
    the declared replicated vocabulary width and must be checked before loss;
    None leaves vocabulary-sharded geometry to the sharded loss contract.
    """

    def __call__(
        self,
        stages: list[PipelineStage],
        /,
        *,
        cfg: ParallelConfig,
        loss_sum: LossSum | None,
        tp_group: ProcessGroup | None = None,
        global_vocab_size: int | None = None,
        logits_width: int | None = None,
    ) -> _PipelineSchedule: ...


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


@dataclass(frozen=True, slots=True)
class TrainingBuilders:
    """Optional factories using the default component's constructor contract.

    None resolves the existing factory at assembly time, so registration and
    scoped overrides keep working. Scheduler receives the actual optimizers;
    EMA is constructed only when enabled. Returned optimizer containers must
    support the existing state, iteration, hook and update interfaces. Metrics
    factories must return a processor supporting reporting and close().

    Services with background resources must expose close(). A failed scheduler
    close retains its optimizer until a later close attempt succeeds. Factories
    own cleanup of resources created before their result is returned.
    Optimizer services may expose observation_metrics as a snapshot of named
    rank-local scalar counters; training reporting consumes it without inspecting
    optimizer implementations.

    loss receives the run configuration and returns LossFunctions. A pipeline
    schedule factory accepts stages, parallel cfg, loss_sum and vocabulary
    reduction keywords, using build_pipeline_schedule's calling convention.

    profiler accepts profiler config, global_step and base_folder, and returns
    a context manager with close() whose entered value supports step(). checkpoint
    accepts the selected manager's constructor arguments and returns a manager with
    save/load/wait/close semantics. Failed close calls retain the manager and
    its dependencies for retry; successful close calls must be idempotent.
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
        for factory in fields(self):
            value = getattr(self, factory.name)
            if value is not None and not callable(value):
                raise TypeError(f"Training builder {factory.name!r} must be callable")
