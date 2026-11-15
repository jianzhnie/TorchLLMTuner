"""Inject service factories without exposing Trainer to components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...components.loss_contract import LossFunctions
    from ...components.metrics import MetricsProcessor
    from ...components.optimizer.ema import EMA
    from ...components.optimizer.optimizer import OptimizersContainer
    from ...config.llmtuner_config import LLMTunerConfig


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
    pipeline_schedule: Callable[..., Any] | None = None
    optimizer: Callable[..., OptimizersContainer] | None = None
    scheduler: Callable[..., Any] | None = None
    ema: Callable[..., EMA] | None = None
    metrics: Callable[..., MetricsProcessor] | None = None
    profiler: Callable[..., Any] | None = None
    checkpoint: Callable[..., Any] | None = None

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
