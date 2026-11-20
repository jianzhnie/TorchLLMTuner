"""LLMTunerConfig: the aggregate config root."""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.pipeline_parallel.parallel_config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig as BaseLLMTunerConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig, ProfilerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.training_config import (
    TrainingConfig,
)
from llmtuner.config.utils import rebuild_config_group


@dataclass
class LLMTunerConfig(BaseLLMTunerConfig):
    """Use shared run configuration with the experimental parallel group."""

    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_groups(
        cls,
        *,
        model: ModelConfig,
        parallel: ParallelConfig,
        optimizer: OptimizerConfig,
        lr_scheduler: LRSchedulerConfig,
        training: TrainingConfig,
        checkpoint: CheckpointConfig,
        dataloader: DataloaderConfig,
        metrics: MetricsConfig,
        profiler: ProfilerConfig,
        fsdp_config: FSDPConfig | None = None,
        tp_config: TensorParallelConfig | None = None,
        pp_config: PipelineParallelConfig | None = None,
        cp_config: ContextParallelConfig | None = None,
        ep_config: ExpertParallelConfig | None = None,
    ) -> LLMTunerConfig:
        """Compose parsed argument groups into the aggregate run configuration."""
        # CLI views hide nested fields with init=False. Rebuild the base
        # configs to graft parsed groups without modifying caller instances.
        plans = {
            name: plan
            for name, plan in {
                "fsdp_config": fsdp_config,
                "tp_config": tp_config,
                "pp_config": pp_config,
                "cp_config": cp_config,
                "ep_config": ep_config,
            }.items()
            if plan is not None
        }
        parallel = rebuild_config_group(ParallelConfig, parallel, **plans)
        optimizer = rebuild_config_group(
            OptimizerConfig, optimizer, lr_scheduler_config=lr_scheduler
        )
        training = rebuild_config_group(
            TrainingConfig,
            training,
            checkpoint_config=checkpoint,
            dataloader_config=dataloader,
            metrics_config=metrics,
            profiler_config=profiler,
        )
        return cls(
            model=model, parallel=parallel, optimizer=optimizer, training=training
        )

    # -- Flat view: lets the trainer read cfg.lr / cfg.steps / ... uniformly. --
    # The parallel degrees (dp/tp/pp/cp/ep) are deliberately NOT here: the
    # parallel layer takes ``cfg.parallel`` (a ParallelConfig) directly, so a
    # second flat spelling of the same numbers could only drift apart.

    @property
    def pipeline_parallel_schedule(self) -> str:
        return self.parallel.pp_config.pipeline_parallel_schedule
