"""LLMTunerConfig: the aggregate config root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig, ProfilerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from llmtuner.config.training_config import TrainingConfig
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.errors import ConfigError
from llmtuner.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMTunerConfig:
    """Single entry point: composes the argument groups (no multiple inheritance)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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
        """Compose the nine parsed argument groups into the aggregate config."""
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
        parallel_fields = {
            item.name: getattr(parallel, item.name)
            for item in fields(ParallelConfig)
            if item.init
        }
        parallel = ParallelConfig(**(parallel_fields | plans))
        optimizer_fields = {
            item.name: getattr(optimizer, item.name)
            for item in fields(OptimizerConfig)
            if item.init
        }
        optimizer_fields["lr_scheduler_config"] = lr_scheduler
        optimizer = OptimizerConfig(**optimizer_fields)
        training_fields = {
            item.name: getattr(training, item.name)
            for item in fields(TrainingConfig)
            if item.init
        }
        training_fields.update(
            checkpoint_config=checkpoint,
            dataloader_config=dataloader,
            metrics_config=metrics,
            profiler_config=profiler,
        )
        training = TrainingConfig(**training_fields)
        return cls(
            model=model, parallel=parallel, optimizer=optimizer, training=training
        )

    def __post_init__(self) -> None:
        # Cross-group check: the sequence is sharded CP first, TP second, so
        # the real invariant is (T/cp) % tp == 0 -- which, for the contiguous
        # splits both sharders perform, is exactly max_seq_len % (cp * tp) == 0.
        # Two separate modulo checks would miss ragged cases (e.g. T=6, cp=2,
        # tp=2 passes both but leaves a length-3 shard for tp=2).
        joint = self.parallel.cp_size * self.parallel.tp_size
        if self.training.max_seq_len % joint != 0:
            raise ConfigError(
                f"max_seq_len ({self.training.max_seq_len}) must be divisible by "
                f"cp * tp ({self.parallel.cp_size} * {self.parallel.tp_size})"
            )
        # Async TP is a compiled-TP optimization: without compile there is no
        # inductor pass to pipeline the collectives, and without TP there are
        # no collectives. Reject both halves here so the error names the flag
        # rather than surfacing deep inside the compile step.
        if (
            self.training.compile_config.enable_async_tensor_parallel
            and not self.training.compile
        ):
            raise ConfigError(
                "training.compile_config.enable_async_tensor_parallel "
                "requires training.compile=True: async TP is an inductor "
                "pass over compiled regions, so without compile it would "
                "silently do nothing."
            )
        if (
            self.training.compile_config.enable_async_tensor_parallel
            and self.training.compile
            and self.parallel.tp_size < 2
        ):
            raise ConfigError(
                "training.compile_config.enable_async_tensor_parallel "
                "requires tensor_parallel_size > 1 (got "
                f"{self.parallel.tp_size}): it pipelines the TP collectives, "
                "and there are none at tp=1."
            )

    # -- Flat view: lets the trainer read cfg.lr / cfg.steps / ... uniformly. --
    # The parallel degrees (dp/tp/pp/cp/ep) are deliberately NOT here: the
    # parallel layer takes ``cfg.parallel`` (a ParallelConfig) directly, so a
    # second flat spelling of the same numbers could only drift apart.
    @property
    def hf_model(self) -> str:
        return self.model.model_name_or_path

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.model.intermediate_size

    @property
    def num_hidden_layers(self) -> int:
        return self.model.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.model.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.model.num_key_value_heads

    @property
    def arch_overrides(self) -> dict[str, Any]:
        return self.model.arch_overrides

    @property
    def experts_implementation(self) -> str:
        return self.model.experts_implementation

    @property
    def norm_backend(self) -> str:
        return self.model.norm_backend

    @property
    def compute_dtype(self) -> str | None:
        return self.model.compute_dtype

    @property
    def lr(self) -> float:
        return self.optimizer.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def betas(self) -> tuple[float, float]:
        return self.optimizer.betas

    @property
    def eps(self) -> float:
        return self.optimizer.eps

    @property
    def lr_scheduler_config(self) -> LRSchedulerConfig:
        return self.optimizer.lr_scheduler_config

    @property
    def global_batch_size(self) -> int:
        return self.training.global_batch_size

    @property
    def max_seq_len(self) -> int:
        return self.training.max_seq_len

    @property
    def steps(self) -> int:
        return self.training.steps

    @property
    def seed(self) -> int:
        return self.training.seed

    @property
    def deterministic(self) -> bool:
        return self.training.deterministic

    @property
    def detect_anomaly(self) -> bool:
        return self.training.detect_anomaly

    @property
    def pipeline_parallel_schedule(self) -> str:
        return self.parallel.pp_config.pipeline_parallel_schedule

    @property
    def max_norm(self) -> float:
        return self.training.max_norm

    @property
    def dump_folder(self) -> str:
        return self.training.dump_folder

    @property
    def gc_freq(self) -> int:
        return self.training.gc_freq

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.training.gradient_accumulation_steps

    @property
    def checkpoint(self) -> CheckpointConfig:
        return self.training.checkpoint

    @property
    def metrics(self) -> MetricsConfig:
        return self.training.metrics

    @property
    def dataloader(self) -> DataloaderConfig:
        return self.training.dataloader

    @property
    def profiler(self) -> ProfilerConfig:
        return self.training.profiler

    @property
    def validation(self) -> ValidationConfig | None:
        return self.training.validation

    def auto_fill_model(self) -> None:
        """Fill architecture fields from a local HF directory or Hub config."""
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        mp = self.model.model_name_or_path
        if not os.path.exists(mp) and mp in CONFIG_MAPPING:
            return  # offline architecture name; explicit sizes are authoritative
        try:
            hf_config = AutoConfig.from_pretrained(mp)
        except Exception as e:  # noqa: BLE001 - warn, keep explicit values
            logger.warning("Could not load AutoConfig for '%s': %s", mp, e)
            return
        # The model factory trains the decoder described by text_config for
        # composite HF configurations; the aggregate run sizes must match it.
        hf_config = getattr(hf_config, "text_config", hf_config)
        self.model.update_architecture(hf_config)
