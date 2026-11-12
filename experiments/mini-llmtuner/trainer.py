"""Framework-independent lifecycle trainer for the Qwen3-MoE prototype."""

import hashlib
import json
import random

import torch
import torch.distributed as dist

from .checkpoint import CheckpointManager
from .components import (
    AdamWFactory,
    CrossEntropyLoss,
    NoOpProfiler,
    NoOpScheduler,
    TokenMetrics,
)
from .config import ExperimentConfig, load_model_config
from .data import SyntheticData
from .engine import TrainingEngine
from .llama import LlamaModel
from .model import Qwen3Model, Qwen3MoeModel


class BaseTrainer:
    """Lifecycle only: execution and data are injected, never constructed here."""

    def __init__(
        self, engine, data, checkpoint, *, accumulation=1, metrics=None, profiler=None
    ):
        if accumulation < 1:
            raise ValueError("accumulation must be positive")
        self.closed = False
        self.engine, self.data, self.checkpoint = engine, data, checkpoint
        self.accumulation = accumulation
        self.metrics = metrics or TokenMetrics()
        self.profiler = profiler or NoOpProfiler()
        self.start_step = checkpoint.step

    def train(self, steps):
        if self.closed:
            raise RuntimeError("Trainer is closed")
        if steps < self.checkpoint.step:
            raise ValueError("Target steps precede the restored step")
        history = []
        self.engine.model.train()
        for step in range(self.checkpoint.step, steps):
            batches = [self.data.next_batch() for _ in range(self.accumulation)]
            result = self.engine.train_step(batches)
            self.checkpoint.step = step + 1
            labels = torch.cat([batch["labels"].flatten() for batch in batches])
            record = self.metrics.record(
                step, torch.tensor(result.loss_sum / result.targets), labels
            )
            record.update(tokens=result.targets, grad_norm=result.grad_norm)
            history.append(record)
            self.profiler.step()
        return history

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            close = getattr(self.profiler, "close", None)
            if close is not None:
                close()
        finally:
            close = getattr(self.data, "close", None)
            if close is not None:
                close()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Trainer is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class ExperimentTrainer(BaseTrainer):
    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        *,
        model=None,
        optimizer_factory=None,
        scheduler_factory=None,
        loss=None,
        data=None,
        metrics=None,
        profiler=None,
        checkpoint_adapter=None,
        runtime_signature: str = "default-v1",
    ) -> None:
        self.config, self.device = config, device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world_size > 1 and (config.checkpoint or config.resume):
            raise ValueError(
                "Distributed checkpoint is not implemented in this prototype"
            )
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if model is None:
            model_class = {
                "qwen3": Qwen3Model,
                "qwen3_moe": Qwen3MoeModel,
                "llama": LlamaModel,
            }[config.architecture]
            model = model_class(
                load_model_config(config.model_config, config.architecture)
            )
        model = model.to(device)
        self.assembly = model.parallelize(config.parallel, world_size=self.world_size)
        self.model = self.assembly.model
        optimizer_factory = optimizer_factory or AdamWFactory()
        self.optimizer = optimizer_factory(
            self.model.parameters(), config.learning_rate
        )
        parallel_dict = {
            "tp": config.parallel.tp_size,
            "ep": config.parallel.ep_size,
            "fsdp": config.parallel.data_parallel_shard_size,
            "cp": config.parallel.cp_size,
            "pp": config.parallel.pp_size,
        }
        signature = hashlib.sha256(
            json.dumps(
                {
                    "config": model.model.config.to_dict(),
                    "architecture": model.contract.architecture,
                    "runtime_signature": runtime_signature,
                    "parallel": parallel_dict,
                    "seed": config.seed,
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "accumulation": config.gradient_accumulation_steps,
                    "learning_rate": config.learning_rate,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        scheduler = (
            scheduler_factory(self.optimizer)
            if scheduler_factory is not None
            else NoOpScheduler()
        )
        if data is None:
            data = SyntheticData(self.model.contract.vocab_size, config, device)
            data.stream_rank = self.assembly.runtime.data_rank
            data.stream_size = self.assembly.runtime.data_size
        engine = TrainingEngine(
            self.model,
            self.optimizer,
            scheduler,
            loss or CrossEntropyLoss(),
            self.assembly.runtime,
        )
        checkpoint = CheckpointManager(
            self.model,
            self.optimizer,
            signature,
            adapter=checkpoint_adapter,
            scheduler=scheduler,
            data=data,
        )
        if config.resume is not None:
            checkpoint.load(config.resume)
        super().__init__(
            engine,
            data,
            checkpoint,
            accumulation=config.gradient_accumulation_steps,
            metrics=metrics,
            profiler=profiler,
        )

    def train(self):
        metrics = super().train(self.config.steps)
        if self.config.checkpoint is not None:
            self.checkpoint.save(self.config.checkpoint)
        return metrics if self.rank == 0 else []
