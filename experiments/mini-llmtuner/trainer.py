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
from .model import Qwen3MoeModel


class BaseTrainer:
    """Lifecycle only: execution and data are injected, never constructed here."""

    def __init__(
        self, engine, data, checkpoint, *, accumulation=1, metrics=None, profiler=None
    ):
        self.engine, self.data, self.checkpoint = engine, data, checkpoint
        self.accumulation = accumulation
        self.metrics = metrics or TokenMetrics()
        self.profiler = profiler or NoOpProfiler()
        self.start_step = checkpoint.step

    def train(self, steps):
        history = []
        self.engine.model.train()
        for step in range(self.checkpoint.step, steps):
            batches = [self.data.next_batch() for _ in range(self.accumulation)]
            result = self.engine.train_step(batches)
            labels = torch.cat([batch["labels"].flatten() for batch in batches])
            history.append(
                self.metrics.record(
                    step, torch.tensor(result.loss_sum / result.targets), labels
                )
            )
            self.checkpoint.step = step + 1
            self.profiler.step()
        return history


class ExperimentTrainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        self.config, self.device = config, device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        model = Qwen3MoeModel(load_model_config(config.model_config)).to(device)
        self.assembly = model.parallelize(config.parallel, world_size=self.world_size)
        self.model = self.assembly.model
        optimizer_factory = getattr(config, "optimizer_factory", None) or AdamWFactory()
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
                    "config": load_model_config(config.model_config).to_dict(),
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
        scheduler = NoOpScheduler()
        data = SyntheticData(self.model.contract.vocab_size, config, device)
        engine = TrainingEngine(
            self.model, self.optimizer, scheduler, CrossEntropyLoss()
        )
        checkpoint = CheckpointManager(
            self.model, self.optimizer, signature, scheduler=scheduler, data=data
        )
        if config.resume is not None:
            checkpoint.load(config.resume)
        super().__init__(
            engine, data, checkpoint, accumulation=config.gradient_accumulation_steps
        )

    def train(self):
        metrics = super().train(self.config.steps)
        if self.config.checkpoint is not None:
            self.checkpoint.save(self.config.checkpoint)
        return metrics if self.rank == 0 else []
