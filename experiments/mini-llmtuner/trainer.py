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
from .model import Qwen3MoeModel


class BaseTrainer:
    """Own lifecycle only; model parallelism is completed before training starts."""

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        *,
        loss=None,
        metrics=None,
        scheduler=None,
        profiler=None,
    ) -> None:
        self.config, self.device = config, device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.loss_fn = loss or CrossEntropyLoss()
        self.metrics = metrics or TokenMetrics()
        self.scheduler = scheduler or NoOpScheduler()
        self.profiler = profiler or NoOpProfiler()

    def train(self):
        raise NotImplementedError


class ExperimentTrainer(BaseTrainer):
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        super().__init__(config, device)
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
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        self.checkpoint = CheckpointManager(self.model, self.optimizer, signature)
        self.start_step = 0
        if config.resume is not None:
            self.start_step = self.checkpoint.load(config.resume)

    def _batch(self, step: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + step)
        ids = torch.randint(
            0,
            self.model.contract.vocab_size,
            (self.config.batch_size, self.config.seq_len),
            generator=generator,
        )
        labels = ids.roll(-1, dims=1)
        labels[:, -1] = -100
        positions = torch.arange(self.config.seq_len).expand(self.config.batch_size, -1)
        return {
            "input_ids": ids.to(self.device),
            "labels": labels.to(self.device),
            "position_ids": positions.to(self.device),
        }

    def train(self):
        metrics = []
        for step in range(self.start_step, self.config.steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = None
            batch = None
            for micro_step in range(self.config.gradient_accumulation_steps):
                batch = self.model.preprocess_inputs(
                    self._batch(
                        step * self.config.gradient_accumulation_steps + micro_step
                    )
                )
                logits = self.model(
                    batch["input_ids"], position_ids=batch["position_ids"]
                )
                micro_loss = self.loss_fn(logits, batch["labels"])
                micro_loss.backward()
                loss = micro_loss if loss is None else loss + micro_loss
            loss = loss / self.config.gradient_accumulation_steps
            self.optimizer.step()
            self.scheduler.step()
            self.profiler.step()
            metrics.append(self.metrics.record(step, loss, batch["labels"]))
            self.checkpoint.step = step + 1
        if self.config.checkpoint is not None:
            self.checkpoint.save(self.config.checkpoint)
        return metrics if self.rank == 0 else []
