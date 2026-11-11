"""Minimal trainer for the experiment, with explicit TP gradient ownership."""

import random

import torch
import torch.distributed as dist

from llmtuner.components.loss import cross_entropy_loss, next_token_targets
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.parallel.fully_shard_parallel.policies import (
    disable_fsdp_gradient_division,
)
from llmtuner.trainer.runtime.gradients import (
    allreduce_fsdp_excluded_grads,
    allreduce_replicated_tp_grads,
)

from .assembly import assemble_model
from .checkpoint import CheckpointManager, TrainingState
from .config import ExperimentConfig, load_model_config
from .model import Qwen3MoeModel


class ExperimentTrainer:
    """Own model assembly, optimization, metrics and checkpoint lifecycle."""

    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        model_config = load_model_config(config.model_config)
        if config.seq_len > model_config.max_position_embeddings:
            raise ValueError("seq_len exceeds model max_position_embeddings")
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        dtype = torch.bfloat16 if device.type == "npu" else torch.float32
        model = Qwen3MoeModel(model_config).to(device=device, dtype=dtype)
        self.assembly = assemble_model(
            model, config.parallel, world_size=self.world_size
        )
        self.model = self.assembly.model
        self.model_parts = [self.model]
        self.parallel_dims = self.assembly.topology if self.world_size > 1 else None
        disable_fsdp_gradient_division(self.model)
        self.optimizer = OptimizersContainer(
            OptimizerConfig(
                learning_rate=config.learning_rate, implementation="for-loop"
            ),
            model_parts=[self.model],
        )
        self.training_state = TrainingState(
            device,
            {
                "version": 1,
                "model": model_config.to_dict(),
                "world_size": self.world_size,
                "dtype": str(dtype),
                "device": device.type,
                "degrees": self.assembly.execution.degrees,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "learning_rate": config.learning_rate,
            },
        )
        self.checkpoint = CheckpointManager(
            self.model, self.optimizer, self.training_state
        )
        self.start_step = 0
        if config.resume is not None:
            self.start_step = self.checkpoint.load(config.resume)
            if self.start_step > config.steps:
                raise ValueError("Requested steps precede the checkpoint cursor")

    def _batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_mesh = (
            self.assembly.topology.get_optional_mesh("batch")
            if self.world_size > 1
            else None
        )
        batch_rank = 0 if batch_mesh is None else batch_mesh.get_local_rank()
        generator = torch.Generator().manual_seed(
            self.config.seed + step * 100003 + batch_rank
        )
        ids = torch.randint(
            0,
            self.model.contract.vocab_size,
            (self.config.batch_size, self.config.seq_len),
            generator=generator,
        ).to(self.device)
        labels = next_token_targets(
            ids.reshape(-1), seq_len=self.config.seq_len
        ).reshape_as(ids)
        tp_size = self.config.parallel.tp_size
        tp_rank = 0
        if self.world_size > 1 and tp_size > 1:
            tp_mesh = self.assembly.topology.get_mesh("tp")
            tp_rank = tp_mesh.get_local_rank()
        local_seq = self.config.seq_len // tp_size
        start = tp_rank * local_seq
        positions = torch.arange(self.config.seq_len, device=self.device)
        positions = positions.unsqueeze(0).expand(self.config.batch_size, -1)
        return (
            ids[:, start : start + local_seq],
            labels[:, start : start + local_seq],
            positions,
        )

    def _metrics(self, loss: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
        group = None
        if self.world_size > 1:
            group = self.assembly.topology.get_mesh("loss").get_group()
        total = torch.stack((loss.detach().float(), (labels != -100).sum().float()))
        if group is not None:
            dist.all_reduce(total, group=group)
        return total[0].item(), int(total[1].item())

    def train(self) -> list[dict[str, float | int]]:
        """Run real forward/backward/optimizer steps and return rank0 metrics."""
        metrics = []
        for step in range(self.start_step, self.config.steps):
            ids, labels, positions = self._batch(step)
            self.optimizer.zero_grad(set_to_none=True)
            # Sequence-parallel realizers consume B=1 in rank-major order.
            # Execute independent documents as microbatches, never concatenate
            # rank-local [B,T,H] tensors across the sequence boundary.
            logits = torch.cat(
                [
                    self.model(row.unsqueeze(0), position_ids=pos.unsqueeze(0))
                    for row, pos in zip(ids, positions, strict=True)
                ],
                dim=0,
            )
            tp_group = None
            if self.config.parallel.tp_size > 1:
                tp_group = self.assembly.topology.get_mesh("tp").get_group()
            loss = cross_entropy_loss(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                tp_group=tp_group,
                global_vocab_size=(
                    self.model.contract.vocab_size if tp_group is not None else None
                ),
            )
            _, valid_tokens = self._metrics(loss, labels)
            (loss / valid_tokens).backward()
            allreduce_replicated_tp_grads(self)
            allreduce_fsdp_excluded_grads(self)
            self.optimizer.step()
            self.training_state.next_step = step + 1
            global_loss, token_count = self._metrics(loss, labels)
            metrics.append(
                {
                    "step": step,
                    "loss": global_loss / token_count,
                    "tokens": token_count,
                }
            )
        if self.config.checkpoint is not None:
            self.checkpoint.save(self.config.checkpoint)
        return metrics if self.rank == 0 else []
