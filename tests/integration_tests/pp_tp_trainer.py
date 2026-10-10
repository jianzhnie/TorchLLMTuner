"""Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=4``.

Exercises a real PP=2, TP=2 trainer step and checks that every stage reports
the last stage's finite, positive cross entropy. Also checks that closing a
programmatic Trainer leaves the caller's process group usable.
"""

import math

import torch
import torch.distributed as dist

from llmtuner.config import MetricsConfig
from llmtuner.trainer import (
    LLMTunerConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer


def main() -> None:
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
            data_parallel_shard_size=1,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=4,
            max_seq_len=16,
            steps=1,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )
    trainer = Trainer(cfg)
    assert trainer.world_size == 4
    metrics = trainer.train_step(trainer.data_iterator())
    assert metrics is not None
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0
    assert math.isclose(metrics["loss"], metrics["max_loss"], abs_tol=1e-5)

    gathered = [None] * trainer.world_size
    dist.all_gather_object(gathered, (metrics["loss"], metrics["max_loss"]))
    assert all(value == gathered[0] for value in gathered), gathered

    trainer.close()
    probe = torch.ones(1)
    dist.all_reduce(probe)
    assert probe.item() == trainer.world_size
    if trainer.rank == 0:
        print("PP+TP trainer checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
