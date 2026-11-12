"""Run the minimal Qwen3-MoE experiment under Python or torchrun."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .config import ExperimentConfig, ParallelConfig
from .trainer import ExperimentTrainer


def _device() -> torch.device:
    if os.environ.get("LLMTUNER_TEST_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--dp-shard-size", type=int, default=-1)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    device = _device()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device.type != "cpu":
        getattr(torch, device.type).set_device(device)
    if world_size > 1:
        dist.init_process_group("gloo")
    try:
        config = ExperimentConfig(
            parallel=ParallelConfig(
                tensor_parallel_size=args.tp_size,
                expert_parallel_size=args.ep_size,
                context_parallel_size=args.cp_size,
                pipeline_parallel_size=args.pp_size,
                data_parallel_shard_size=(
                    1 if args.dp_shard_size < 1 else args.dp_shard_size
                ),
            ),
            steps=args.steps,
            model_config=args.model_config,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            checkpoint=args.checkpoint,
            resume=args.resume,
        )
        trainer = ExperimentTrainer(config, device)
        if trainer.rank == 0:
            print(
                json.dumps(
                    {
                        "architecture": trainer.model.contract.architecture,
                        "operations": trainer.assembly.execution.operations,
                        "targets": len(trainer.assembly.execution.targets),
                        "degrees": trainer.assembly.execution.degrees,
                        "dtype": str(next(trainer.model.parameters()).dtype),
                    }
                ),
                flush=True,
            )
        metrics = trainer.train()
        if metrics:
            print(json.dumps(metrics, indent=2), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
