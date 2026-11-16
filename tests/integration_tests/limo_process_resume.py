"""Small-model LIMO training and exact fresh-process recovery on real NPUs.

Run --phase write, then --phase resume with identical arguments and --folder.
The write process trains to --steps, saving the --save-step boundary. Resume
loads that boundary and compares every subsequent loss/grad norm, full state,
and next input batch. Defaults exercise 100 steps and 50 -> 100 recovery.
This is corpus/runtime acceptance, not a production model or throughput test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.trainer_ema_process_resume import snapshot


def on_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(on_cpu(item) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-step", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--dataset",
        default="/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl",
    )
    parser.add_argument(
        "--tokenizer", default="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B"
    )
    args = parser.parse_args()
    if not 0 < args.save_step < args.steps:
        parser.error("Require 0 < save-step < steps")
    device = init_distributed()
    trainer = None
    try:
        if device.type != "npu":
            raise RuntimeError("This acceptance check requires real Ascend NPUs")
        rank, world = dist.get_rank(), dist.get_world_size()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        cfg = LLMTunerConfig(
            model=ModelConfig(
                model_name_or_path="qwen3_moe",
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                arch_overrides={
                    "head_dim": 8,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 16,
                    "mlp_only_layers": [0, 1],
                },
            ),
            parallel=ParallelConfig(data_parallel_shard_size=world),
            training=TrainingConfig(
                steps=args.steps,
                max_seq_len=args.seq_len,
                global_batch_size=world,
                seed=42,
                deterministic=True,
                dump_folder=str(args.folder),
                dataloader_config=DataloaderConfig(
                    dataset="local_jsonl_sft",
                    dataset_path=args.dataset,
                    tokenizer_path=args.tokenizer,
                    prompt_field="question",
                    response_field="solution",
                    packing="first_fit",
                    max_num_documents=1,
                ),
                ema_config=EMAConfig(decay=0.9),
            ),
        )
        cfg.optimizer.lr_scheduler_config.total_steps = args.steps
        cfg.optimizer.lr_scheduler_config.decay_ratio = 1.0
        cfg.checkpoint.enable = True
        cfg.checkpoint.interval = args.save_step
        cfg.checkpoint.keep_latest_k = 0
        cfg.checkpoint.last_save_model_only = False
        cfg.checkpoint.load_step = args.save_step if args.phase == "resume" else 0
        trainer = Trainer(cfg)
        metrics = []
        step = trainer.train_step

        def record_step(iterator):
            result = step(iterator)
            if trainer.step > args.save_step:
                assert result is not None
                metrics.append({key: result[key] for key in ("loss", "grad_norm")})
            return result

        trainer.train_step = record_step
        load = trainer.checkpointer.load

        def checked_load(*positional, **keywords):
            loaded = load(*positional, **keywords)
            expected = args.save_step if args.phase == "resume" else 0
            assert trainer.step == expected, (loaded, trainer.step, expected)
            if args.phase == "resume":
                assert loaded
            return loaded

        trainer.checkpointer.load = checked_load
        trainer.train()
        assert trainer.step == args.steps
        state = on_cpu(snapshot(trainer))
        batch = next(trainer.data_iterator())
        result = {
            "state": state,
            "metrics": metrics,
            "next_input": on_cpu(batch.input_ids),
            "next_labels": on_cpu(batch.labels),
        }
        artifact = args.folder / f"continuous-rank{rank}.pt"
        if args.phase == "write":
            torch.save(result, artifact)
        else:
            expected = torch.load(artifact, map_location="cpu", weights_only=False)
            assert_state_equal(result, expected)
        dist.barrier()
        print(f"PASS LIMO {args.phase} rank={rank} steps={args.steps}", flush=True)
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
