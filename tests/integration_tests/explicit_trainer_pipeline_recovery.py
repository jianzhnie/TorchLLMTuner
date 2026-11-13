"""Production two-rank explicit-model Trainer recovery across independent launches.

Run --phase write then --phase resume with a shared --folder under torchrun.
Qwen3 uses PP2; --model kimi uses EP2 with latent/shared experts.
Uses the synthetic checkpointable dataloader and native DCP. This
is lifecycle acceptance for a small model on the selected device, not corpus-
scale evidence. State comparisons include CPU and selected-device RNG draws.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map_only
from torch.utils.checkpoint import DefaultDeviceType

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import ParallelConfig, PipelineParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.explicit_model_process_recovery import assert_equal
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3", "kimi"), default="qwen3")
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument(
        "--initialization-device", choices=("cpu", "meta"), default="cpu"
    )
    parser.add_argument(
        "--activation-checkpoint", choices=("none", "full"), default="none"
    )
    parser.add_argument(
        "--checkpoint-async-mode",
        choices=("disabled", "async", "async_with_pinned_mem"),
        default="disabled",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--save-step", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.save_step < args.steps:
        parser.error("Expected 0 < --save-step < --steps")
    device = init_distributed()
    previous_checkpoint_device = DefaultDeviceType.get_device_type()
    # torch-npu sets the default to NPU even for an explicitly selected CPU
    # integration run. Non-reentrant AC metadata inference has CPU-only inputs.
    # Use the public checkpoint default-device API for this harness, restoring
    # it after closing resources; accelerator tests retain their real device.
    DefaultDeviceType.set_device_type(device.type)
    trainer = None
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This Trainer recovery check requires two ranks")
        args.folder.mkdir(parents=True, exist_ok=True)
        cfg = LLMTunerConfig(
            model=ModelConfig(
                model_name_or_path="kimi_linear"
                if args.model == "kimi"
                else "qwen3_moe",
                initialization_device=args.initialization_device,
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                arch_overrides={
                    "num_experts": 4,
                    "num_experts_per_token": 2,
                    "moe_intermediate_size": 12,
                    "num_shared_experts": 1,
                    "num_expert_group": 2,
                    "topk_group": 1,
                    "routed_expert_hidden_size": 8,
                    "latent_moe_use_norm": True,
                    "linear_attn_config": {
                        "kda_layers": [1, 2],
                        "full_attn_layers": [],
                        "head_dim": 8,
                        "num_heads": 2,
                        "short_conv_kernel_size": 3,
                    },
                }
                if args.model == "kimi"
                else {
                    "head_dim": 8,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 12,
                    "tie_word_embeddings": False,
                    "mlp_only_layers": [1],
                    "attention_dropout": 0.0,
                },
            ),
            optimizer=OptimizerConfig(learning_rate=3e-4, implementation="for-loop"),
            parallel=ParallelConfig(
                pipeline_parallel_size=1 if args.model == "kimi" else 2,
                expert_parallel_size=2 if args.model == "kimi" else 1,
                pp_config=PipelineParallelConfig(num_pp_microbatches=2),
            ),
            training=TrainingConfig(
                seed=71,
                activation_checkpoint_mode=args.activation_checkpoint,
                steps=args.steps,
                max_seq_len=8,
                global_batch_size=2,
                dump_folder=str(args.folder),
                checkpoint_config=CheckpointConfig(
                    enable=True,
                    interval=args.save_step,
                    keep_latest_k=0,
                    async_mode=args.checkpoint_async_mode,
                ),
            ),
        )
        trainer = Trainer(cfg)
        if len(trainer.model_parts) != 1:
            raise AssertionError("Expected one local model part")
        if args.model == "qwen3" and trainer.model is not None:
            raise AssertionError("Trainer did not assemble a local pipeline stage")
        if args.model == "kimi" and trainer.model is None:
            raise AssertionError("Kimi EP requires an assembled whole model")
        tokens_per_step = (
            cfg.training.global_batch_size
            * cfg.training.max_seq_len
            // (trainer.parallel_dims.dp_shard * trainer.parallel_dims.dp_replicate)
        )
        if args.phase == "resume":
            # Keep the dataset's seed contract fixed while making runtime RNG
            # and destination weights differ from the checkpoint intentionally.
            torch.manual_seed(93 + dist.get_rank())
            random.seed(93 + dist.get_rank())
            np.random.seed(93 + dist.get_rank())
            with torch.no_grad():
                for part in trainer.model_parts:
                    for parameter in part.parameters():
                        parameter.add_(0.125)
            assert trainer.checkpointer.load(args.save_step)
            assert trainer.step == args.save_step
            assert trainer.ntokens_seen == args.save_step * tokens_per_step
            assert all(optimizer.state for optimizer in trainer.optimizer.optimizers)
        iterator = trainer.data_iterator()
        while trainer.step < args.steps:
            trainer.step += 1
            before = next(trainer.model_parts[0].parameters()).detach().clone()
            trainer.train_step(iterator)
            after = next(trainer.model_parts[0].parameters()).detach()
            if not torch.isfinite(after).all() or torch.equal(before, after):
                raise AssertionError("Trainer step produced no finite parameter update")
            assert trainer.ntokens_seen == trainer.step * tokens_per_step
            if args.phase == "write" and trainer.step == args.save_step:
                assert trainer.checkpointer.save(args.save_step)
        next_batch = next(iterator)
        result = {
            "model": [part.state_dict() for part in trainer.model_parts],
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
            "trainer": trainer.state_dict(),
            "dataloader": trainer.dataloader.state_dict(),
            "next_input": next_batch.input_ids,
            "next_labels": next_batch.labels,
            "random": (
                torch.rand(4),
                torch.rand(4, device=device),
                random.random(),
                float(np.random.random()),
            ),
        }
        # Compare state values on one device; accelerator optimizer counters
        # and model tensors can legitimately have different residency policies.
        result = tree_map_only(torch.Tensor, lambda value: value.detach().cpu(), result)
        artifact = args.folder / f"{args.model}-continuous-rank-{dist.get_rank()}.pt"
        if args.phase == "write":
            torch.save(result, artifact)
        else:
            expected = torch.load(artifact, map_location="cpu", weights_only=True)
            assert_equal(result, expected)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"Trainer {args.model} {args.phase}: "
                "updates and lifecycle checks passed",
                flush=True,
            )
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            DefaultDeviceType.set_device_type(previous_checkpoint_device)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
