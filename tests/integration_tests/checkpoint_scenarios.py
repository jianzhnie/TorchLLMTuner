"""Small production Trainer checkpoint scenarios shared by integration drivers.

State capture and exact comparisons reuse the current fresh-process recovery
oracle. Model recipes supply configuration; no pipeline schedule is involved.

Run the default Qwen3 scenario in separate torchrun invocations:
``torchrun --standalone --nproc_per_node=2 -m
tests.integration_tests.checkpoint_scenarios full WORKDIR`` and then the same
command with ``resume WORKDIR``. Uses the runtime-selected device.
Set LLMTUNER_TEST_CHECKPOINT_BACKEND=torch_checkpointing for the optional backend.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.trainer_ema_process_resume import snapshot

SPLIT_STEP = 2
EXTRA_STEPS = 2
VOCAB = 128


def build_config(workdir: str) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={"head_dim": 16, "mlp_only_layers": [0, 1]},
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=-1,
            fsdp_config=FSDPConfig(
                fsdp_implementation=os.environ.get(
                    "LLMTUNER_TEST_FSDP_IMPLEMENTATION", "native"
                )
            ),
        ),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=SPLIT_STEP + EXTRA_STEPS,
            seed=42,
            deterministic=True,
            dump_folder=workdir,
            ema_config=EMAConfig(half_life_fraction=0.5, step_bias=1),
            checkpoint_config=CheckpointConfig(
                enable=True,
                backend=os.environ.get("LLMTUNER_TEST_CHECKPOINT_BACKEND", "dcp"),
                interval=SPLIT_STEP,
                last_save_model_only=False,
            ),
        ),
    )


def kimi_config(workdir: str) -> LLMTunerConfig:
    cfg = build_config(workdir)
    latent = os.environ.get("LLMTUNER_TEST_KIMI_LATENT", "1") == "1"
    cfg.model.model_name_or_path = "kimi_linear"
    cfg.model.arch_overrides = {
        "linear_attn_config": {
            "kda_layers": [1, 2],
            "full_attn_layers": [],
            "head_dim": 16,
            "num_heads": 4,
            "short_conv_kernel_size": 3,
        },
        "routed_expert_hidden_size": 32 if latent else None,
        "latent_moe_use_norm": latent,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 48,
        "num_shared_experts": 1,
        "num_expert_group": 2,
        "topk_group": 1,
    }
    cfg.parallel.expert_parallel_size = int(
        os.environ.get("LLMTUNER_TEST_EP_SIZE", "1")
    )
    return cfg


def train_steps(trainer: Trainer, source, count: int) -> list[float]:
    """Continue the live runtime without rebuilding its consumed data cursor."""
    losses = []
    for _ in range(count):
        trainer.step += 1
        metrics = trainer.train_step(source)
        assert metrics is not None
        losses.append(metrics["loss"])
    return losses


def run_phase(config_factory=build_config) -> None:
    phase, directory = sys.argv[1:]
    if phase not in {"full", "resume"}:
        raise ValueError("Expected full|resume WORKDIR")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config_factory(str(root)))
    rank = trainer.rank
    try:
        if phase == "full":
            source = trainer.data_iterator()
            losses = train_steps(trainer, source, SPLIT_STEP)
            assert trainer.checkpointer.save(SPLIT_STEP)
            torch.save(snapshot(trainer), root / f"split-state-rank-{rank}.pt")
            losses += train_steps(trainer, source, EXTRA_STEPS)
            torch.save(snapshot(trainer), root / f"final-state-rank-{rank}.pt")
            (root / f"losses-rank-{rank}.json").write_text(json.dumps(losses))
        else:
            assert trainer.checkpointer.load(SPLIT_STEP)
            expected = torch.load(
                root / f"split-state-rank-{rank}.pt", weights_only=False
            )
            assert_state_equal(snapshot(trainer), expected, "loaded")
            losses = train_steps(trainer, trainer.data_iterator(), EXTRA_STEPS)
            expected = torch.load(
                root / f"final-state-rank-{rank}.pt", weights_only=False
            )
            assert_state_equal(snapshot(trainer), expected, "continued")
            reference = json.loads((root / f"losses-rank-{rank}.json").read_text())
            assert losses == reference[SPLIT_STEP:]
        print(f"PASS {phase} rank={rank}: complete state and loss", flush=True)
    finally:
        try:
            trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    run_phase()
