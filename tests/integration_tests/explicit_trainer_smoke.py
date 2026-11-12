"""Run the production explicit-model Trainer on a real accelerator.

Run with torchrun --standalone --nproc_per_node=1. This is a short synthetic
training check, not target-scale, performance or recovery acceptance.
"""

import torch
import torch.distributed as dist

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    trainer = None
    try:
        if dist.get_world_size() != 1:
            raise ValueError("This acceptance check requires exactly one rank")
        if device.type == "cpu":
            raise RuntimeError("A real accelerator is required")
        cfg = LLMTunerConfig(
            model=ModelConfig(
                model_name_or_path="qwen3_moe",
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                arch_overrides={
                    "head_dim": 8,
                    "moe_intermediate_size": 8,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                },
            ),
            training=TrainingConfig(steps=3, max_seq_len=8, global_batch_size=1),
        )
        trainer = Trainer(cfg)
        if not isinstance(trainer.model, Qwen3MoeForCausalLM):
            raise AssertionError("Trainer did not select the explicit Qwen3 model")
        before = trainer.model.tok_embeddings.weight.detach().clone()
        trainer.train()
        after = trainer.model.tok_embeddings.weight.detach()
        if trainer.step != 3 or not torch.isfinite(after).all():
            raise AssertionError("Training did not complete three finite updates")
        if torch.equal(before, after):
            raise AssertionError("Training did not update embedding parameters")
        print("Explicit Trainer: three accelerator training steps passed", flush=True)
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
