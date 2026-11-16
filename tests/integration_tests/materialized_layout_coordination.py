"""Reject rank-local ownership drift after meta initialization before optimizer.

Run explicitly with two CPU/Gloo ranks and hidden accelerators:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
        tests/integration_tests/materialized_layout_coordination.py
"""

from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed(timeout=timedelta(seconds=30))
    try:
        assert device.type == "cpu" and dist.get_world_size() == 2
        rank = dist.get_rank()
        initialize = Qwen3MoeForCausalLM.initialize_parameters

        def initialize_with_extra_parameter(model):
            initialize(model)
            if rank == 0:
                model.register_parameter(
                    "unexpected", torch.nn.Parameter(torch.ones(1))
                )

        optimizer_entered = False

        def optimizer_sentinel(*args, **kwargs):
            nonlocal optimizer_entered
            optimizer_entered = True
            raise AssertionError("Optimizer entered after ownership drift")

        with TemporaryDirectory() as root:
            config = LLMTunerConfig(
                model=ModelConfig(
                    model_name_or_path="qwen3_moe",
                    initialization_device="meta",
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    arch_overrides={
                        "head_dim": 8,
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "moe_intermediate_size": 8,
                        "mlp_only_layers": [0, 1],
                    },
                ),
                parallel=ParallelConfig(data_parallel_shard_size=2),
                training=TrainingConfig(
                    steps=1,
                    max_seq_len=8,
                    global_batch_size=4,
                    activation_checkpoint_mode="none",
                    dump_folder=root,
                ),
            )
            error = None
            trainer = None
            try:
                with (
                    patch.object(
                        Qwen3MoeForCausalLM,
                        "initialize_parameters",
                        initialize_with_extra_parameter,
                    ),
                    patch.object(assembly, "build_optimizer_stack", optimizer_sentinel),
                ):
                    trainer = Trainer(config)
            except (ValueError, RuntimeError) as caught:
                error = caught
            finally:
                if trainer is not None:
                    trainer.close()
            outcomes = [None, None]
            dist.all_gather_object(
                outcomes, (type(error).__name__, str(error), optimizer_entered)
            )
            assert outcomes[0][0] == "ValueError", outcomes
            assert "registration paths or order" in outcomes[0][1], outcomes
            assert outcomes[1][0] == "RuntimeError", outcomes
            assert "Materialized model layout validation" in outcomes[1][1], outcomes
            assert not any(item[2] for item in outcomes), outcomes
            dist.barrier()
            print(f"rank={rank}: ownership drift rejected before optimizer", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
