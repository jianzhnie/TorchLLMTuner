"""Reject rank-local EP geometry errors before production TP/PP transforms.

Run with four ranks for TP2/EP2. This validates failure coordination and plan
purity on small explicit models, not numerical or performance equivalence.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.parallelize import parallelize_model
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=60))
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This check requires four ranks for TP2/EP2")
        cfg = ParallelConfig(
            tensor_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_shard_size=2,
        )
        topology = ParallelTopology.from_config(cfg, dist.get_world_size())
        mesh = build_device_mesh(topology)
        for failing_rank in (0, 3):
            config = Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                moe_intermediate_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                num_experts=3 if dist.get_rank() == failing_rank else 4,
                num_experts_per_tok=2,
            )
            model = Qwen3MoeForCausalLM(config).to(device)
            parameters = tuple(model.parameters())
            state = {key: value.clone() for key, value in model.state_dict().items()}
            forwards = tuple(module.forward for module in model.modules())
            rng = torch.get_rng_state().clone()
            try:
                parallelize_model(model, cfg=cfg, mesh=mesh, parallel_dims=topology)
            except (ValueError, RuntimeError) as error:
                expected = (
                    "EP degree 2 does not divide num_experts=3"
                    if dist.get_rank() == failing_rank
                    else "Expert parallel global plan validation failed on another rank"
                )
                assert expected in str(error), error
            else:
                raise AssertionError("Invalid EP ownership reached parallel execution")
            assert all(
                a is b for a, b in zip(parameters, model.parameters(), strict=True)
            )
            assert forwards == tuple(module.forward for module in model.modules())
            assert torch.equal(rng, torch.get_rng_state())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: global EP errors stop all TP/EP ranks before mutation")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
