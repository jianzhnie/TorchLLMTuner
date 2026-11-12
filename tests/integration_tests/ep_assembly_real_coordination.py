"""Real MoE conversion failure coordination and fresh-model retry on two ranks."""

from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from tests.integration_tests.harness import init_distributed


def make_model(device: torch.device) -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
        max_position_embeddings=32,
    )
    config._experts_implementation = "eager"
    config._attn_implementation = "eager"
    torch.manual_seed(81)
    return Qwen3MoeForCausalLM(config).to(device)


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        cfg = ParallelConfig(expert_parallel_size=2)
        model = torch.nn.Linear(2, 2).to(device) if rank == 0 else make_model(device)
        caught = None
        try:
            apply_ep(model, cfg, ep_group=dist.group.WORLD)
        except (UnsupportedCombinationError, RuntimeError) as error:
            caught = error
        assert caught is not None
        if rank == 0:
            assert isinstance(caught, UnsupportedCombinationError)
        else:
            assert "EP assembly validation failed on another rank" in str(caught)
        # A successful peer may already be converted; no rollback is promised.
        # Both ranks retry with fresh models and exercise real all-to-all autograd.
        model = make_model(device)
        apply_ep(model, cfg, ep_group=dist.group.WORLD)
        block = model.layers[0].mlp
        inputs = torch.randn(8, 16, device=device, requires_grad=True)
        output = block(inputs)
        assert torch.isfinite(output).all() and output.count_nonzero() > 0
        output.square().sum().backward()
        assert inputs.grad is not None and inputs.grad.count_nonzero() > 0
        experts = block.routed_experts.inner_experts
        assert experts.w1_EFD.shape[0] == 2
        assert experts.w1_EFD.grad is not None
        assert torch.isfinite(experts.w1_EFD.grad).all()
        assert experts.w1_EFD.grad.count_nonzero() > 0
        dist.barrier()
        print(f"rank={rank}: real EP failure, fresh retry and backward passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
