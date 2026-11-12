"""Real MoE conversion failure coordination and fresh-model retry on two ranks."""

from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep
from llmtuner.parallel.expert_parallel.model_replacement import resolve_ep_plan
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
        for invalid_option in ("dispatcher", "pre_down"):
            model = make_model(device)
            implementation = "alltoall"
            if rank == 0:
                if invalid_option == "dispatcher":
                    implementation = "unknown"
                else:
                    model.config.experts_implementation = "pre_down"
            parameters = tuple(model.parameters())
            block = model.layers[0].mlp
            rng = torch.get_rng_state().clone()
            caught = None
            try:
                with synchronize_local_errors(
                    "EP plan validation", group=dist.group.WORLD
                ):
                    resolve_ep_plan(model, degree=2, implementation=implementation)
            except (ValueError, UnsupportedCombinationError, RuntimeError) as error:
                caught = error
            assert caught is not None, invalid_option
            if rank == 0:
                assert invalid_option in str(caught), str(caught)
            else:
                assert "EP plan validation failed on another rank" in str(caught)
            assert model.layers[0].mlp is block
            assert all(
                a is b for a, b in zip(parameters, model.parameters(), strict=True)
            )
            assert torch.equal(rng, torch.get_rng_state())
        model = make_model(device)
        plan = resolve_ep_plan(model, degree=2)
        if rank == 0:
            plan = replace(plan, targets=plan.targets * 2)
        caught = None
        try:
            apply_ep(model, cfg, ep_group=dist.group.WORLD, resolved_plan=plan)
        except (ValueError, RuntimeError) as error:
            caught = error
        assert caught is not None
        if rank == 0:
            assert "cover declared MoE slots" in str(caught)
            assert model.layers[0].mlp is plan.targets[0].module
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
