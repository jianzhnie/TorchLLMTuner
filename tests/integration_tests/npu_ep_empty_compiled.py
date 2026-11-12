"""Two real NPU ranks: fullgraph EP with alternating empty expert owners.

Run with torchrun --standalone --nproc-per-node=2 -m
tests.integration_tests.npu_ep_empty_compiled. Module correctness only; no FSDP,
target-model or performance acceptance is implied.
"""

import copy
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import (
    resolve_ep_plan,
    swap_moe_blocks,
)
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(52)
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        source = Qwen3MoeForCausalLM(config).to(device=device, dtype=torch.bfloat16)
        source.layers[0].mlp.gate.float()
        with torch.no_grad():
            gate = source.layers[0].mlp.gate.weight
            gate.copy_(gate.new_tensor([0.04, 0.02, -0.02, -0.04])[:, None])
        local = copy.deepcopy(source)
        assert swap_moe_blocks(local, router_aux_loss_coef=0.0) == 1
        plan = resolve_ep_plan(source, degree=2)
        assert len(plan.targets) == 1
        assert plan.targets[0].module is source.layers[0].mlp
        with patch(
            "llmtuner.parallel.expert_parallel.model_replacement.find_moe_candidates",
            side_effect=AssertionError("EP execution rediscovered model targets"),
        ):
            assert (
                swap_moe_blocks(
                    source,
                    ep_group=dist.group.WORLD,
                    router_aux_loss_coef=0.0,
                    resolved_plan=plan,
                )
                == 1
            )
        reference = local.layers[0].mlp
        candidate = source.layers[0].mlp
        candidate.compile(backend="aot_eager", fullgraph=True)
        optimizer = torch.optim.SGD(candidate.parameters(), lr=0.001)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            for owner in (0, 1, 0):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                values = torch.randn(12, 16, device=device, dtype=torch.bfloat16)
                values = (values.abs() + 0.1) * (1 if owner == 0 else -1)
                values.requires_grad_()
                inputs = values.detach().chunk(2)[rank].clone().requires_grad_()
                with torch.no_grad():
                    _, ids, _ = reference.router(values)
                    assert (ids // 2 == owner).all(), ids
                expected = reference(values)
                output = candidate(inputs)
                torch.testing.assert_close(
                    output, expected.chunk(2)[rank], rtol=5e-3, atol=5e-3
                )
                upstream = torch.randn_like(expected)
                expected.backward(upstream)
                output.backward(upstream.chunk(2)[rank])
                torch.testing.assert_close(
                    inputs.grad, values.grad.chunk(2)[rank], rtol=5e-3, atol=5e-3
                )
                dist.all_reduce(candidate.router.gate.weight.grad)
                targets = dict(reference.named_parameters())
                for name, parameter in candidate.named_parameters():
                    assert parameter.grad is not None, name
                    target = targets[name].grad
                    if name.startswith("routed_experts."):
                        target = target.chunk(2)[rank]
                        assert bool(parameter.grad.count_nonzero()) == (rank == owner)
                    torch.testing.assert_close(
                        parameter.grad, target, rtol=5e-3, atol=5e-3
                    )
                optimizer.step()
                reference_optimizer.step()
                for name, parameter in candidate.named_parameters():
                    target = targets[name]
                    if name.startswith("routed_experts."):
                        target = target.chunk(2)[rank]
                    torch.testing.assert_close(parameter, target, rtol=5e-3, atol=5e-3)
                print(
                    f"rank={rank} owner={owner}: fullgraph EP empty-owner "
                    "output/input-gradient/expert-gradient/update passed",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
