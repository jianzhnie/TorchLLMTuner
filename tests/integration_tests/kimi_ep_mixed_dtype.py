"""Real two-rank NPU Kimi EP with BF16 w1/w3 and FP32 w2 under autocast.

Run explicitly with torchrun --standalone --nproc_per_node=2. This small-model
check compares against the original individual-expert implementation.
"""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.moe import KimiSparseMoE
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        rank = dist.get_rank()
        failures = []

        def compare(actual, expected, name):
            try:
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
            except AssertionError as error:
                failures.append(f"{name}: {error}")

        torch.manual_seed(17)
        config = KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_intermediate_size=12,
            num_shared_experts=0,
            num_expert_group=1,
            topk_group=1,
            hidden_act="silu",
        )
        source = KimiSparseMoE(config).to(device=device, dtype=torch.bfloat16)
        source.gate.float()
        source.gate.e_score_correction_bias.requires_grad_(False)
        for expert in source.experts:
            expert.w2.float()
        reference = deepcopy(source)
        model = nn.Module()
        layer = nn.Module()
        layer.block_sparse_moe = source
        model.layers = nn.ModuleList([layer])
        assert swap_moe_blocks(model, ep_group=dist.group.WORLD) == 1
        candidate = layer.block_sparse_moe
        grouped = candidate.routed_experts.inner_experts
        assert grouped.w1_EFD.dtype == grouped.w3_EFD.dtype == torch.bfloat16
        assert grouped.w2_EDF.dtype == torch.float32
        torch.manual_seed(91 + rank)
        inputs = torch.randn(1, 12, 8, device=device, dtype=torch.bfloat16)
        inputs.requires_grad_()
        gathered = [torch.empty_like(inputs) for _ in range(2)]
        dist.all_gather(gathered, inputs.detach())
        # EP expert GEMMs consume tokens from both ranks together. Use one
        # independent full-batch reference, rather than summing already-rounded
        # BF16 GEMM gradients from two smaller reference batches.
        other = torch.cat(gathered, dim=1).requires_grad_()
        with torch.autocast("npu", dtype=torch.bfloat16):
            actual, expected = candidate(inputs), reference(other)
        compare(actual, expected.chunk(2, dim=1)[rank], "output")
        gradient = torch.randn_like(actual)
        dist.all_gather(gathered, gradient)
        actual.backward(gradient)
        expected.backward(torch.cat(gathered, dim=1))
        compare(inputs.grad, other.grad.chunk(2, dim=1)[rank], "input gradient")
        dist.all_reduce(candidate.router.gate.weight.grad)
        compare(
            candidate.router.gate.weight.grad,
            reference.gate.weight.grad,
            "router gradient",
        )
        for source_name, target_name in (
            ("w1", "w1_EFD"),
            ("w2", "w2_EDF"),
            ("w3", "w3_EFD"),
        ):
            parameters = [
                getattr(expert, source_name).weight for expert in reference.experts
            ]
            expected_weight = torch.stack(parameters).chunk(2)[rank]
            expected_grad = torch.stack([p.grad for p in parameters]).chunk(2)[rank]
            parameter = getattr(grouped, target_name)
            assert parameter.grad.dtype == parameter.dtype
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0
            torch.testing.assert_close(parameter, expected_weight, rtol=0, atol=0)
            compare(parameter.grad, expected_grad, f"{target_name} gradient")
        initial = {name: p.detach().clone() for name, p in grouped.named_parameters()}
        optimizer = torch.optim.AdamW(
            candidate.parameters(), lr=1e-3, foreach=False, fused=False
        )
        optimizer.step()
        for name, parameter in grouped.named_parameters():
            assert not torch.equal(parameter, initial[name])
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state[parameter][key]
                assert state.dtype == parameter.dtype and torch.isfinite(state).all()
                assert state.count_nonzero() > 0
        # Agree only after completing every collective, so a numerical failure
        # on one rank does not strand peers in a later gradient all-reduce.
        failed = torch.tensor(bool(failures), device=device, dtype=torch.int32)
        dist.all_reduce(failed)
        for failure in failures:
            print(f"rank {rank}: {failure}", flush=True)
        assert failed.item() == 0, "Kimi mixed-dtype EP numerical checks failed"
        print(f"rank {rank}: Kimi mixed-dtype EP forward/backward/update passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
