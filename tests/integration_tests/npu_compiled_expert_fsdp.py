"""Two NPU ranks: compiled BF16 experts under native FSDP.

Run with torchrun --standalone --nproc-per-node=2 -m
tests.integration_tests.npu_compiled_expert_fsdp. This checks the kernel/FSDP
boundary, not production parallel assembly, TP, Inductor or target-model parity.
"""

import copy

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from llmtuner.models.common.moe.experts import GroupedExperts
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        torch.manual_seed(77)
        reference = GroupedExperts(16, 24, 4, use_grouped_mm=False).to(
            device=device, dtype=torch.bfloat16
        )
        for parameter in reference.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        actual = copy.deepcopy(reference)
        actual.use_grouped_mm = True
        actual.compile(backend="aot_eager", fullgraph=True)
        fully_shard(actual, mesh=init_device_mesh("npu", (2,)))
        assert all(isinstance(p, DTensor) for p in actual.parameters())
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.02, momentum=0.9)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.02, momentum=0.9
        )
        rank = dist.get_rank()
        tokens = torch.tensor(
            [2, 0, 3, 1] if rank == 0 else [0, 4, 1, 1], device=device
        )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            # Distinct data and empty-expert ownership exercise the actual
            # DP mean; a missing reduction must not pass this comparison.
            torch.manual_seed(100 + step * 2 + rank)
            inputs = torch.randn(6, 16, device=device, dtype=torch.bfloat16)
            gathered = [torch.empty_like(inputs) for _ in range(2)]
            dist.all_gather(gathered, inputs)
            assert not torch.equal(gathered[0], gathered[1])
            inputs.requires_grad_()
            storage = inputs.new_empty(6, 24)
            other = storage[:, 1:17]
            other.copy_(inputs.detach())
            other.requires_grad_()
            assert other.storage_offset() == 1
            expected = reference(inputs, tokens)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profiler:
                output = actual(other, tokens)
            assert any(
                event.key == "llmtuner::aligned_npu_grouped_mm"
                for event in profiler.key_averages()
            ), "compiled aligned grouped GEMM was not executed"
            torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            output.backward(upstream)
            torch.testing.assert_close(other.grad, inputs.grad, rtol=5e-3, atol=5e-3)
            for target in reference.parameters():
                averaged = target.grad.float()
                dist.all_reduce(averaged)
                target.grad.copy_(averaged.div_(dist.get_world_size()))
            for parameter, target in zip(
                actual.parameters(), reference.parameters(), strict=True
            ):
                assert isinstance(parameter.grad, DTensor)
                gradient = parameter.grad.full_tensor()
                assert gradient.count_nonzero() > 0
                torch.testing.assert_close(gradient, target.grad, rtol=5e-3, atol=5e-3)
            optimizer.step()
            reference_optimizer.step()
            for parameter, target in zip(
                actual.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    parameter.full_tensor(), target, rtol=5e-3, atol=5e-3
                )
                torch.testing.assert_close(
                    optimizer.state[parameter]["momentum_buffer"].full_tensor(),
                    reference_optimizer.state[target]["momentum_buffer"],
                    rtol=5e-3,
                    atol=5e-3,
                )
            print(
                f"rank={dist.get_rank()} step={step + 1}: compiled BF16 "
                "FSDP distinct-rank output/input-gradient/mean-gradient/"
                "update/momentum passed",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
