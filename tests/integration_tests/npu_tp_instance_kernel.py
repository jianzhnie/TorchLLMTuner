"""Two NPU ranks: explicit MoE forward must execute the TP sequence boundary.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/npu_tp_instance_kernel.py

Compare against the same local expert kernel followed by an explicit HCCL
sum and sequence split. This tests boundary composition, not unsharded BF16
arithmetic equivalence or target-model training acceptance.
"""

import os
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.models.hf.model_setup import forward_torchtitan_moe
from llmtuner.parallel.tensor_parallel.tensor_parallel import realize_moe_blocks_tp
from tests.caps import require_env


class Gate(nn.Module):
    def forward(self, hidden):
        count = hidden.shape[0]
        indices = torch.tensor([0, 1], device=hidden.device).expand(count, 2)
        scores = torch.full((count, 2), 0.5, device=hidden.device)
        return None, scores, indices


def main():
    require_env("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        torch.manual_seed(37)
        block = nn.Module()
        block.gate = Gate()
        block.top_k = 2
        block.experts = nn.Module()
        block.experts.num_experts = 4
        block.experts.gate_up_proj = nn.Parameter(
            torch.randn(4, 128, 32, device="npu", dtype=torch.bfloat16) * 0.1
        )
        block.experts.down_proj = nn.Parameter(
            torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16) * 0.1
        )
        block.forward = MethodType(forward_torchtitan_moe, block)
        realize_moe_blocks_tp(
            [("mlp", block)],
            tp_size=2,
            tp_rank=dist.get_rank(),
            group=dist.group.WORLD,
        )
        inputs = torch.randn(1, 16, 32, device="npu", dtype=torch.bfloat16)
        with torch.no_grad():
            expected = block._tp_moe_original_forward(inputs)
            dist.all_reduce(expected)
            expected = expected.chunk(2, dim=1)[dist.get_rank()]
            actual = block(inputs.chunk(2, dim=1)[dist.get_rank()].contiguous())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        probe = torch.randn_like(inputs)
        full_input = inputs.detach().clone().requires_grad_()
        parameters = (block.experts.gate_up_proj, block.experts.down_proj)
        reference_grads = torch.autograd.grad(
            block._tp_moe_original_forward(full_input),
            (full_input, *parameters),
            probe,
        )
        input_gradient = reference_grads[0].clone()
        dist.all_reduce(input_gradient)
        local_input = (
            inputs.chunk(2, dim=1)[dist.get_rank()].detach().clone().requires_grad_()
        )
        actual_grads = torch.autograd.grad(
            block(local_input),
            (local_input, *parameters),
            probe.chunk(2, dim=1)[dist.get_rank()].contiguous(),
        )
        expected_grads = (
            input_gradient.chunk(2, dim=1)[dist.get_rank()],
            *reference_grads[1:],
        )
        for observed, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        print(
            f"PASS rank={dist.get_rank()} NPU instance kernel TP forward/backward",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
