"""Two-rank real RMSNorm parameter-gradient partition contract."""

import torch
import torch.distributed as dist

from tests.integration_tests.alignment_norm_control import (
    PartitionedNormWeightGradient,
    native_norm,
)
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size != 2:
            raise ValueError("This contract requires two ranks")
        torch.manual_seed(81)
        dtype = torch.bfloat16 if device.type == "npu" else torch.float64
        full = torch.randn(2, 16, 128, device=device, dtype=dtype)
        upstream = torch.randn_like(full)
        weight = torch.ones(128, device=device, dtype=dtype, requires_grad=True)
        reference_weight = weight.detach().clone().requires_grad_()
        local = full.flatten(0, 1).chunk(size)[rank].detach().requires_grad_()
        reference = full.chunk(size, dim=1)[rank].detach().contiguous().requires_grad_()
        output = PartitionedNormWeightGradient.apply(
            local, weight, 1e-6, dist.group.WORLD, (2, 16)
        )
        output.backward(upstream.flatten(0, 1).chunk(size)[rank])
        expected = native_norm(reference, reference_weight, 1e-6)
        expected.backward(upstream.chunk(size, dim=1)[rank].contiguous())
        dist.all_reduce(weight.grad)
        dist.all_reduce(reference_weight.grad)
        torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)
        pieces = [torch.empty_like(reference.grad) for _ in range(size)]
        dist.all_gather(pieces, reference.grad.contiguous())
        expected_input = torch.cat(pieces, dim=1).flatten(0, 1).chunk(size)[rank]
        torch.testing.assert_close(local.grad, expected_input, rtol=0, atol=0)
        print(f"PASS rank{rank}: norm input and repartitioned weight gradients exact")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
