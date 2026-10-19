"""Production CPU RMSNorm DTensor contracts against native HF autograd.

Run with torchrun --nproc_per_node=2 (one mesh axis) or 4 (two axes).
Set LLMTUNER_TEST_COMPILE_BACKEND=inductor to check actual compilation.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from llmtuner.models.hf.normalization import rms_norm


def main():
    dist.init_process_group("gloo")
    try:
        world_size = dist.get_world_size()
        assert world_size in (2, 4)
        mesh = init_device_mesh("cpu", (2, 2) if world_size == 4 else (2,))
        layouts = (
            [(Replicate(), Replicate()), (Shard(0), Replicate()), (Shard(0), Shard(0))]
            if world_size == 4
            else [(Replicate(),), (Shard(0),), (Shard(1),)]
        )
        backend = os.environ.get("LLMTUNER_TEST_COMPILE_BACKEND")
        for placements in layouts:
            torch.manual_seed(42)
            dense = torch.randn(8, 8, requires_grad=True)
            reference = Qwen3RMSNorm(8, eps=1e-6)
            upstream = torch.randn_like(dense)
            value = distribute_tensor(dense.detach(), mesh, placements).requires_grad_()
            weight = distribute_tensor(
                reference.weight.detach(), mesh, [Replicate()] * mesh.ndim
            ).requires_grad_()
            if backend:
                from torch._dynamo.utils import counters

                torch._dynamo.reset()
                counters.clear()
                compute = torch.compile(rms_norm, backend=backend, fullgraph=True)
            else:
                compute = rms_norm
            output = compute(value, weight, reference.variance_epsilon)
            expected = reference(dense)
            torch.testing.assert_close(output.full_tensor(), expected)
            output.backward(distribute_tensor(upstream, mesh, placements))
            expected.backward(upstream)
            torch.testing.assert_close(value.grad.full_tensor(), dense.grad)
            torch.testing.assert_close(weight.grad.full_tensor(), reference.weight.grad)
            if all(p in (Shard(0), Replicate()) for p in placements):
                assert weight.grad.placements == tuple(
                    Partial() if p == Shard(0) else Replicate() for p in placements
                )
            if placements == (Shard(1),):
                assert output.placements == (Replicate(),)
            if backend:
                assert counters["stats"]["unique_graphs"] > 0
            print("PASS production aten norm", dist.get_rank(), placements, backend)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
