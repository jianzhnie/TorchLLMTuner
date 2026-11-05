"""Shared decoder slots with explicit prefetch, gradients and SGD updates.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/fsdp_shared_decoder.py

Use ASCEND_RT_VISIBLE_DEVICES="" LLMTUNER_TEST_DEVICE=cpu for CPU/Gloo.
"""

import argparse
import copy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp_to_decoder,
    force_sum_grad_reduction,
)
from tests.integration_tests.harness import init_distributed


class SharedDecoder(nn.Module):
    def __init__(self, *, adjacent: bool = False):
        super().__init__()
        self.enable_weight_tying = False
        self.tok_embeddings = nn.Linear(4, 4, bias=False)
        shared = nn.Linear(4, 4, bias=False)
        separate = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList(
            [shared, shared, separate] if adjacent else [shared, separate, shared]
        )
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, value):
        value = self.tok_embeddings(value)
        for layer in self.layers:
            value = layer(value).tanh()
        return self.lm_head(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacent", action="store_true")
    parser.add_argument("--prefetch", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=90))
    try:
        world_size = dist.get_world_size()
        assert world_size == 2
        torch.manual_seed(730)
        model = SharedDecoder(adjacent=args.adjacent).to(device)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh(
            device.type, (world_size,), mesh_dim_names=("dp_shard",)
        )
        apply_fsdp_to_decoder(
            model,
            mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            forward_prefetch_count=args.prefetch,
            backward_prefetch_count=args.prefetch,
        )
        if dist.get_backend() != "nccl":
            force_sum_grad_reduction(model)
        assert model.layers[0] is model.layers[1 if args.adjacent else 2]
        assert isinstance(model.layers[0].weight, DTensor)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, foreach=False)
        oracle = torch.optim.SGD(reference.parameters(), lr=0.03, foreach=False)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            oracle.zero_grad(set_to_none=True)
            inputs = [
                (torch.arange(12, device=device).reshape(3, 4).float() / 7)
                + rank / 3
                + step / 5
                for rank in range(world_size)
            ]
            expected = [reference(value) for value in inputs]
            output = model(inputs[dist.get_rank()])
            torch.testing.assert_close(output, expected[dist.get_rank()])
            output.square().sum().backward()
            sum(value.square().sum() for value in expected).backward()
            for actual, dense in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                gradient = actual.grad.full_tensor()
                assert gradient.abs().max().item() > 0
                torch.testing.assert_close(gradient, dense.grad, atol=2e-6, rtol=2e-5)
            optimizer.step()
            oracle.step()
            for actual, dense in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    actual.full_tensor(), dense, atol=2e-6, rtol=2e-5
                )
        print(
            f"PASS rank={dist.get_rank()}: adjacent={args.adjacent}, "
            f"prefetch={args.prefetch}, three SGD updates"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
