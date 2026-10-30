"""Check opt-in norm precision against the actual NPU baseline on sharded grads."""

import importlib

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from tests.caps import require_env


def main():
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    import os

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        mesh = init_device_mesh("npu", (dist.get_world_size(),))
        torch.manual_seed(37)
        parameters = []
        for shape in ((257, 32), (128, 32), (3, 32)):
            gradient = distribute_tensor(
                torch.randn(shape, dtype=torch.bfloat16, device="npu"), mesh, [Shard(0)]
            )
            parameter = torch.nn.Parameter(torch.zeros_like(gradient))
            parameter.grad = gradient
            parameters.append(parameter)
        for order in (1.0, 2.0):
            reference = torch.nn.utils.get_total_norm(
                [p.grad for p in parameters], order, foreach=True
            )
            actual = clip_grad_norm_(
                parameters, 0, norm_type=order, norm_precision="parameter"
            )
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            if dist.get_rank() == 0:
                print(
                    f"PASS parameter precision p={order}: {actual.item()}", flush=True
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
