"""Check PP/TP/EP fp16 clipping against an independent full-vector norm.

Run with two CPU/Gloo ranks:
    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/parallel_grad_norm_equivalence.py

This covers norm reductions, not a full model's parallel layout or device kernels.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.accelerator.collectives import clip_grad_norm_


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("norm",))
        reference = torch.tensor([300.0, 400.0], dtype=torch.float64)
        expected_norm = torch.linalg.vector_norm(reference)
        for axis in ("pp", "tp", "ep"):
            parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
            parameter.grad = reference[rank : rank + 1].to(torch.float16)
            kwargs = {f"{axis}_mesh": mesh}
            if axis == "tp":
                kwargs["tp_sharded_parameters"] = [parameter]
            elif axis == "ep":
                kwargs["expert_parameters"] = [parameter]
            norm = clip_grad_norm_(
                [parameter], max_norm=1.0, error_if_nonfinite=True, **kwargs
            )
            torch.testing.assert_close(norm.double(), expected_norm, rtol=0, atol=0)
            torch.testing.assert_close(
                parameter.grad.double(),
                reference[rank : rank + 1] / expected_norm,
                rtol=1e-3,
                atol=0,
            )
            print(f"rank={rank} axis={axis} norm={norm.item()} clipped=True")

        # A stage with no local gradient must still join the PP reduction.
        parameter = torch.nn.Parameter(torch.zeros(1))
        if rank == 1:
            parameter.grad = torch.tensor([400.0])
        norm = clip_grad_norm_(
            [parameter], max_norm=1.0, pp_mesh=mesh, error_if_nonfinite=True
        )
        torch.testing.assert_close(norm, torch.tensor(400.0), rtol=0, atol=0)
        if rank == 0:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, torch.ones(1))
        print(f"rank={rank} empty-stage PP norm={norm.item()} passed")

        for axis in ("pp", "tp", "ep"):
            parameters = [] if rank == 0 else [parameter]
            if rank == 1:
                parameter.grad = torch.tensor([400.0])
            kwargs = {f"{axis}_mesh": mesh}
            if axis == "tp":
                kwargs["tp_sharded_parameters"] = parameters
            elif axis == "ep":
                kwargs["expert_parameters"] = parameters
            norm = clip_grad_norm_(parameters, max_norm=1.0, **kwargs)
            torch.testing.assert_close(norm, torch.tensor(400.0), rtol=0, atol=0)
            if rank == 1:
                torch.testing.assert_close(parameter.grad, torch.ones(1))
            print(f"rank={rank} axis={axis} parameterless-peer norm=400 passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
