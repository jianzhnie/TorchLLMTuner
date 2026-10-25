"""Verify TP-replaced FSDP exclusions against independent dense SGD updates.

PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/fsdp_pattern_references.py native

CPU/Gloo, TP2 x FSDP2. The child unit must not manage the excluded column
weight. This test explicitly reduces its local gradients when it remains a
plain Tensor, as required of callers managing exclusions.
Optional second argument: compile.
"""

import copy
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology


def main():
    implementation = sys.argv[1] if len(sys.argv) > 1 else "native"
    compiled = len(sys.argv) > 2 and sys.argv[2] == "compile"
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 4
        torch.manual_seed(309)
        model = nn.Sequential(
            nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False)
        ).double()
        reference = copy.deepcopy(model)
        model._tp_config = {"0": "colwise", "1": "rowwise"}
        original = model[0].weight
        cfg = ParallelConfig(
            tensor_parallel_size=2,
            fsdp_config=FSDPConfig(
                fsdp_implementation=implementation,
                fsdp_reduce_dtype="float64",
                fsdp_apply_modules={"0": {"ignored_params": {original}}, "1": {}},
            ),
        )
        dims = ParallelTopology.from_config(cfg, world_size=4)
        dims.build_device_mesh()
        tp_rank = dims.get_mesh("tp").get_local_rank()
        dp_rank = dims.get_mesh("dp_shard").get_local_rank()
        model = parallelize_hf_transformers(
            model,
            cfg=cfg,
            mesh=dims.spmd_dense_mesh(),
            parallel_dims=dims,
            compile=compiled,
            compile_config=CompileConfig(backend="aot_eager"),
        )
        base = model._orig_mod if compiled else model
        assert base[0]._get_fsdp_state()._fsdp_param_group is None
        assert cfg.fsdp_config.fsdp_apply_modules["0"]["ignored_params"] == {original}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dense_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for step in range(3):
            inputs = [
                (torch.arange(16, dtype=torch.float64).reshape(1, 4, 4) / 16)
                + rank / 5
                + step / 10
                for rank in range(2)
            ]
            optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            expected = [reference(value) for value in inputs]
            output = model(inputs[dp_rank].chunk(2, dim=1)[tp_rank].contiguous())
            torch.testing.assert_close(
                output,
                expected[dp_rank].chunk(2, dim=1)[tp_rank],
                atol=1e-12,
                rtol=1e-12,
            )
            output.square().sum().backward()
            local_grad = torch.autograd.grad(
                expected[dp_rank].square().sum(),
                reference[0].weight,
                retain_graph=True,
            )[0].chunk(2, dim=0)[tp_rank]
            excluded_weight = base[0].weight
            if not isinstance(excluded_weight, DTensor):
                torch.testing.assert_close(
                    excluded_weight.grad, local_grad, atol=1e-12, rtol=1e-12
                )
                dist.all_reduce(
                    excluded_weight.grad, group=dims.get_mesh("dp").get_group()
                )
            sum(value.square().sum() for value in expected).backward()
            for index, (actual, dense) in enumerate(
                zip(model.parameters(), reference.parameters(), strict=True)
            ):
                expected_grad = dense.grad.chunk(2, dim=index)[tp_rank]
                torch.testing.assert_close(
                    (
                        actual.grad.full_tensor()
                        if isinstance(actual.grad, DTensor)
                        else actual.grad
                    ),
                    expected_grad,
                    atol=1e-12,
                    rtol=1e-12,
                )
            optimizer.step()
            dense_optimizer.step()
            for index, (actual, dense) in enumerate(
                zip(model.parameters(), reference.parameters(), strict=True)
            ):
                torch.testing.assert_close(
                    (actual.full_tensor() if isinstance(actual, DTensor) else actual),
                    dense.chunk(2, dim=index)[tp_rank],
                    atol=1e-12,
                    rtol=1e-12,
                )
        print(f"rank={dist.get_rank()} {implementation} compile={compiled}: passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
