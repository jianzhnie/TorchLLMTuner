"""Diagnose the active baseline TP expert kernel with distinct token streams.

A row-parallel sum is valid only when corresponding rows represent the same
routed tokens on every rank. This probe does not modify baseline code.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    from torchtitan_npu.converters.kernels.gmm import (
        _expert_activation,
        _run_experts_grouped_mm,
        npu_grouped_experts_forward,
    )

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = False
        torch.manual_seed(37)
        gate = torch.randn(4, 64, 32, device="npu", dtype=torch.bfloat16) * 0.1
        up = torch.randn_like(gate) * 0.1
        down = torch.randn(4, 32, 64, device="npu", dtype=torch.bfloat16) * 0.1
        local_packed = torch.cat((gate.chunk(2, 1)[rank], up.chunk(2, 1)[rank]), 1)
        local_down = down.chunk(2, 2)[rank].contiguous()
        block = SimpleNamespace(
            w13=DTensor.from_local(local_packed, mesh, [Shard(1)], run_check=False),
            w2=DTensor.from_local(local_down, mesh, [Shard(2)], run_check=False),
            _expert_activation_fn=_expert_activation,
        )
        counts = torch.full((4,), 8, device="npu", dtype=torch.int64)
        shared = torch.randn(32, 32, device="npu", dtype=torch.bfloat16)
        report = []
        for distinct in (False, True):
            inputs = shared + rank if distinct else shared
            with torch.no_grad():
                actual = npu_grouped_experts_forward(block, inputs, counts)
                # Independent explicit oracle: every feature shard must see
                # this rank's same tokens before its partials can be summed.
                partials = [
                    _run_experts_grouped_mm(
                        torch.cat((gate.chunk(2, 1)[i], up.chunk(2, 1)[i]), 1),
                        down.chunk(2, 2)[i].contiguous(),
                        None,
                        inputs,
                        counts,
                    )
                    for i in range(2)
                ]
                expected = partials[0] + partials[1]
            delta = (actual.float() - expected.float()).abs()
            exact = torch.equal(actual, expected)
            if not distinct:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            report.append(
                {
                    "distinct_tokens": distinct,
                    "exact": exact,
                    "max_ae": delta.max().item(),
                    "mean_ae": delta.mean().item(),
                }
            )
        path = args.output.with_name(args.output.stem + f"-rank{rank}.json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"rank={rank} {report}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
