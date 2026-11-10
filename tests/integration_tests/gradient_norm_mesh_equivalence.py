"""Real Gloo/HCCL regression for distinct gradient DeviceMesh objects.

Run from the repository root with torchrun --standalone --nproc-per-node=2
--module tests.integration_tests.gradient_norm_mesh_equivalence.
"""

import argparse
import hashlib
import inspect
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.accelerator.collectives import _parameter_precision_norm
from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "npu":
        require_env("torch_npu")
        assert torch.npu.is_available(), "NPU explicitly requested but unavailable"
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
    dist.init_process_group(
        "hccl" if device.type == "npu" else "gloo",
        timeout=timedelta(seconds=120),
    )
    try:
        assert dist.get_world_size() == 2
        first = init_device_mesh(device.type, (2,), mesh_dim_names=("shard",))
        second = init_device_mesh(device.type, (2,), mesh_dim_names=("shard",))
        assert first is not second and first == second
        generator = torch.Generator().manual_seed(719)
        differences = []
        for trial in range(16):
            values = [
                torch.randn(18, generator=generator).bfloat16() * scale
                for scale in (0.125, 3.0, 17.0)
            ]
            mixed = [
                distribute_tensor(value.to(device), mesh, [Shard(0)])
                for value, mesh in zip(values, (first, second, first), strict=True)
            ]
            expected = torch.nn.utils.get_total_norm(
                mixed, 2.0, foreach=False
            ).full_tensor()
            actual = _parameter_precision_norm(mixed, 2.0)
            assert actual.device.type == expected.device.type == device.type
            assert torch.isfinite(actual) and torch.isfinite(expected)
            if not torch.equal(actual, expected):
                differences.append((trial, actual.item(), expected.item()))
        assert not differences, differences
        # Distinct ownership groups follow upstream scalar power aggregation,
        # which has a different BF16 rounding contract from vector_norm.
        other = init_device_mesh(device.type, (2,), mesh_dim_names=("other",))
        assert other != first
        differences = []
        for trial in range(16):
            gradients = [
                distribute_tensor(
                    (torch.randn(18, generator=generator).bfloat16() * scale).to(
                        device
                    ),
                    mesh,
                    [Shard(0)],
                )
                for mesh, scale in ((first, 3.0), (other, 17.0))
            ]
            norms = [
                torch.nn.utils.get_total_norm([grad], 2.0, foreach=False).full_tensor()
                for grad in gradients
            ]
            expected = sum(norm**2.0 for norm in norms) ** 0.5
            actual = _parameter_precision_norm(gradients, 2.0)
            assert actual.device.type == expected.device.type == device.type
            assert torch.isfinite(actual) and torch.isfinite(expected)
            if not torch.equal(actual, expected):
                differences.append((trial, actual.item(), expected.item()))
        assert not differences, differences
        report = {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "device": str(device),
            "backend": dist.get_backend(),
            "torch_version": torch.__version__,
            "equivalent_mesh_trials": 16,
            "distinct_mesh_trials": 16,
            "zero_tolerance": True,
            "passed": True,
            "seed": 719,
            "input_generation": (
                "CPU BF16 generator, unchanged scales, then device transfer"
            ),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "collectives_sha256": hashlib.sha256(
                Path(inspect.getfile(_parameter_precision_norm)).read_bytes()
            ).hexdigest(),
            "scope": "Native BF16 mesh norm contract, not target-model P4 acceptance",
        }
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"receipt-rank{dist.get_rank()}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        print(f"MESH_NORM_EQUIVALENCE_PASS {json.dumps(report)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
