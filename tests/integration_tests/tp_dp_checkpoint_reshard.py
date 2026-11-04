"""CPU or accelerator DCP restore between explicit TP/DP or EP/DP layouts.

Run with ``PYTHONPATH=. python tests/integration_tests/tp_dp_checkpoint_reshard.py``.
The default launches separate TP2/DP2 save and TP2/DP1 load jobs.
Use --save-tp/--save-dp/--load-tp/--load-dp to change either degree.
For EP use --shard-axis ep with --save-degree/--load-degree; TP flags remain aliases.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.components.checkpointer.core.model_state import ModelCheckpointState
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded
from tests.integration_tests.harness import device as harness_device
from tests.integration_tests.harness import init_distributed


def build_optimizer(
    model: nn.Module, *, cold: bool = False, shard_axis: str = "tp"
) -> OptimizersContainer:
    return OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.9 if cold else 0.01},
                )
            ],
        ),
        model_parts=[model],
        sharded_tensor_parallel=shard_axis == "tp",
        sharded_expert_parallel=shard_axis == "ep",
    )


def reference_after_two_steps() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = nn.Parameter(
        torch.arange(8 * 4, dtype=torch.float32, device=harness_device()).reshape(8, 4)
    )
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, weight_decay=0.01, foreach=False, fused=False
    )
    coordinates = parameter.detach().clone()
    parameter.grad = (coordinates + 1) / 100
    optimizer.step()
    optimizer.zero_grad()
    parameter.grad = coordinates.cos() + 2
    optimizer.step()
    state = {
        name: value.detach().clone()
        for name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }
    return parameter.detach().clone(), state


def worker(
    mode: str, checkpoint: str, *, degree: int, dp_size: int, shard_axis: str
) -> None:
    device = init_distributed(timeout=timedelta(seconds=90))
    try:
        saving = mode == "save"
        assert dist.get_world_size() == dp_size * degree
        mesh = init_device_mesh(
            device.type, (dp_size, degree), mesh_dim_names=("dp", shard_axis)
        )
        partition_rank = mesh[shard_axis].get_local_rank()
        dp_rank = mesh["dp"].get_local_rank()
        full = torch.arange(8 * 4, dtype=torch.float32, device=device).reshape(8, 4)
        partition_dim = 1 if shard_axis == "tp" else 0
        dp_dim = 1 - partition_dim
        partition_local = full.chunk(degree, dim=partition_dim)[
            partition_rank
        ].contiguous()
        initial = torch.zeros_like(partition_local) if not saving else partition_local

        model = nn.Module()
        model.weight = nn.Parameter(
            distribute_tensor(initial, mesh["dp"], [Shard(dp_dim)])
        )
        if shard_axis == "tp":
            model._tp_checkpoint_layouts = {
                "weight": (1, False, mesh[shard_axis].get_group())
            }
        else:
            mark_experts_ep_sharded(model, mesh[shard_axis].get_group())
        wrapper = ModelCheckpointState(
            model,
            sharded_tensor_parallel=shard_axis == "tp",
            sharded_expert_parallel=shard_axis == "ep",
        )
        optimizer = build_optimizer(model, cold=not saving, shard_axis=shard_axis)

        if saving:
            gradient = (
                ((full + 1) / 100)
                .chunk(degree, dim=partition_dim)[partition_rank]
                .contiguous()
            )
            model.weight.grad = distribute_tensor(gradient, mesh["dp"], [Shard(dp_dim)])
            optimizer.step()
            optimizer.zero_grad()
            dcp.save(
                {"model": wrapper, "optimizer": optimizer},
                checkpoint_id=checkpoint,
            )
            dist.barrier()
            if dist.get_rank() == 0:
                print(
                    f"PASS: saved {shard_axis.upper()}{degree} x DP{dp_size} "
                    "model and optimizer",
                    flush=True,
                )
            return

        assert all(not inner.state for inner in optimizer.optimizers)
        dcp.load(
            {"model": wrapper, "optimizer": optimizer},
            checkpoint_id=checkpoint,
        )
        gradient = (
            (full.cos() + 2)
            .chunk(degree, dim=partition_dim)[partition_rank]
            .contiguous()
        )
        model.weight.grad = distribute_tensor(gradient, mesh["dp"], [Shard(dp_dim)])
        optimizer.step()
        optimizer.zero_grad()

        expected_model, expected_optimizer = reference_after_two_steps()
        expected_local = expected_model.chunk(degree, dim=partition_dim)[
            partition_rank
        ].chunk(dp_size, dim=dp_dim)[dp_rank]
        torch.testing.assert_close(
            model.weight.to_local(), expected_local, atol=0, rtol=0
        )

        saved_optimizer = optimizer.state_dict()
        for name, expected in expected_optimizer.items():
            key = f"state.weight.{name}"
            actual = saved_optimizer[key]
            if hasattr(actual, "chunks"):
                for offset, chunk in actual.chunks:
                    slices = tuple(
                        slice(start, start + size)
                        for start, size in zip(offset, chunk.shape, strict=True)
                    )
                    torch.testing.assert_close(chunk, expected[slices], atol=0, rtol=0)
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"PASS: {shard_axis.upper()}{degree} x DP{dp_size} "
                "cold restore and next step",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=("save", "load"))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--save-tp", "--save-degree", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument("--save-dp", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--load-tp", "--load-degree", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument("--shard-axis", choices=("tp", "ep"), default="tp")
    parser.add_argument("--load-dp", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.worker_mode:
        if not args.checkpoint:
            parser.error("--checkpoint is required in worker mode")
        saving = args.worker_mode == "save"
        worker(
            args.worker_mode,
            args.checkpoint,
            degree=args.save_tp if saving else args.load_tp,
            dp_size=args.save_dp if saving else args.load_dp,
            shard_axis=args.shard_axis,
        )
        return

    script = str(Path(__file__).resolve())
    repo_root = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="tp-dp-reshard-") as checkpoint:
        for mode, degree, dp_size in (
            ("save", args.save_tp, args.save_dp),
            ("load", args.load_tp, args.load_dp),
        ):
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={degree * dp_size}",
                script,
                "--shard-axis",
                args.shard_axis,
                "--worker-mode",
                mode,
                "--checkpoint",
                checkpoint,
                "--save-tp",
                str(args.save_tp),
                "--save-dp",
                str(args.save_dp),
                "--load-tp",
                str(args.load_tp),
                "--load-dp",
                str(args.load_dp),
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = (
                repo_root + os.pathsep + environment.get("PYTHONPATH", "")
            )
            subprocess.run(command, cwd=repo_root, env=environment, check=True)


if __name__ == "__main__":
    main()
