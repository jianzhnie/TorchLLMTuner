"""CPU/Gloo DCP restore from TP2 x DP2 into TP2 x DP1.

Run with ``PYTHONPATH=. python tests/integration_tests/tp_dp_checkpoint_reshard.py``.
The driver launches separate four-rank save and two-rank load jobs so each job
uses a real process group with a different data-parallel degree.
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

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig


def build_optimizer(model: nn.Module, *, cold: bool = False) -> OptimizersContainer:
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
        sharded_tensor_parallel=True,
    )


def reference_after_two_steps() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameter = nn.Parameter(torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.01)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    state = {
        name: value.detach().clone()
        for name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }
    return parameter.detach().clone(), state


def worker(mode: str, checkpoint: str) -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        saving = mode == "save"
        dp_size = 2 if saving else 1
        assert dist.get_world_size() == dp_size * 2
        mesh = init_device_mesh("cpu", (dp_size, 2), mesh_dim_names=("dp", "tp"))
        tp_rank = mesh["tp"].get_local_rank()
        full = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        tp_local = full.chunk(2, dim=1)[tp_rank].contiguous()
        initial = torch.zeros_like(tp_local) if not saving else tp_local

        model = nn.Module()
        model.weight = nn.Parameter(distribute_tensor(initial, mesh["dp"], [Shard(0)]))
        model._tp_checkpoint_layouts = {"weight": (1, False, mesh["tp"].get_group())}
        wrapper = ModelCheckpointState(model, sharded_tensor_parallel=True)
        optimizer = build_optimizer(model, cold=not saving)

        if saving:
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad()
            dcp.save(
                {"model": wrapper, "optimizer": optimizer},
                checkpoint_id=checkpoint,
            )
            dist.barrier()
            if dist.get_rank() == 0:
                print("PASS: saved TP2 x DP2 model and optimizer", flush=True)
            return

        assert all(not inner.state for inner in optimizer.optimizers)
        dcp.load(
            {"model": wrapper, "optimizer": optimizer},
            checkpoint_id=checkpoint,
        )
        model.weight.grad = torch.full_like(model.weight, 0.25)
        optimizer.step()
        optimizer.zero_grad()

        expected_model, expected_optimizer = reference_after_two_steps()
        expected_local = expected_model.chunk(2, dim=1)[tp_rank]
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
            print("PASS: TP2 x DP2 -> TP2 x DP1 cold restore and next step", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=("save", "load"))
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.worker_mode:
        if not args.checkpoint:
            parser.error("--checkpoint is required in worker mode")
        worker(args.worker_mode, args.checkpoint)
        return

    script = str(Path(__file__).resolve())
    repo_root = str(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory(prefix="tp-dp-reshard-") as checkpoint:
        for mode, world_size in (("save", 4), ("load", 2)):
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world_size}",
                script,
                "--worker-mode",
                mode,
                "--checkpoint",
                checkpoint,
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = (
                repo_root + os.pathsep + environment.get("PYTHONPATH", "")
            )
            subprocess.run(command, cwd=repo_root, env=environment, check=True)


if __name__ == "__main__":
    main()
