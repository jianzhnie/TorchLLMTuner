"""DistMuon FQN and native DCP restores match uninterrupted updates.

Defaults to CPU/Gloo; DIST_MUON_DEVICE=npu selects real NPU/HCCL execution.
DIST_MUON_SHAPE=matrix_batch additionally covers batched expert matrices.
Use torchrun for multi-rank sharded storage and real redistribution.
"""

import copy
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from llmtuner.components.optimizer.flex_shard import BucketConfig, ComputeLayout, Owned
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import OptimizerConfig, ParamGroupConfig
from tests.integration_tests.harness import init_distributed


def _model(mesh):
    shape = (2, 4, 4) if os.environ.get("DIST_MUON_SHAPE") == "matrix_batch" else (4, 4)
    if len(shape) == 2:
        stride = (shape[-1], 1)
    else:
        stride = (shape[-2] * shape[-1], shape[-1], 1)
    rank = dist.get_rank()
    local_size, _ = Shard.local_shard_size_and_offset(
        shape[0], dist.get_world_size(), rank
    )
    local_shape = (local_size, *shape[1:])
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        DTensor.from_local(
            torch.randn(*local_shape, device=mesh.device_type),
            mesh,
            (Shard(0),),
            shape=torch.Size(shape),
            stride=stride,
        )
    )
    return model


def _optimizer(model):
    return OptimizersContainer(
        OptimizerConfig(
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="DistMuon",
                    optimizer_kwargs={"lr": 1e-3},
                )
            ],
            compute_sharding_by_fqn={
                "weight": ComputeLayout(
                    {"dp_shard": Owned()}, compute_dtype=torch.bfloat16
                )
            },
            bucket_configs=[BucketConfig(patterns=("weight",))],
        ),
        model_parts=[model],
    )


def _step(model, optimizer, value):
    model.weight.grad = DTensor.from_local(
        torch.full_like(model.weight.to_local(), value),
        model.weight.device_mesh,
        model.weight.placements,
        shape=model.weight.shape,
        stride=model.weight.stride(),
    )
    optimizer.step()
    optimizer.zero_grad()


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("DIST_MUON_DEVICE", "cpu")
    if "RANK" in os.environ:
        # Keep the explicit device selector compatible with standalone runs.
        os.environ["LLMTUNER_TEST_DEVICE"] = device
        init_distributed()
    else:
        dist.init_process_group(
            "hccl" if device == "npu" else "gloo",
            init_method=f"file://{root.resolve()}/process_group_store",
            rank=0,
            world_size=1,
        )
    try:
        mesh = DeviceMesh(
            device, list(range(dist.get_world_size())), mesh_dim_names=("dp_shard",)
        )
        torch.manual_seed(7 + dist.get_rank())
        reference_model = _model(mesh)
        resumed_model = _model(mesh)
        with torch.no_grad():
            resumed_model.weight.to_local().copy_(reference_model.weight.to_local())
        reference = _optimizer(reference_model)
        _step(reference_model, reference, 0.1)
        _step(reference_model, reference, 0.2)
        state = copy.deepcopy(reference.state_dict())
        checkpoint_weight = reference_model.weight.to_local().detach().clone()

        resumed = _optimizer(resumed_model)
        with torch.no_grad():
            resumed_model.weight.to_local().copy_(checkpoint_weight)
        resumed.load_state_dict(state)

        checkpoint_path = root / "checkpoint"
        dcp.save(
            {"model": reference_model, "optimizer": reference},
            checkpoint_id=checkpoint_path,
        )
        assert (checkpoint_path / ".metadata").is_file()
        disk_model = _model(mesh)
        disk_optimizer = _optimizer(disk_model)
        dcp.load(
            {"model": disk_model, "optimizer": disk_optimizer},
            checkpoint_id=checkpoint_path,
        )
        torch.testing.assert_close(
            disk_model.weight.to_local(), checkpoint_weight, rtol=0, atol=0
        )
        for key, value in disk_optimizer.state_dict().items():
            expected = state[key]
            if isinstance(value, DTensor):
                torch.testing.assert_close(
                    value.to_local(), expected.to_local(), rtol=0, atol=0
                )
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                assert value == expected, key

        _step(reference_model, reference, 0.3)
        _step(resumed_model, resumed, 0.3)
        _step(disk_model, disk_optimizer, 0.3)
        changed = torch.tensor(
            int(not torch.equal(checkpoint_weight, reference_model.weight.to_local())),
            device=reference_model.weight.device,
        )
        dist.all_reduce(changed)
        assert changed.item() > 0
        for restored_model in (resumed_model, disk_model):
            torch.testing.assert_close(
                restored_model.weight.to_local().cpu(),
                reference_model.weight.to_local().cpu(),
                rtol=0,
                atol=0,
            )
        print(f"PASS DistMuon updates, FQN and disk DCP restore on {device}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
