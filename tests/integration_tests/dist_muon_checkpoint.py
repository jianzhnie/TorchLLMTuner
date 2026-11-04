"""DistMuon FQN and native DCP restores match uninterrupted updates.

Defaults to CPU/Gloo; DIST_MUON_DEVICE=npu selects real NPU/HCCL execution.
DIST_MUON_SHAPE=matrix_batch additionally covers batched expert matrices.
Use torchrun for multi-rank sharded storage and real redistribution.
DIST_MUON_PHASE=produce/consume runs save and restore in separate torchrun jobs;
both jobs must use the same output directory, shape and world size.
DIST_MUON_GLOBAL_REFERENCE=1 permits different world sizes using full CPU references.
"""

import copy
import os
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from llmtuner.accelerator.collectives import synchronize_local_errors
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
    local = model.weight.to_local()
    _, row_offset = Shard.local_shard_size_and_offset(
        model.weight.shape[0], dist.get_world_size(), dist.get_rank()
    )
    indices = torch.arange(local.numel(), device=local.device).reshape(local.shape)
    gradient = (indices + row_offset * model.weight.stride()[0]).float().sin() + value
    model.weight.grad = DTensor.from_local(
        gradient.to(local.dtype),
        model.weight.device_mesh,
        model.weight.placements,
        shape=model.weight.shape,
        stride=model.weight.stride(),
    )
    optimizer.step()
    optimizer.zero_grad()


def assert_optimizer_state_equal(actual, expected):
    assert actual.keys() == expected.keys(), "optimizer state keys differ"
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, DTensor):
            assert isinstance(reference, DTensor), key
            assert value.shape == reference.shape, key
            assert value.placements == reference.placements, key
            torch.testing.assert_close(
                value.to_local(), reference.to_local(), rtol=0, atol=0, msg=key
            )
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, rtol=0, atol=0, msg=key)
        else:
            assert value == reference, key


def reference_tensor(value, *, global_state=False):
    if isinstance(value, DTensor):
        value = value.full_tensor() if global_state else value.to_local()
    return value.detach().cpu().clone()


def reference_optimizer_state(optimizer, *, global_state=False):
    """Persist test references without serializing a live process-group mesh."""
    state = {}
    for key, value in optimizer.state_dict().items():
        state[key] = (
            reference_tensor(value, global_state=global_state)
            if isinstance(value, torch.Tensor)
            else copy.deepcopy(value)
        )
    return state


def consume_checkpoint(root, mesh, *, global_state):
    with synchronize_local_errors("DistMuon reference validation"):
        path = (
            root / "global_reference.pt"
            if global_state
            else root / f"rank{dist.get_rank()}_reference.pt"
        )
        expected = torch.load(path, weights_only=True)
        assert expected.get("global_reference", False) == global_state
        if not global_state:
            assert expected["world_size"] == dist.get_world_size()
        assert expected["shape"] == os.environ.get("DIST_MUON_SHAPE", "matrix")
    model = _model(mesh)
    optimizer = _optimizer(model)
    dcp.load(
        {"model": model, "optimizer": optimizer}, checkpoint_id=root / "checkpoint"
    )
    # Full-tensor reference gathering contains collectives and must stay outside
    # the local comparison/error-coordination boundary.
    weight = reference_tensor(model.weight, global_state=global_state)
    state = reference_optimizer_state(optimizer, global_state=global_state)
    with synchronize_local_errors("DistMuon independent restore comparison"):
        torch.testing.assert_close(
            weight, expected["checkpoint_weight"], rtol=0, atol=0
        )
        assert_optimizer_state_equal(state, expected["checkpoint_state"])
    _step(model, optimizer, 0.3)
    weight = reference_tensor(model.weight, global_state=global_state)
    state = reference_optimizer_state(optimizer, global_state=global_state)
    with synchronize_local_errors("DistMuon independent next-step comparison"):
        torch.testing.assert_close(weight, expected["next_weight"], rtol=0, atol=0)
        assert_optimizer_state_equal(state, expected["next_state"])
    print(f"PASS DistMuon independent-process restore rank={dist.get_rank()}")


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    device = os.environ.get("DIST_MUON_DEVICE", "cpu")
    phase = os.environ.get("DIST_MUON_PHASE", "combined")
    if phase not in ("combined", "produce", "consume"):
        raise ValueError("DIST_MUON_PHASE must be combined, produce or consume")
    global_reference = os.environ.get("DIST_MUON_GLOBAL_REFERENCE", "0")
    if global_reference not in ("0", "1"):
        raise ValueError("DIST_MUON_GLOBAL_REFERENCE must be 0 or 1")
    global_reference = global_reference == "1"
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
        if phase == "consume":
            consume_checkpoint(root, mesh, global_state=global_reference)
            return
        reference_model = _model(mesh)
        resumed_model = _model(mesh)
        with torch.no_grad():
            resumed_model.weight.to_local().copy_(reference_model.weight.to_local())
        reference = _optimizer(reference_model)
        _step(reference_model, reference, 0.1)
        _step(reference_model, reference, 0.2)
        state = copy.deepcopy(reference.state_dict())
        checkpoint_weight = reference_model.weight.to_local().detach().clone()
        if phase == "produce":
            reference_checkpoint_weight = reference_tensor(
                reference_model.weight, global_state=global_reference
            )
            checkpoint_state = reference_optimizer_state(
                reference, global_state=global_reference
            )

        resumed = _optimizer(resumed_model)
        with torch.no_grad():
            resumed_model.weight.to_local().copy_(checkpoint_weight)
        failure_stage = os.environ.get("DIST_MUON_INJECT_LOAD_PLAN_FAILURE", "")
        if failure_stage == "local":
            inner = resumed.optimizers[0]
            original_initialize_plan = inner._initialize_plan

            def fail_local_plan(_self, layouts):
                if dist.get_rank() == 0:
                    raise ValueError("injected DistMuon load plan failure")
                return original_initialize_plan(layouts)

            inner._initialize_plan = MethodType(fail_local_plan, inner)
            error = None
            try:
                resumed.load_state_dict(state)
            except (ValueError, RuntimeError) as caught:
                error = caught
            finally:
                del inner._initialize_plan
            outcomes = [None] * dist.get_world_size()
            dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
            assert outcomes[0][0] == "ValueError", outcomes
            assert "injected DistMuon load plan failure" in outcomes[0][1]
            for outcome in outcomes[1:]:
                assert outcome[0] == "RuntimeError", outcomes
                assert (
                    "Optimizer checkpoint application failed on another rank"
                    in outcome[1]
                )
            print(
                "PASS DistMuon local load-plan failure coordinated "
                "before mesh validation"
            )
        elif failure_stage == "mesh":
            inner = resumed.optimizers[0]
            original_validate_plan = inner._validate_plan_across_ranks

            def fail_mesh_plan():
                original_validate_plan()
                if dist.get_rank() == 0:
                    raise ValueError("injected DistMuon mesh plan failure")

            inner._validate_plan_across_ranks = fail_mesh_plan
            error = None
            try:
                resumed.load_state_dict(state)
            except (ValueError, RuntimeError) as caught:
                error = caught
            finally:
                del inner._validate_plan_across_ranks
            outcomes = [None] * dist.get_world_size()
            dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
            assert outcomes[0][0] == "ValueError", outcomes
            assert "injected DistMuon mesh plan failure" in outcomes[0][1]
            assert outcomes[1][0] == "RuntimeError", outcomes
            assert (
                "Optimizer mesh plan validation failed on another rank"
                in (outcomes[1][1])
            )
            print(
                "PASS DistMuon mesh plan failure coordinated before "
                "checkpoint continuation"
            )
        resumed.load_state_dict(state)
        assert_optimizer_state_equal(resumed.state_dict(), state)

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
        assert_optimizer_state_equal(disk_optimizer.state_dict(), state)

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
        for restored_optimizer in (resumed, disk_optimizer):
            assert_optimizer_state_equal(
                restored_optimizer.state_dict(), reference.state_dict()
            )
        if phase == "produce":
            expected = {
                "world_size": dist.get_world_size(),
                "global_reference": global_reference,
                "shape": os.environ.get("DIST_MUON_SHAPE", "matrix"),
                "checkpoint_weight": reference_checkpoint_weight,
                "checkpoint_state": checkpoint_state,
                "next_weight": reference_tensor(
                    reference_model.weight, global_state=global_reference
                ),
                "next_state": reference_optimizer_state(
                    reference, global_state=global_reference
                ),
            }
            if not global_reference or dist.get_rank() == 0:
                path = (
                    root / "global_reference.pt"
                    if global_reference
                    else root / f"rank{dist.get_rank()}_reference.pt"
                )
                torch.save(expected, path)
        print(f"PASS DistMuon updates, FQN and disk DCP restore on {device}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
