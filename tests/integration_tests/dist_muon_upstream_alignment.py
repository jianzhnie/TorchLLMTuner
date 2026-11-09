"""Compare real upstream FlexShard updates, momentum and buffer lifetimes.

Run with torchrun and --upstream pointing to a TorchTitan checkout. The
reference is loaded as an isolated package, without TorchTitan's training stack.
Set LLMTUNER_TEST_DEVICE=cpu for Gloo; otherwise use the available accelerator.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from llmtuner.components.optimizer.flex_shard._optimizer_reshard_schedule import (
    _redistribution_plan_key,
)
from llmtuner.components.optimizer.flex_shard.dist_muon import DistMuon
from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from tests.integration_tests.harness import init_distributed


def assert_released(optimizer: DistMuon) -> None:
    """Check backing storage, rather than tensor metadata or allocator caches."""
    runtime = optimizer._redistribution_runtime
    slots = [runtime._local_slot]
    if runtime._context is not None:
        slots.extend(slot.buffers for slot in runtime._context.slots)
    for slot in slots:
        for buffers in slot.buffers.values():
            for name in (
                "storage_exchange",
                "compute_exchange",
                "compute_scratch",
                "storage_scratch",
            ):
                tensor = getattr(buffers, name)
                if tensor is not None and tensor.device.type != "cpu":
                    assert tensor.untyped_storage().nbytes() == 0, name


def plan_signature(optimizer):
    """Compare logical routes and bucket order without process-group identity."""
    return tuple(
        (
            tuple(item.fqn for item in bucket.items)
            if hasattr(bucket, "items")
            else (
                tuple(item.fqn for item in bucket.unredistributed_items),
                tuple(item.fqn for item in bucket.redistributed_items),
            ),
            tuple(
                _redistribution_plan_key(plan)
                for plan in getattr(bucket, "redistribution_plans", ())
            ),
        )
        for bucket in optimizer._bucket_plans
    )


def mixed_bucket_case(device, mesh, reference_muon, reference_api, *, pipelined=False):
    """Exercise fixed work, unequal chunks, empty shards and cumulative loads."""
    torch.manual_seed(57)
    specifications = (
        ("fixed", (8, 3, 4), "native"),
        ("owned_a", (8, 4), "owned"),
        ("owned_b", (3, 7), "owned"),
        ("block", (18, 3), "block"),
        ("tail_empty", (1, 3, 4), "owned"),
        ("tail_batch", (8, 3, 4), "native"),
    )
    params, references, local_layouts, reference_layouts = [], [], {}, {}
    for name, shape, kind in specifications:
        source = torch.randn(shape, device=device)
        params.append(
            torch.nn.Parameter(distribute_tensor(source.clone(), mesh, [Shard(0)]))
        )
        references.append(
            torch.nn.Parameter(distribute_tensor(source.clone(), mesh, [Shard(0)]))
        )
        local_placement = (
            Owned()
            if kind == "owned"
            else BlockShard(0, (2, 4))
            if kind == "block"
            else Shard(0)
        )
        ref_placement = (
            reference_api.Owned()
            if kind == "owned"
            else reference_api.BlockShard(0, (2, 4))
            if kind == "block"
            else Shard(0)
        )
        local_layouts[name] = ComputeLayout(
            {"dp": local_placement}, compute_dtype=torch.bfloat16
        )
        reference_layouts[name] = reference_api.ComputeLayout(
            {"dp": ref_placement}, compute_dtype=torch.bfloat16
        )
    patterns = (
        (
            ("owned_a",),
            ("owned_b",),
            ("block",),
            ("fixed",),
            ("tail_empty",),
            ("tail_batch",),
        )
        if pipelined
        else (("fixed", "owned_*", "block"), ("tail_*",))
    )
    # A separately created mesh tests explicitly selecting a different group
    # with the same axis names and rank order.
    bucket_mesh = DeviceMesh(device.type, mesh.mesh.clone(), mesh_dim_names=("dp",))
    names = [name for name, _, _ in specifications]
    options = dict(lr=0.01, ns_steps=3)
    local = DistMuon(
        [dict(params=params, param_names=names)],
        compute_sharding_by_fqn=local_layouts,
        bucket_configs=[
            BucketConfig(pattern, name=str(i), mesh=bucket_mesh)
            for i, pattern in enumerate(patterns)
        ],
        **options,
    )
    reference = reference_muon.DistMuon(
        [dict(params=references, param_names=names)],
        compute_sharding_by_fqn=reference_layouts,
        bucket_configs=[
            reference_api.BucketConfig(pattern, name=str(i), mesh=bucket_mesh)
            for i, pattern in enumerate(patterns)
        ],
        **options,
    )
    assert plan_signature(local) == plan_signature(reference)
    handle = torch.get_device_module(device)
    streams = (
        [handle.Stream(), handle.Stream()] if pipelined and device.type != "cpu" else []
    )
    for step in range(3):
        for parameter, other in zip(params, references, strict=True):
            gradient = distribute_tensor(
                torch.randn(parameter.shape, device=device), mesh, [Shard(0)]
            )
            parameter.grad = gradient.clone()
            other.grad = gradient.clone()
        # Exercise missing-gradient behavior on a later step, on every rank.
        if step == 1:
            # Loading may change NS work costs and therefore chunk ownership.
            # Repeated load must rebuild identical plans and release capacity.
            for optimizer in (local, reference):
                state = copy.deepcopy(optimizer.state_dict())
                state["param_groups"][0].update(
                    ns_steps=2,
                    ns_coefficients=((3.4445, -4.7750, 2.0315), (3.0, -4.0, 2.0)),
                )
                optimizer.load_state_dict(state)
                optimizer.load_state_dict(state)
            assert plan_signature(local) == plan_signature(reference)
            assert_released(local)
            local_grad, reference_grad = params[1].grad, references[1].grad
            params[1].grad = references[1].grad = None
            errors = []
            for optimizer in (local, reference):
                try:
                    optimizer.step()
                except RuntimeError as error:
                    errors.append(str(error))
                else:
                    raise AssertionError("missing gradient was accepted")
            assert errors[0] == errors[1]
            params[1].grad, references[1].grad = local_grad, reference_grad
        if streams:
            caller = handle.current_stream(device)
            stream = streams[step % len(streams)]
            stream.wait_stream(caller)
            with handle.stream(stream):
                local.step()
                reference.step()
            caller.wait_stream(stream)
        else:
            local.step()
            reference.step()
        for parameter, other in zip(params, references, strict=True):
            torch.testing.assert_close(
                parameter.full_tensor(), other.full_tensor(), atol=0, rtol=0
            )
            torch.testing.assert_close(
                local.state[parameter]["momentum_buffer"].full_tensor(),
                reference.state[other]["momentum_buffer"].full_tensor(),
                atol=0,
                rtol=0,
            )
        assert_released(local)
    if dist.get_rank() == 0:
        print(
            f"PASS {device.type}: mixed buckets, pipelined={pipelined}, "
            "fixed work, empty shards, routes exact",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    reference_path = args.upstream / "torchtitan/distributed/flex_shard"
    if not (reference_path / "dist_muon.py").is_file():
        parser.error(f"missing upstream FlexShard: {reference_path}")
    package = types.ModuleType("reference_flex_shard")
    package.__path__ = [str(reference_path)]
    sys.modules[package.__name__] = package
    reference_muon = importlib.import_module(package.__name__ + ".dist_muon")
    reference_api = importlib.import_module(package.__name__ + ".optimizer_reshard")
    device = init_distributed()
    try:
        size = dist.get_world_size()
        mesh = init_device_mesh(device.type, (size,), mesh_dim_names=("dp",))
        mixed_bucket_case(device, mesh, reference_muon, reference_api)
        mixed_bucket_case(device, mesh, reference_muon, reference_api, pipelined=True)
        for mode, shape in (
            ("owned", (8, 4)),
            ("block", (12, 3)),
            ("batch", (4, 3, 4)),
            ("omitted", (4, 3, 4)),
            ("mesh2d", (8, 3, 4)),
            ("mesh2d_local", (8, 3, 4)),
            ("reversed", (8, 4)),
            ("replicated", (8, 4)),
            ("zero_eps", (8, 4)),
            ("zero_ns", (8, 4)),
            ("boolean_options", (8, 4)),
            ("empty_bucket", (8, 4)),
            ("unused_axis", (8, 4)),
        ):
            storage_mesh = (
                DeviceMesh(
                    device.type, [list(range(size))], mesh_dim_names=("dp", "tp")
                )
                if mode == "omitted"
                else mesh
            )
            if mode in ("mesh2d", "mesh2d_local"):
                if size < 4 or size % 2:
                    continue
                storage_mesh = DeviceMesh(
                    device.type,
                    torch.arange(size).reshape(size // 2, 2),
                    mesh_dim_names=("dp", "tp"),
                )
            elif mode == "reversed":
                storage_mesh = DeviceMesh(
                    device.type, list(reversed(range(size))), mesh_dim_names=("dp",)
                )
            placements = [
                Replicate() if mode == "replicated" else Shard(0)
            ] * storage_mesh.ndim
            if mode == "mesh2d":
                placements[1] = Replicate()
            for scheduled in (False, True):
                torch.manual_seed(42)
                source = torch.randn(shape, device=device)
                local_param = torch.nn.Parameter(
                    distribute_tensor(source.clone(), storage_mesh, placements)
                )
                ref_param = torch.nn.Parameter(
                    distribute_tensor(source.clone(), storage_mesh, placements)
                )
                local_placement = (
                    BlockShard(0, (2, 4))
                    if mode == "block"
                    else Shard(0)
                    if mode == "batch"
                    else Owned()
                )
                ref_placement = (
                    reference_api.BlockShard(0, (2, 4))
                    if mode == "block"
                    else Shard(0)
                    if mode == "batch"
                    else reference_api.Owned()
                )
                coefficients = (3.4445, -4.7750, 2.0315)
                options = dict(
                    lr=0.01,
                    ns_steps=3,
                    ns_coefficients=(coefficients, (3.0, -4.0, 2.0), coefficients)
                    if scheduled
                    else coefficients,
                )
                if mode == "zero_eps":
                    options["eps"] = 0.0
                elif mode == "zero_ns":
                    options.update(
                        ns_steps=0, ns_coefficients=() if scheduled else coefficients
                    )
                elif mode == "boolean_options":
                    options.update(
                        ns_steps=True,
                        eps=True,
                        nesterov=1,
                        ns_coefficients=((True, -4.0, 2.0),)
                        if scheduled
                        else (True, -4.0, 2.0),
                    )
                local_axes = {"dp": local_placement}
                reference_axes = {"dp": ref_placement}
                if mode == "mesh2d_local":
                    local_axes = reference_axes = {"dp": Shard(0), "tp": Shard(0)}
                if mode == "unused_axis":
                    local_axes[""] = reference_axes[""] = Replicate()
                local_buckets = [BucketConfig(("weight",), mesh=storage_mesh)]
                reference_buckets = [
                    reference_api.BucketConfig(("weight",), mesh=storage_mesh)
                ]
                if mode == "empty_bucket":
                    local_buckets = [
                        BucketConfig(()),
                        BucketConfig(("", "weight"), name=None),
                    ]
                    reference_buckets = [
                        reference_api.BucketConfig(()),
                        reference_api.BucketConfig(("", "weight"), name=None),
                    ]
                local = DistMuon(
                    [dict(params=[local_param], param_names=["weight"])],
                    compute_sharding_by_fqn={
                        "weight": ComputeLayout(
                            local_axes, compute_dtype=torch.bfloat16
                        )
                    },
                    bucket_configs=local_buckets,
                    **options,
                )
                reference = reference_muon.DistMuon(
                    [dict(params=[ref_param], param_names=["weight"])],
                    compute_sharding_by_fqn={
                        "weight": reference_api.ComputeLayout(
                            reference_axes, compute_dtype=torch.bfloat16
                        )
                    },
                    bucket_configs=reference_buckets,
                    **options,
                )
                assert_released(local)
                assert plan_signature(local) == plan_signature(reference)
                for _ in range(3):
                    gradient = distribute_tensor(
                        torch.randn(shape, device=device), storage_mesh, placements
                    )
                    local_param.grad = gradient.clone()
                    ref_param.grad = gradient.clone()
                    local.step()
                    reference.step()
                    torch.testing.assert_close(
                        local_param.full_tensor(),
                        ref_param.full_tensor(),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        local.state[local_param]["momentum_buffer"].full_tensor(),
                        reference.state[ref_param]["momentum_buffer"].full_tensor(),
                        atol=0,
                        rtol=0,
                    )
                    assert_released(local)
                if dist.get_rank() == 0:
                    print(
                        f"PASS {device.type}: {mode}, schedule={scheduled}, "
                        "3 updates + momentum exact; device buffers released",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
