"""Real noncoordinator payload write failure, old-generation load and retry.

CPU/Gloo or NPU/HCCL mechanism integration, NOT target-model acceptance.
A directory blocks rank 1's actual DCP shard file; successful writes,
reads, optimizer serialization and recovery all use production implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.components.checkpointer.backends import distributed_checkpoint as backend
from llmtuner.components.checkpointer.core.chunked_tensor import ChunkedCheckpointTensor
from llmtuner.components.checkpointer.core.random_state import RNGCheckpointState
from llmtuner.components.checkpointer.storage.checkpoint_writer import (
    GenerationFileSystemWriter,
)
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import CheckpointConfig, LRSchedulerConfig, OptimizerConfig
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded
from tests.integration_tests.compare_training_checkpoints import (
    _validate_tensor_coverage,
)


class BlockedPayloadWriter(GenerationFileSystemWriter):
    """Arrange a real EISDIR in the background writer, without replacing IO."""

    def write_data(self, plan, planner):
        if dist.get_rank() == 1 and (Path(self.path) / "block-payload").exists():
            assert plan.items, "Failing rank must own actual checkpoint data"
            blocked = Path(self.path) / f"{plan.storage_data.prefix}0.distcp"
            blocked.mkdir()
        return super().write_data(plan, planner)


class CPUProcessStager(backend.DefaultStager):
    """Real shared-memory staging without accelerator-only pin/copy options."""

    def __init__(self, options):
        super().__init__(
            replace(options, use_pinned_memory=False, use_non_blocking_copy=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--async-mode", choices=("async", "async_with_pinned_mem"), default="async"
    )
    parser.add_argument("--cpu-process-staging", action="store_true")
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument(
        "--shard-axis", choices=("dp", "tp", "ep", "joint"), default="dp"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "npu":
        import torch_npu  # noqa: F401

        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
        assert not args.cpu_process_staging
    dist.init_process_group(
        "gloo" if args.device == "cpu" else "hccl",
        timeout=timedelta(seconds=90),
    )
    active = []
    patches = ExitStack()
    if args.cpu_process_staging:
        assert args.async_mode == "async_with_pinned_mem"
        patches.enter_context(patch.object(backend, "DefaultStager", CPUProcessStager))
    try:
        rank = dist.get_rank()
        joint = args.shard_axis == "joint"
        tp_chunks = args.shard_axis in ("tp", "joint")
        ep_chunks = args.shard_axis in ("ep", "joint")
        assert dist.get_world_size() == (
            8 if joint else 2 if args.shard_axis == "dp" else 4
        )
        topology_mesh = init_device_mesh(
            device.type,
            (2, 2, 2) if joint else (2, 1 if args.shard_axis == "dp" else 2),
            mesh_dim_names=("dp_shard", "tp", "ep")
            if joint
            else ("dp_shard", "partition"),
        )
        mesh = (
            topology_mesh["dp_shard", "ep"]._flatten("dense_dp")
            if joint
            else topology_mesh["dp_shard"]
        )
        partition = topology_mesh["tp"] if joint else topology_mesh["partition"]
        expert_mesh = (
            topology_mesh["dp_shard", "tp"]._flatten("expert_dp") if joint else None
        )
        expert_partition = topology_mesh["ep"] if joint else None
        partition_dim = 1 if tp_chunks else 0
        dp_dim = 1 if args.shard_axis == "ep" else 0

        def local_partition(value):
            return value.chunk(partition.size(), dim=partition_dim)[
                partition.get_local_rank()
            ].contiguous()

        global_shapes = (
            {"weight": (7, 10), "experts.weight": (2, 3, 5)}
            if joint
            else {"weight": (8, 4)}
        )

        def initial_value(name):
            shape = global_shapes[name]
            count = 1
            for size in shape:
                count *= size
            return torch.arange(count, device=device).reshape(shape).float()

        def destination(name, value):
            if name == "experts.weight":
                value = value.chunk(2, dim=0)[
                    expert_partition.get_local_rank()
                ].contiguous()
                return distribute_tensor(value, expert_mesh, [Shard(0)])
            return distribute_tensor(local_partition(value), mesh, [Shard(dp_dim)])

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()

        def build():
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(
                destination("weight", initial_value("weight"))
            )
            if joint:
                model.experts = torch.nn.Module()
                model.experts.weight = torch.nn.Parameter(
                    destination("experts.weight", initial_value("experts.weight"))
                )
                mark_experts_ep_sharded(model.experts, expert_partition.get_group())
            if tp_chunks:
                model._tp_checkpoint_layouts = {
                    "weight": (1, False, partition.get_group())
                }
            elif args.shard_axis == "ep":
                mark_experts_ep_sharded(model, partition.get_group())
            optimizer = OptimizersContainer(
                OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
                model_parts=[model],
                sharded_tensor_parallel=tp_chunks,
                sharded_expert_parallel=ep_chunks,
            )
            scheduler = build_lr_scheduler(
                LRSchedulerConfig(decay_ratio=1.0),
                optimizers=optimizer.optimizers,
                training_steps=8,
            )
            rng = RNGCheckpointState(device)
            manager = backend.DistributedCheckpointManager(
                CheckpointConfig(
                    enable=True,
                    interval=1,
                    keep_latest_k=0,
                    async_mode=args.async_mode,
                    sharded_tensor_parallel=tp_chunks,
                    sharded_expert_parallel=ep_chunks,
                ),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=scheduler,
                states={"rng_state": rng},
                folder=str(args.output),
            )
            active.append(manager)
            return model, optimizer, scheduler, manager

        def update(model, optimizer, scheduler, number):
            for name, parameter in model.named_parameters():
                gradient = (initial_value(name) + number).cos() + 2
                parameter.grad = destination(name, gradient)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        def snapshot(model, optimizer, scheduler):
            def project(value):
                if isinstance(value, ChunkedCheckpointTensor):
                    return {
                        "shape": tuple(value.shape),
                        "chunks": [
                            (tuple(offset), tensor.clone().cpu())
                            for offset, tensor in value.chunks
                        ],
                    }
                if isinstance(value, dict):
                    return {key: project(item) for key, item in value.items()}
                return value

            return deepcopy(
                (
                    model.state_dict(),
                    project(optimizer.state_dict()),
                    scheduler.state_dict(),
                )
            )

        def expected_local_boxes(name):
            # Independent oracle for rank order (dp, tp, ep), without calling
            # production offset helpers or reading checkpoint metadata.
            dp, tp, ep = topology_mesh.get_coordinate()
            if name == "weight":
                row_rank = dp * 2 + ep
                return [((row_rank * 2, tp * 5), (2 if row_rank < 3 else 1, 5))]
            return [((ep, 0, 0), (1, 3, 5))] if dp * 2 + tp == 0 else []

        ownership = {}

        def validate_joint_views(model_state, optimizer_state):
            for name, shape in global_shapes.items():
                expected_boxes = expected_local_boxes(name)
                ownership[name] = expected_boxes
                for label, value in (
                    (name, model_state[name]),
                    (f"state.{name}.exp_avg", optimizer_state[f"state.{name}.exp_avg"]),
                    (
                        f"state.{name}.exp_avg_sq",
                        optimizer_state[f"state.{name}.exp_avg_sq"],
                    ),
                ):
                    assert isinstance(value, ChunkedCheckpointTensor), label
                    assert tuple(value.shape) == shape, label
                    boxes = [
                        (tuple(offset), tuple(tensor.shape))
                        for offset, tensor in value.chunks
                    ]
                    assert boxes == expected_boxes, (rank, label, boxes, expected_boxes)
                    for _, tensor in value.chunks:
                        assert tensor.device.type == device.type
                        if label.startswith("state."):
                            assert (
                                torch.isfinite(tensor).all()
                                and (tensor.abs() > 0).all()
                            )

        model, optimizer, scheduler, manager = build()
        torch.manual_seed(911 + rank)
        update(model, optimizer, scheduler, 1)
        if joint:
            validate_joint_views(
                manager.states["model"].state_dict(), optimizer.state_dict()
            )
        saved = snapshot(model, optimizer, scheduler)
        saved_rng = torch.get_rng_state().clone()
        saved_device_rng = (
            torch.npu.get_rng_state(device).clone() if device.type == "npu" else None
        )
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        checkpoint = args.output / "checkpoint/step-1"
        old_metadata = (checkpoint / ".metadata").read_bytes()
        if joint:
            metadata = pickle.loads(old_metadata)
            for name, shape in global_shapes.items():
                if name == "weight":
                    expected_boxes = {
                        ((row * 2, col * 5), (2 if row < 3 else 1, 5))
                        for row in range(4)
                        for col in range(2)
                    }
                else:
                    expected_boxes = {
                        ((expert, 0, 0), (1, 3, 5)) for expert in range(2)
                    }
                for key in (
                    name,
                    f"optimizer.state.{name}.exp_avg",
                    f"optimizer.state.{name}.exp_avg_sq",
                ):
                    spec = metadata.state_dict_metadata[key]
                    assert tuple(spec.size) == shape
                    assert {
                        (tuple(chunk.offsets), tuple(chunk.sizes))
                        for chunk in spec.chunks
                    } == expected_boxes, key
                    _validate_tensor_coverage(spec, key)
        elif args.shard_axis != "dp":
            metadata = pickle.loads(old_metadata)
            for key in (
                "weight",
                "optimizer.state.weight.exp_avg",
                "optimizer.state.weight.exp_avg_sq",
            ):
                spec = metadata.state_dict_metadata[key]
                assert tuple(spec.size) == (8, 4) and len(spec.chunks) == 4
                _validate_tensor_coverage(spec, key)
            assert isinstance(
                optimizer.state_dict()["state.weight.exp_avg"], ChunkedCheckpointTensor
            )
        update(model, optimizer, scheduler, 2)
        expected = snapshot(model, optimizer, scheduler)
        if rank == 0:
            (checkpoint / "block-payload").touch()
        dist.barrier()
        with patch.object(backend, "GenerationFileSystemWriter", BlockedPayloadWriter):
            assert manager.save(1)
            try:
                manager.maybe_wait_for_saving()
            except BaseException as error:
                assert "IsADirectoryError" in str(error), str(error)
                failure = str(error)
            else:
                raise AssertionError("Noncoordinator payload failure was accepted")
        assert manager.save_future is None
        assert (checkpoint / ".metadata").read_bytes() == old_metadata
        torch.testing.assert_close(
            snapshot(model, optimizer, scheduler), expected, rtol=0, atol=0
        )
        dist.barrier()

        cold_model, cold_optimizer, cold_scheduler, cold = build()
        assert all(not optim.state for optim in cold_optimizer.optimizers)
        # Make omission of RNG restoration observable, rather than comparing
        # streams that deterministic arange/cos updates never advanced.
        torch.rand(17)
        assert not torch.equal(torch.get_rng_state(), saved_rng)
        if saved_device_rng is not None:
            torch.rand(17, device=device)
            assert not torch.equal(torch.npu.get_rng_state(device), saved_device_rng)
        assert cold.load(1)
        torch.testing.assert_close(
            snapshot(cold_model, cold_optimizer, cold_scheduler), saved, rtol=0, atol=0
        )
        assert torch.equal(torch.get_rng_state(), saved_rng)
        if saved_device_rng is not None:
            assert torch.equal(torch.npu.get_rng_state(device), saved_device_rng)
        update(cold_model, cold_optimizer, cold_scheduler, 2)
        torch.testing.assert_close(
            snapshot(cold_model, cold_optimizer, cold_scheduler),
            expected,
            rtol=0,
            atol=0,
        )
        cold.close()
        if rank == 0:
            (checkpoint / "block-payload").unlink()
        dist.barrier()
        # Retry the SAME live manager after its failed future has been consumed.
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        assert (checkpoint / ".metadata").read_bytes() != old_metadata
        fresh_model, fresh_optimizer, fresh_scheduler, fresh = build()
        assert fresh.load(1)
        torch.testing.assert_close(
            snapshot(fresh_model, fresh_optimizer, fresh_scheduler),
            expected,
            rtol=0,
            atol=0,
        )
        for parts in (
            (model, optimizer, scheduler),
            (fresh_model, fresh_optimizer, fresh_scheduler),
        ):
            update(*parts, 3)
        torch.testing.assert_close(
            snapshot(fresh_model, fresh_optimizer, fresh_scheduler),
            snapshot(model, optimizer, scheduler),
            rtol=0,
            atol=0,
        )
        report = {
            "rank": rank,
            "passed": True,
            "async_mode": args.async_mode,
            "cpu_process_staging": args.cpu_process_staging,
            "device": str(device),
            "shard_axis": args.shard_axis,
            "world_size": dist.get_world_size(),
            "global_shapes": global_shapes,
            "joint_coordinates": topology_mesh.get_coordinate() if joint else None,
            "joint_local_ownership": ownership,
            "empty_expert_ownership": joint and not ownership["experts.weight"],
            "scope": ("Sharded backend mechanism; not six-model target acceptance"),
            "noncoordinator_error": failure,
            "old_generation_cold_load_and_next_update_exact": True,
            "same_manager_retry_cold_load_and_next_update_exact": True,
            "rng_restored_exact": True,
            "rng_perturbed_before_restore": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (args.output / f"receipt-rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"PASS rank={rank} real payload failure, old load, retry and continuation",
            flush=True,
        )
        dist.barrier()
    finally:
        for manager in active:
            manager.close()
        patches.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
