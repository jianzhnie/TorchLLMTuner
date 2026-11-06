"""Four ranks: disjoint TP/EP state with empty DP ownership and cold restore.

Run explicitly with torchrun --standalone --nproc-per-node=4.
CHECKPOINT_TEST_ROOT may select a shared directory for multi-node execution.
Use --phase save/load with the same --checkpoint for independent-process restore.
--async-save uses a Gloo upload group and waits for upload completion before
proceeding; it does not test overlap with subsequent optimizer updates.
"""

import argparse
import os
import tempfile
from datetime import timedelta

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
from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("combined", "save", "load"), default="combined"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--async-save", action="store_true")
    args = parser.parse_args()
    if args.phase != "combined" and not args.checkpoint:
        parser.error("--checkpoint is required for separate save/load phases")
    device = init_distributed(timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("dp", "partition"))
        partition = mesh["partition"].get_local_rank()
        dp_rank = mesh["dp"].get_local_rank()
        full = {
            "dense": torch.arange(8.0, device=device).reshape(1, 8),
            "experts.weight": torch.arange(32.0, device=device).reshape(4, 1, 8),
        }
        local = {
            "dense": full["dense"].chunk(2, dim=1)[partition].contiguous(),
            "experts.weight": full["experts.weight"]
            .chunk(2, dim=0)[partition]
            .contiguous(),
        }

        def build():
            model = nn.Module()
            model.dense = nn.Parameter(
                distribute_tensor(local["dense"].clone(), mesh["dp"], [Shard(0)])
            )
            model._tp_checkpoint_layouts = {
                "dense": (1, False, mesh["partition"].get_group())
            }
            model.experts = nn.Module()
            model.experts.weight = nn.Parameter(
                distribute_tensor(
                    local["experts.weight"].clone(), mesh["dp"], [Shard(1)]
                )
            )
            mark_experts_ep_sharded(model.experts, mesh["partition"].get_group())
            wrapper = ModelCheckpointState(
                model, sharded_tensor_parallel=True, sharded_expert_parallel=True
            )
            optimizer = OptimizersContainer(
                OptimizerConfig(
                    implementation="for-loop",
                    param_groups=[
                        ParamGroupConfig(
                            pattern=".*",
                            optimizer_name="AdamW",
                            optimizer_kwargs={"lr": 0.01},
                        )
                    ],
                ),
                model_parts=[model],
                sharded_tensor_parallel=True,
                sharded_expert_parallel=True,
            )
            return model, wrapper, optimizer

        reference = {key: nn.Parameter(value.clone()) for key, value in full.items()}
        native = torch.optim.AdamW(
            list(reference.values()),
            lr=0.01,
            weight_decay=0.01,
            foreach=False,
            fused=False,
        )
        for key, parameter in reference.items():
            parameter.grad = (full[key] + 1) / 100
        native.step()
        paths = [args.checkpoint]
        if args.phase != "load":
            model, wrapper, optimizer = build()
            for key, parameter in model.named_parameters():
                axis = 0 if key == "dense" else 1
                assert (parameter.to_local().numel() == 0) == (dp_rank == 1)
                parameter.grad = distribute_tensor(
                    (local[key] + 1) / 100, mesh["dp"], [Shard(axis)]
                )
            optimizer.step()
            optimizer.zero_grad()
            if args.checkpoint is None and dist.get_rank() == 0:
                paths[0] = tempfile.mkdtemp(
                    prefix="joint-empty-", dir=os.environ.get("CHECKPOINT_TEST_ROOT")
                )
            dist.broadcast_object_list(paths)
            state = {"model": wrapper, "optimizer": optimizer}
            if args.async_save:
                # Async DCP stages CPU tensors; HCCL alone is insufficient.
                upload_group = dist.new_group(backend="gloo")
                try:
                    dcp.async_save(
                        state, checkpoint_id=paths[0], process_group=upload_group
                    ).result()
                finally:
                    dist.destroy_process_group(upload_group)
            else:
                dcp.save(state, checkpoint_id=paths[0])
            if args.phase == "save":
                print(f"PASS joint empty checkpoint saved rank={dist.get_rank()}")
                return
        cold, cold_wrapper, cold_optimizer = build()
        assert all(not inner.state for inner in cold_optimizer.optimizers)
        with torch.no_grad():
            for parameter in cold.parameters():
                parameter.fill_(float("nan"))
        dcp.load(
            {"model": cold_wrapper, "optimizer": cold_optimizer}, checkpoint_id=paths[0]
        )
        for key, parameter in cold.named_parameters():
            axis = 0 if key == "dense" else 1
            parameter.grad = distribute_tensor(
                local[key].cos() + 2, mesh["dp"], [Shard(axis)]
            )
            reference[key].grad = full[key].cos() + 2
        cold_optimizer.step()
        native.step()
        restored = cold_optimizer.state_dict()
        for key, parameter in cold.named_parameters():
            partition_dim = 1 if key == "dense" else 0
            dp_dim = 0 if key == "dense" else 1
            expected = reference[key].detach().chunk(2, dim=partition_dim)[partition]
            expected = expected.narrow(
                dp_dim, min(dp_rank, expected.shape[dp_dim]), 1 if dp_rank == 0 else 0
            )
            torch.testing.assert_close(parameter.to_local(), expected, rtol=0, atol=0)
            for name, value in native.state[reference[key]].items():
                actual = restored[f"state.{key}.{name}"]
                if name == "step":
                    torch.testing.assert_close(actual, value, rtol=0, atol=0)
                    continue
                assert (not actual.chunks) == (dp_rank == 1)
                for offset, chunk in actual.chunks:
                    slices = tuple(
                        slice(start, start + size)
                        for start, size in zip(offset, chunk.shape, strict=True)
                    )
                    torch.testing.assert_close(chunk, value[slices], rtol=0, atol=0)
            assert dp_rank == 1 or not torch.equal(parameter.to_local(), local[key])
        print(
            "PASS joint TP/EP empty ownership cold restore and next step "
            f"rank={dist.get_rank()}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
