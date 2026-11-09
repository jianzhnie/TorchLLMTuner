"""CPU/Gloo: reject rank-local optimizer policy errors before applying state.

Run with PYTHONPATH=. torchrun --standalone --nproc_per_node=2
tests/integration_tests/optimizer_load_coordination.py.
This validates collective ordering, not NPU virtual-storage allocation.
"""

from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
    BucketConfig,
    ComputeLayout,
    Owned,
)
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig
from llmtuner.errors import UnsupportedCombinationError


def check_dist_muon_group_rejection() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dp_shard",))
    for failing_rank in range(world_size):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(
            DTensor.from_local(
                torch.full((2, 4), float(rank + 1)),
                mesh,
                (Shard(0),),
                shape=(2 * world_size, 4),
                stride=(4, 1),
            )
        )
        optimizer = OptimizersContainer(
            OptimizerConfig(
                param_groups=[
                    ParamGroupConfig(pattern=".*", optimizer_name="DistMuon")
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
        original = deepcopy(optimizer.state_dict())
        incoming = deepcopy(original)
        incoming["param_groups.weight.lr"] = 0.5
        if rank == failing_rank:
            incoming["param_groups.weight.ns_steps"] = -1
        try:
            optimizer.load_state_dict(incoming)
        except ValueError as error:
            assert rank == failing_rank
            assert "unsupported DistMuon group" in str(error)
        except RuntimeError as error:
            assert rank != failing_rank
            assert "Optimizer checkpoint validation" in str(error)
        else:
            raise AssertionError("invalid DistMuon checkpoint options were accepted")
        for key, value in optimizer.state_dict().items():
            expected = original[key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                assert value == expected, key
        dist.barrier()
        optimizer.load_state_dict(original)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        assert torch.isfinite(model.weight.to_local()).all()
        dist.barrier()


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        for failing_rank in range(dist.get_world_size()):
            model = torch.nn.Linear(2, 1)
            optimizer = OptimizersContainer(
                OptimizerConfig(
                    implementation="for-loop",
                    param_groups=[
                        ParamGroupConfig(pattern="weight$", optimizer_name="Adam"),
                        ParamGroupConfig(pattern="bias$", optimizer_name="AdamW"),
                    ],
                ),
                model_parts=[model],
            )
            # Exercise the storage policy's validation boundary with ordinary
            # CPU moments. No NPU allocator or collective is emulated here.
            optimizer.state_residency = "npu_virtual"
            original = deepcopy(optimizer.state_dict())
            incoming = deepcopy(original)
            first, second = optimizer.optimizers
            first_name = first.param_groups[0]["param_names"][0]
            second_name = second.param_groups[0]["param_names"][0]
            original_lr = first.param_groups[0]["lr"]
            incoming[f"param_groups.{first_name}.lr"] = original_lr + 1
            if rank == failing_rank:
                incoming[f"param_groups.{second_name}.fused"] = True
            try:
                optimizer.load_state_dict(incoming)
            except UnsupportedCombinationError:
                assert rank == failing_rank
            except RuntimeError as error:
                assert rank != failing_rank
                assert "Optimizer checkpoint validation" in str(error)
            else:
                raise AssertionError("rank-local invalid options were accepted")
            assert first.param_groups[0]["lr"] == original_lr
            for key, value in optimizer.state_dict().items():
                expected = original[key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    assert value == expected, key
            # Peers must reach the next collective and a valid retry together.
            dist.barrier()
            optimizer.load_state_dict(original)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            dist.barrier()
        check_dist_muon_group_rejection()
        if rank == 0:
            print("PASS coordinated optimizer policy rejection and retry", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
