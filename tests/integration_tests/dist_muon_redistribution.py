"""Two ranks: DistMuon shard-to-owner updates match a full-matrix reference.

Run with ``torchrun --standalone --nproc_per_node=2``. CPU/Gloo exercises the
packed redistribution protocol without making an NPU performance claim.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from llmtuner.components.optimizer.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.components.optimizer.state_dict import collect_local_optimizer_state
from llmtuner.config import OptimizerConfig, ParamGroupConfig
from tests.integration_tests.harness import init_distributed


def _model(
    mesh: DeviceMesh,
    local_weight: torch.Tensor,
    placement: Shard | Replicate,
    *,
    global_shape: tuple[int, int] = (4, 4),
) -> torch.nn.Module:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        DTensor.from_local(
            local_weight.clone(),
            mesh,
            (placement,),
            shape=global_shape,
            stride=(global_shape[1], 1),
        )
    )
    return model


def _optimizer(
    model: torch.nn.Module, compute_sharding: Owned | BlockShard
) -> OptimizersContainer:
    return OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="DistMuon",
                    optimizer_kwargs={"lr": 1e-3},
                )
            ],
            compute_sharding_by_fqn={
                "weight": ComputeLayout(
                    {"dp_shard": compute_sharding}, compute_dtype=torch.bfloat16
                )
            },
            bucket_configs=[BucketConfig(patterns=("weight",))],
        ),
        model_parts=[model],
    )


def main() -> None:
    device = init_distributed()
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert world_size == 2

        full_mesh = DeviceMesh(
            device.type, list(range(world_size)), mesh_dim_names=("dp_shard",)
        )
        initial = (
            torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) / 16
        )
        gradient = (
            torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) / 32
        )
        local_slice = slice(rank * 2, (rank + 1) * 2)
        sharded_model = _model(full_mesh, initial[local_slice], Shard(0))
        reference_model = _model(full_mesh, initial, Replicate())
        sharded_optimizer = _optimizer(sharded_model, Owned())
        reference_optimizer = _optimizer(reference_model, Owned())

        # Only rank 0 already owns checkpoint state. Local preparation must not
        # enter an optimizer collective skipped by that rank.
        inner = sharded_optimizer.optimizers[0]
        if rank == 0:
            inner.state[sharded_model.weight]["momentum_buffer"] = torch.zeros_like(
                sharded_model.weight
            )
        local_state = collect_local_optimizer_state(sharded_optimizer.optimizers)
        assert "state.weight.momentum_buffer" in local_state
        assert sharded_model.weight.grad is None
        assert not inner._first_step_validated
        torch.testing.assert_close(
            sharded_model.weight.to_local(), initial[local_slice], rtol=0, atol=0
        )

        for scale in (1.0, -0.25):
            sharded_model.weight.grad = DTensor.from_local(
                (gradient * scale)[local_slice].clone(),
                full_mesh,
                (Shard(0),),
                shape=(4, 4),
                stride=(4, 1),
            )
            reference_model.weight.grad = DTensor.from_local(
                gradient * scale,
                full_mesh,
                (Replicate(),),
                shape=(4, 4),
                stride=(4, 1),
            )
            sharded_optimizer.step()
            reference_optimizer.step()
            sharded_optimizer.zero_grad()
            reference_optimizer.zero_grad()
            torch.testing.assert_close(
                sharded_model.weight.to_local(),
                reference_model.weight.to_local()[local_slice],
                rtol=1e-3,
                atol=1e-3,
            )

        # The storage shard boundary at row 9 cuts a four-row block. Compute
        # ownership must follow whole (2, 4) row blocks and redistribute rows.
        global_shape = (18, 4)
        initial_blocks = (
            torch.arange(
                global_shape[0] * global_shape[1], dtype=torch.float32, device=device
            ).reshape(global_shape)
            / 32
        )
        gradient_blocks = (
            torch.arange(
                global_shape[0] * global_shape[1], dtype=torch.float32, device=device
            ).reshape(global_shape)
            / 64
        )
        shard = slice(rank * 9, (rank + 1) * 9)
        block_layout = BlockShard(dim=0, block_sizes=(2, 4))
        block_sharded_model = _model(
            full_mesh, initial_blocks[shard], Shard(0), global_shape=global_shape
        )
        block_reference_model = _model(
            full_mesh, initial_blocks, Replicate(), global_shape=global_shape
        )
        block_sharded_optimizer = _optimizer(block_sharded_model, block_layout)
        block_reference_optimizer = _optimizer(block_reference_model, block_layout)

        for scale in (1.0, -0.25):
            block_sharded_model.weight.grad = DTensor.from_local(
                (gradient_blocks * scale)[shard].clone(),
                full_mesh,
                (Shard(0),),
                shape=global_shape,
                stride=(global_shape[1], 1),
            )
            block_reference_model.weight.grad = DTensor.from_local(
                gradient_blocks * scale,
                full_mesh,
                (Replicate(),),
                shape=global_shape,
                stride=(global_shape[1], 1),
            )
            block_sharded_optimizer.step()
            block_reference_optimizer.step()
            block_sharded_optimizer.zero_grad()
            block_reference_optimizer.zero_grad()
            torch.testing.assert_close(
                block_sharded_model.weight.to_local(),
                block_reference_model.weight.to_local()[shard],
                rtol=1e-3,
                atol=1e-3,
            )

        tp_mesh = DeviceMesh(device.type, [[0, 1]], mesh_dim_names=("dp_shard", "tp"))
        tp_model = torch.nn.Module()
        # Omission preserves storage Shard, which upstream rejects for a 2D
        # matrix: its rows may not be treated as independent matrices.
        tp_model.weight = torch.nn.Parameter(
            DTensor.from_local(
                initial[local_slice].clone(),
                tp_mesh,
                (Shard(0), Shard(0)),
                shape=(4, 4),
                stride=(4, 1),
            )
        )
        try:
            _optimizer(tp_model, Owned())
        except ValueError as error:
            assert "2D Muon compute cannot use Shard" in str(error)
        else:
            raise AssertionError("DistMuon accepted an incomplete 2D matrix")

        print(
            f"rank {rank}: Owned and misaligned BlockShard redistribution "
            "match references; incomplete 2D matrix is rejected",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
