"""Four CPU or accelerator ranks: EP2 x DP2 with an empty local expert shard.

For multiple nodes, set CHECKPOINT_TEST_ROOT to an existing shared directory.
"""

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
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.tp_dp_checkpoint_reshard import build_optimizer


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh(device.type, (2, 2), mesh_dim_names=("ep", "dp"))
        ep_rank = mesh["ep"].get_local_rank()
        dp_rank = mesh["dp"].get_local_rank()
        full = torch.arange(4 * 8, dtype=torch.float32, device=device).reshape(4, 1, 8)
        ep_slice = full.chunk(2, dim=0)[ep_rank].contiguous()

        def build_model():
            model = nn.Module()
            model.experts = nn.Module()
            model.experts.weight = nn.Parameter(
                distribute_tensor(ep_slice.clone(), mesh["dp"], [Shard(1)])
            )
            mark_experts_ep_sharded(model.experts, mesh["ep"].get_group())
            return model

        model = build_model()
        assert (model.experts.weight.to_local().numel() == 0) == (dp_rank == 1)
        wrapper = ModelCheckpointState(model, sharded_expert_parallel=True)
        checkpoint = wrapper.state_dict()["experts.weight"]
        assert checkpoint.shape == full.shape
        assert (checkpoint.chunks == ()) == (dp_rank == 1)
        assert checkpoint.dtype == full.dtype
        assert checkpoint.device == model.experts.weight.device
        expected = model.experts.weight.to_local().clone()
        paths = [
            tempfile.mkdtemp(
                prefix="empty-expert-checkpoint-",
                dir=os.environ.get("CHECKPOINT_TEST_ROOT"),
            )
            if dist.get_rank() == 0
            else None
        ]
        dist.broadcast_object_list(paths)
        dcp.save({"model": wrapper}, checkpoint_id=paths[0])
        with torch.no_grad():
            model.experts.weight.fill_(float("nan"))
        dcp.load({"model": wrapper}, checkpoint_id=paths[0])
        torch.testing.assert_close(
            model.experts.weight.to_local(), expected, atol=0, rtol=0
        )

        cold = build_model()
        cold_wrapper = ModelCheckpointState(cold, sharded_expert_parallel=True)
        dcp.load({"model": cold_wrapper}, checkpoint_id=paths[0])
        torch.testing.assert_close(
            cold.experts.weight.to_local(), expected, atol=0, rtol=0
        )

        complete = {"model": {"experts.weight": torch.empty_like(full)}}
        dcp.load(complete, checkpoint_id=paths[0])
        torch.testing.assert_close(
            complete["model"]["experts.weight"], full, atol=0, rtol=0
        )

        optimizer = build_optimizer(model, shard_axis="ep")
        reference = nn.Parameter(full.clone())
        reference_optimizer = torch.optim.AdamW(
            [reference], lr=0.01, weight_decay=0.01, foreach=False, fused=False
        )
        reference.grad = (full + 1) / 100
        reference_optimizer.step()
        model.experts.weight.grad = distribute_tensor(
            (ep_slice + 1) / 100, mesh["dp"], [Shard(1)]
        )
        optimizer.step()
        optimizer.zero_grad()
        optimizer_path = paths[0] + "-optimizer"
        dcp.save(
            {"model": wrapper, "optimizer": optimizer},
            checkpoint_id=optimizer_path,
        )
        cold = build_model()
        with torch.no_grad():
            cold.experts.weight.zero_()
        cold_wrapper = ModelCheckpointState(cold, sharded_expert_parallel=True)
        cold_optimizer = build_optimizer(cold, cold=True, shard_axis="ep")
        assert all(not inner.state for inner in cold_optimizer.optimizers)
        dcp.load(
            {"model": cold_wrapper, "optimizer": cold_optimizer},
            checkpoint_id=optimizer_path,
        )
        reference.grad = full.cos() + 2
        reference_optimizer.step()
        cold.experts.weight.grad = distribute_tensor(
            ep_slice.cos() + 2, mesh["dp"], [Shard(1)]
        )
        cold_optimizer.step()
        expected = reference.detach().chunk(2, dim=0)[ep_rank][
            :, dp_rank : dp_rank + 1, :
        ]
        torch.testing.assert_close(
            cold.experts.weight.to_local(), expected, atol=0, rtol=0
        )
        restored_state = cold_optimizer.state_dict()
        for name, value in reference_optimizer.state[reference].items():
            actual = restored_state[f"state.experts.weight.{name}"]
            if name == "step":
                torch.testing.assert_close(actual, value, atol=0, rtol=0)
                continue
            assert (actual.chunks == ()) == (dp_rank == 1)
            for offset, chunk in actual.chunks:
                slices = tuple(
                    slice(start, start + size)
                    for start, size in zip(offset, chunk.shape, strict=True)
                )
                torch.testing.assert_close(chunk, value[slices], atol=0, rtol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                "PASS: EP2 x DP2 empty local shard model/optimizer cold restore "
                "and next step"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
