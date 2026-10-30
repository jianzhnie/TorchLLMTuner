"""Four CPU or accelerator ranks: EP2 x DP2 with an empty local expert shard."""

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded
from tests.integration_tests.harness import init_distributed


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
            tempfile.mkdtemp(prefix="empty-expert-checkpoint-")
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
        dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: EP2 x DP2 empty local shard DCP save and cold restore")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
