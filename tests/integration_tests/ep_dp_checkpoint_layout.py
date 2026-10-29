"""Four Gloo ranks: EP2 x DP2 checkpoint offsets and model restoration."""

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig
from llmtuner.parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    mark_experts_ep_sharded,
    restore_sharded_expert_state,
)


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("ep", "dp"))
        ep_rank = mesh["ep"].get_local_rank()
        dp_rank = mesh["dp"].get_local_rank()
        full = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
        for dim in (0, 1, 2):
            model = nn.Module()
            model.experts = nn.Module()
            ep_slice = full.chunk(2, dim=0)[ep_rank].contiguous()
            value = distribute_tensor(ep_slice, mesh["dp"], [Shard(dim)])
            model.experts.weight = nn.Parameter(value)
            mark_experts_ep_sharded(model.experts, mesh["ep"].get_group())
            expected = model.experts.weight.to_local().clone()
            wrapper = ModelCheckpointState(model, sharded_expert_parallel=True)
            chunk = wrapper.state_dict()["experts.weight"]
            expected_offset = [ep_rank * 2, 0, 0]
            expected_offset[dim] += dp_rank * (ep_slice.shape[dim] // 2)
            assert chunk.shape == full.shape
            assert chunk.chunks[0][0] == tuple(expected_offset)
            assert chunk.chunks[0][1].data_ptr() == value.to_local().data_ptr()
            paths = [
                tempfile.mkdtemp(prefix="ep-dp-checkpoint-")
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
            cold = nn.Module()
            cold.experts = nn.Module()
            cold.experts.weight = nn.Parameter(
                distribute_tensor(
                    torch.full_like(ep_slice, float("nan")), mesh["dp"], [Shard(dim)]
                )
            )
            mark_experts_ep_sharded(cold.experts, mesh["ep"].get_group())
            cold_wrapper = ModelCheckpointState(cold, sharded_expert_parallel=True)
            dcp.load({"model": cold_wrapper}, checkpoint_id=paths[0])
            torch.testing.assert_close(
                cold.experts.weight.to_local(), expected, atol=0, rtol=0
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
                sharded_expert_parallel=True,
            )
            model.experts.weight.grad = torch.ones_like(model.experts.weight)
            optimizer.step()
            optimizer.zero_grad()
            moment_key = "state.experts.weight.exp_avg"
            moment = optimizer.state_dict()[moment_key]
            expected_moment = moment.chunks[0][1].clone()
            dcp.save({"optimizer": optimizer}, checkpoint_id=paths[0] + "-optim")
            moment.chunks[0][1].fill_(float("nan"))
            dcp.load({"optimizer": optimizer}, checkpoint_id=paths[0] + "-optim")
            torch.testing.assert_close(
                optimizer.state_dict()[moment_key].chunks[0][1],
                expected_moment,
                atol=0,
                rtol=0,
            )
            resume_path = paths[0] + "-resume"
            ema = EMA(model_parts=[model], sharded_expert_parallel=True)
            ema.step(1)
            dcp.save(
                {"model": wrapper, "optimizer": optimizer, "ema": ema},
                checkpoint_id=resume_path,
            )
            cold_optimizer = OptimizersContainer(
                OptimizerConfig(
                    implementation="for-loop",
                    param_groups=[
                        ParamGroupConfig(
                            pattern=".*",
                            optimizer_name="AdamW",
                            optimizer_kwargs={"lr": 0.9},
                        )
                    ],
                ),
                model_parts=[cold],
                sharded_expert_parallel=True,
            )
            assert all(not inner.state for inner in cold_optimizer.optimizers)
            cold_ema = EMA(model_parts=[cold], sharded_expert_parallel=True)
            dcp.load(
                {"model": cold_wrapper, "optimizer": cold_optimizer, "ema": cold_ema},
                checkpoint_id=resume_path,
            )
            for runtime_model, runtime_optimizer in (
                (model, optimizer),
                (cold, cold_optimizer),
            ):
                runtime_model.experts.weight.grad = torch.full_like(
                    runtime_model.experts.weight, 0.25
                )
                runtime_optimizer.step()
                runtime_optimizer.zero_grad()
            ema.step(2)
            cold_ema.step(2)
            expected_ema_state = ema.state_dict()
            actual_ema_state = cold_ema.state_dict()
            assert expected_ema_state.keys() == actual_ema_state.keys()
            for key, expected_ema in expected_ema_state.items():
                actual_ema = actual_ema_state[key]
                if hasattr(expected_ema, "chunks"):
                    for (offset, actual), (expected_offset, expected_value) in zip(
                        actual_ema.chunks, expected_ema.chunks, strict=True
                    ):
                        assert offset == expected_offset
                        torch.testing.assert_close(
                            actual, expected_value, atol=0, rtol=0
                        )
                elif isinstance(expected_ema, torch.Tensor):
                    torch.testing.assert_close(actual_ema, expected_ema, atol=0, rtol=0)
                else:
                    assert actual_ema == expected_ema
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer.optimizers[0], lambda _: 1.0
            )
            torch.optim.lr_scheduler.LambdaLR(
                cold_optimizer.optimizers[0], lambda _: 1.0
            )
            manager = DistributedCheckpointManager(
                CheckpointConfig(
                    enable=True,
                    interval=1,
                    sharded_expert_parallel=True,
                    keep_latest_k=0,
                ),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=scheduler,
                ema=ema,
                states={},
                folder=paths[0] + "-manager",
            )
            expected_manager_weight = model.experts.weight.to_local().clone()
            try:
                assert manager.save(2)
                manager.maybe_wait_for_saving()
                with torch.no_grad():
                    model.experts.weight.fill_(float("nan"))
                assert manager.load(2)
                torch.testing.assert_close(
                    model.experts.weight.to_local(),
                    expected_manager_weight,
                    atol=0,
                    rtol=0,
                )
            finally:
                manager.close()
            torch.testing.assert_close(
                cold.experts.weight.to_local(),
                model.experts.weight.to_local(),
                atol=0,
                rtol=0,
            )
            continuous = optimizer.state_dict()
            resumed = cold_optimizer.state_dict()
            assert continuous.keys() == resumed.keys()
            for key, expected_state in continuous.items():
                actual_state = resumed[key]
                if hasattr(expected_state, "chunks"):
                    for (offset, actual), (expected_offset, expected_value) in zip(
                        actual_state.chunks, expected_state.chunks, strict=True
                    ):
                        assert offset == expected_offset
                        torch.testing.assert_close(
                            actual, expected_value, atol=0, rtol=0
                        )
                elif isinstance(expected_state, torch.Tensor):
                    torch.testing.assert_close(
                        actual_state, expected_state, atol=0, rtol=0
                    )
                else:
                    assert actual_state == expected_state
            dcp.load({"model": wrapper}, checkpoint_id=paths[0])
            complete = {"model": {"experts.weight": torch.empty_like(full)}}
            dcp.load(complete, checkpoint_id=paths[0])
            torch.testing.assert_close(
                complete["model"]["experts.weight"], full, atol=0, rtol=0
            )
            with torch.no_grad():
                model.experts.weight.fill_(float("nan"))
            wrapper.load_state_dict(complete["model"])
            torch.testing.assert_close(
                model.experts.weight.to_local(), expected, atol=0, rtol=0
            )
            bad_source = dict(complete["model"])
            if dist.get_rank() == 0:
                bad_source["experts.weight"] = torch.zeros(1)
            try:
                restore_sharded_expert_state(
                    bad_source,
                    local_state=model.state_dict(),
                    shard_map=expert_shard_map([model]),
                    model_parts=[model],
                    coordinate_errors=True,
                )
            except (ValueError, RuntimeError) as error:
                if dist.get_rank() == 0:
                    assert "global shape mismatch" in str(error)
                else:
                    assert "failed on another rank" in str(error)
            else:
                raise AssertionError("Invalid peer checkpoint must fail on all ranks")
            torch.testing.assert_close(
                model.experts.weight.to_local(), expected, atol=0, rtol=0
            )
        if dist.get_rank() == 0:
            print(
                "PASS: EP2 x DP2 model checkpoint across all three DP shard dimensions"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
