"""Four CPU/Gloo ranks: packed TP2 x DP2 checkpoint values and placements."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from llmtuner.components.checkpointer.hf_storage import IndexedHuggingFaceStorageReader
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig
from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
    restore_sharded_tensor_parallel_state,
    shard_tensor_parallel_state,
)


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        tp = mesh["tp"].get_group()
        tp_rank = dist.get_rank(tp)
        dp_rank = dist.get_rank(mesh["dp"].get_group())
        dp_dim = int(os.environ.get("LLMTUNER_TEST_DP_SHARD_DIM", "0"))
        assert dp_dim in (0, 1, 2)
        uneven = os.environ.get("LLMTUNER_TEST_UNEVEN_DP") == "1"
        empty_dp = os.environ.get("LLMTUNER_TEST_EMPTY_DP") == "1"
        assert not empty_dp or dp_dim == 0
        shape = (1, 8, 6) if empty_dp else (5, 8, 7) if uneven else (4, 8, 6)
        full = torch.arange(
            shape[0] * shape[1] * shape[2], dtype=torch.float32
        ).reshape(shape)
        for packed in (False, True):
            local = (
                torch.cat([half.chunk(2, 1)[tp_rank] for half in full.chunk(2, 1)], 1)
                if packed
                else full.chunk(2, 1)[tp_rank]
            ).contiguous()
            model = nn.Module()
            parameter = distribute_tensor(local, mesh["dp"], [Shard(dp_dim)])
            model.register_parameter("weight", nn.Parameter(parameter))
            model._tp_checkpoint_layouts = {"weight": (1, packed, tp)}
            wrapper = ModelCheckpointState(model, sharded_tensor_parallel=True)
            cached = wrapper.state_dict()["weight"]
            assert wrapper.state_dict()["weight"] is cached
            if empty_dp and dp_rank == 1:
                assert parameter.to_local().numel() == 0
                assert cached.chunks == ()
                assert cached.dtype == parameter.dtype
                assert cached.device == parameter.device
            if packed and dp_dim == 1:
                assert len(cached.chunks) == 1
            expected_model = parameter.to_local().clone()
            paths = [
                tempfile.mkdtemp(prefix="tp-model-state-")
                if dist.get_rank() == 0
                else None
            ]
            dist.broadcast_object_list(paths)
            dcp.save({"model": wrapper}, checkpoint_id=paths[0])
            with torch.no_grad():
                model.weight.fill_(-37)
            dcp.load({"model": wrapper}, checkpoint_id=paths[0])
            torch.testing.assert_close(
                model.weight.to_local(), expected_model, atol=0, rtol=0
            )
            # Legacy/global sources remain readable by the sharded wrapper.
            with torch.no_grad():
                model.weight.fill_(-41)
            wrapper.load_state_dict({"weight": full})
            torch.testing.assert_close(
                model.weight.to_local(), expected_model, atol=0, rtol=0
            )
            assert wrapper.state_dict()["weight"].shape == full.shape
            assert (
                sum(value.numel() for _, value in wrapper.state_dict()["weight"].chunks)
                == parameter.to_local().numel()
            )
            hf_path = Path(paths[0]) / "hf"
            if dist.get_rank() == 0:
                hf_path.mkdir()
                save_file({"weight": full}, hf_path / "model.safetensors")
            dist.barrier()
            with torch.no_grad():
                model.weight.fill_(-43)
            destination = wrapper.state_dict()
            dcp.load(
                destination,
                storage_reader=IndexedHuggingFaceStorageReader(str(hf_path)),
            )
            wrapper.load_state_dict(destination)
            torch.testing.assert_close(
                model.weight.to_local(), expected_model, atol=0, rtol=0
            )
            for _, chunk in wrapper.state_dict()["weight"].chunks:
                assert torch._C._is_alias_of(model.weight._local_tensor, chunk)
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
            )
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_state = optimizer.state_dict()
            moment_key = "state.weight.exp_avg"
            expected_moments = [
                value.clone() for _, value in optimizer_state[moment_key].chunks
            ]
            optim_path = Path(paths[0]) / "optimizer"
            dcp.save({"optimizer": optimizer}, checkpoint_id=optim_path)
            for _, value in optimizer_state[moment_key].chunks:
                value.fill_(-47)
            dcp.load({"optimizer": optimizer}, checkpoint_id=optim_path)
            for (_, value), expected in zip(
                optimizer.state_dict()[moment_key].chunks, expected_moments, strict=True
            ):
                torch.testing.assert_close(value, expected, atol=0, rtol=0)
            resume_path = Path(paths[0]) / "model-optimizer-resume"
            dcp.save(
                {"model": wrapper, "optimizer": optimizer}, checkpoint_id=resume_path
            )
            model.weight.grad = torch.full_like(model.weight, 0.25)
            optimizer.step()
            optimizer.zero_grad()
            expected_weight = model.weight.to_local().clone()
            expected_state = {}
            for name, value in optimizer.state_dict().items():
                expected_state[name] = (
                    [chunk.clone() for _, chunk in value.chunks]
                    if hasattr(value, "chunks")
                    else value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
            dcp.load(
                {"model": wrapper, "optimizer": optimizer}, checkpoint_id=resume_path
            )
            model.weight.grad = torch.full_like(model.weight, 0.25)
            optimizer.step()
            optimizer.zero_grad()
            torch.testing.assert_close(
                model.weight.to_local(), expected_weight, atol=0, rtol=0
            )
            for name, value in optimizer.state_dict().items():
                expected = expected_state[name]
                if hasattr(value, "chunks"):
                    for (_, chunk), reference in zip(
                        value.chunks, expected, strict=True
                    ):
                        torch.testing.assert_close(chunk, reference, atol=0, rtol=0)
                elif isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
                else:
                    assert value == expected
            cold_model = nn.Module()
            cold_model.register_parameter(
                "weight", nn.Parameter(torch.zeros_like(parameter))
            )
            cold_model._tp_checkpoint_layouts = {"weight": (1, packed, tp)}
            cold_wrapper = ModelCheckpointState(
                cold_model, sharded_tensor_parallel=True
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
                model_parts=[cold_model],
                sharded_tensor_parallel=True,
            )
            assert all(not inner.state for inner in cold_optimizer.optimizers)
            dcp.load(
                {"model": cold_wrapper, "optimizer": cold_optimizer},
                checkpoint_id=resume_path,
            )
            cold_model.weight.grad = torch.full_like(cold_model.weight, 0.25)
            cold_optimizer.step()
            cold_optimizer.zero_grad()
            torch.testing.assert_close(
                cold_model.weight.to_local(), expected_weight, atol=0, rtol=0
            )
            for name, value in cold_optimizer.state_dict().items():
                expected = expected_state[name]
                if hasattr(value, "chunks"):
                    for (_, chunk), reference in zip(
                        value.chunks, expected, strict=True
                    ):
                        torch.testing.assert_close(chunk, reference, atol=0, rtol=0)
                elif isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
                else:
                    assert value == expected
            # Restore test weights for subsequent layout comparisons.
            wrapper.load_state_dict({"weight": full})
            for key in ("weight", "state.weight.exp_avg", "state.weight.ema_params"):
                sharded = shard_tensor_parallel_state(
                    {key: parameter.detach()}, model_parts=[model]
                )[key]
                assert sharded.shape == full.shape
                assert (
                    sum(value.numel() for _, value in sharded.chunks)
                    == parameter.to_local().numel()
                )
                for offset, value in sharded.chunks:
                    assert torch._C._is_alias_of(parameter._local_tensor, value)
                    expected = full[
                        tuple(
                            slice(start, start + size)
                            for start, size in zip(offset, value.shape, strict=True)
                        )
                    ]
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
                paths = [
                    tempfile.mkdtemp(prefix="tp-dp-layout-")
                    if dist.get_rank() == 0
                    else None
                ]
                dist.broadcast_object_list(paths)
                dcp.save({key: sharded}, checkpoint_id=paths[0])
                restored_full = torch.empty_like(full)
                dcp.load({key: restored_full}, checkpoint_id=paths[0])
                torch.testing.assert_close(restored_full, full, atol=0, rtol=0)
                target = torch.empty_like(parameter)
                restored_shards = restore_sharded_tensor_parallel_state(
                    {key: sharded}, local_state={key: target}, model_parts=[model]
                )[key]
                assert restored_shards is target
                assert restored_shards.placements == parameter.placements
                assert restored_shards.device_mesh == parameter.device_mesh
                torch.testing.assert_close(
                    restored_shards.to_local(), parameter.to_local(), atol=0, rtol=0
                )
                untouched = torch.full_like(parameter, -17)
                try:
                    restore_sharded_tensor_parallel_state(
                        {key: full},
                        local_state={key: untouched},
                        model_parts=[model],
                    )
                except ValueError as error:
                    assert "chunked source" in str(error)
                else:
                    raise AssertionError("Non-chunked source must be rejected")
                assert torch.all(untouched.to_local() == -17)
                # Only one TP owner has bad input; its peer must reject before copy.
                try:
                    restore_sharded_tensor_parallel_state(
                        {key: full if tp_rank == 0 else sharded},
                        local_state={key: untouched},
                        model_parts=[model],
                    )
                except (ValueError, RuntimeError) as error:
                    assert "chunked source" in str(error) or "another rank" in str(
                        error
                    )
                else:
                    raise AssertionError("Peer corruption must be coordinated")
                assert torch.all(untouched.to_local() == -17)
                saved = gather_tensor_parallel_state(
                    {key: parameter.detach().clone()}, model_parts=[model]
                )
                torch.testing.assert_close(saved[key], full, atol=0, rtol=0)
                restored = load_tensor_parallel_state(saved, model_parts=[model])[key]
                assert isinstance(restored, DTensor)
                assert restored.placements == parameter.placements
                assert restored.device_mesh == parameter.device_mesh
                torch.testing.assert_close(
                    restored.to_local(), parameter.to_local(), atol=0, rtol=0
                )
            print(
                f"rank {dist.get_rank()}: {packed=} {dp_dim=} TP2/DP2 exact",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
