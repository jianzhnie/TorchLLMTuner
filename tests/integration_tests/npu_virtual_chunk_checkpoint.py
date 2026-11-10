"""Real NPU virtual-moment chunk IO; auxiliary, not target-model acceptance.

Run with torchrun, a fresh output directory and optionally a DCP async mode.
Nonzero, rank-dependent moments must survive DCP exactly.
"""

import faulthandler
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import (
    BlockingAsyncStager,
    DefaultStager,
    StagingOptions,
)
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncSaveResponse,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointAsyncMode,
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.core.chunked_tensor import ChunkedCheckpointTensor
from llmtuner.components.checkpointer.storage.checkpoint_writer import (
    GenerationFileSystemWriter,
)
from llmtuner.components.checkpointer.storage.virtual_optimizer_planner import (
    VirtualOptimizerLoadPlanner,
    VirtualOptimizerSavePlanner,
    VirtualOptimizerStager,
)
from llmtuner.components.optimizer.state_dict import init_optim_state
from llmtuner.components.optimizer.state_residency import install_state_residency
from tests.caps import require_env
from tests.integration_tests.harness import init_distributed


class FailingBlockingStager(BlockingAsyncStager):
    """Inject a rank-local failure after the real CPU staging copy."""

    def stage(self, state_dict):
        snapshot = super().stage(state_dict)
        if dist.get_rank() == 0:
            raise OSError("injected virtual CPU staging failure")
        return snapshot


def main() -> None:
    require_env("torch_npu")
    faulthandler.dump_traceback_later(60, repeat=True)
    device = init_distributed(timeout=timedelta(seconds=120))
    try:
        assert device.type == "npu"
        rank, world = dist.get_rank(), dist.get_world_size()
        root = Path(sys.argv[1])
        mode = sys.argv[2] if len(sys.argv) > 2 else "disabled"
        assert mode in ("disabled", "async", "async_with_pinned_mem")
        layout = sys.argv[3] if len(sys.argv) > 3 else "chunk"
        assert layout in ("chunk", "dtensor", "empty_chunk")
        assert layout != "empty_chunk" or world > 1
        host_group = dist.new_group(backend="gloo")
        mesh = init_device_mesh("npu", (world,)) if layout == "dtensor" else None

        def checkpoint_view(value):
            if mesh is not None:
                return DTensor.from_local(
                    value,
                    mesh,
                    [Shard(0)],
                    shape=torch.Size((world * 2, 3)),
                    stride=(3, 1),
                )
            return ChunkedCheckpointTensor(
                (world - int(layout == "empty_chunk"), 2, 3),
                [((rank, 0, 0), value.unsqueeze(0))] if value.numel() else [],
                dtype=value.dtype,
                device=device,
            )

        if rank == 0:
            root.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        for dtype in (torch.bfloat16, torch.float32):
            parameter = torch.nn.Parameter(
                torch.ones(
                    (0 if layout == "empty_chunk" and rank == world - 1 else 2, 3),
                    dtype=dtype,
                    device=device,
                )
            )
            optimizer = torch.optim.AdamW(
                [parameter], lr=0.001, foreach=False, fused=False
            )
            install_state_residency(optimizer, "npu_virtual")
            for step in range(50):
                parameter.grad = torch.full_like(
                    parameter, rank + 1.25 + (step % 3) * 0.125
                )
                optimizer.step()
            moments = {
                name: optimizer.state[parameter][name]
                for name in ("exp_avg", "exp_avg_sq")
            }
            expected = {name: value.clone().cpu() for name, value in moments.items()}
            assert all(
                value.abs().min() > 0 for value in expected.values() if value.numel()
            )
            chunks = {name: checkpoint_view(value) for name, value in moments.items()}
            state = {
                "optimizer": {
                    "state": {
                        "weight": {**chunks, "step": optimizer.state[parameter]["step"]}
                    }
                }
            }
            directory = root / str(dtype)
            writer = GenerationFileSystemWriter(directory, per_thread_copy_ahead=0)
            if mode == "disabled":
                dcp.save(
                    state,
                    storage_writer=writer,
                    planner=VirtualOptimizerSavePlanner(),
                    process_group=host_group,
                )
                for value in moments.values():
                    value.zero_()
            else:
                manager = DistributedCheckpointManager.__new__(
                    DistributedCheckpointManager
                )
                manager.pg = host_group
                manager.staging_pg = dist.new_group(backend="gloo")
                manager.states = {
                    "optimizer": SimpleNamespace(state_residency="npu_virtual")
                }
                manager.stager = (
                    DefaultStager(
                        StagingOptions(
                            use_pinned_memory=True,
                            use_shared_memory=True,
                            use_async_staging=True,
                            use_non_blocking_copy=True,
                        )
                    )
                    if mode == "async_with_pinned_mem"
                    else None
                )
                try:
                    result = manager._save_checkpoint(
                        state,
                        str(directory),
                        async_mode=DistributedCheckpointAsyncMode(mode),
                    )
                    if isinstance(result, AsyncSaveResponse):
                        result.staging_completion.result()
                    # Mutate live moments before joining disk upload: saving
                    # must observe the completed snapshot, not these writes.
                    for value in moments.values():
                        value.zero_()
                    if isinstance(result, AsyncSaveResponse):
                        result.upload_completion.result()
                    else:
                        result.result()
                finally:
                    if manager.stager is not None:
                        manager.stager.close()
            dcp.load(
                state, checkpoint_id=directory, planner=VirtualOptimizerLoadPlanner()
            )
            for name, value in moments.items():
                torch.testing.assert_close(
                    value.clone().cpu(), expected[name], rtol=0, atol=0
                )

            # Fresh allocations plus native optimizer load hooks must preserve
            # the restored moments and bias-correction counter for update 51.
            fresh_parameter = torch.nn.Parameter(parameter.detach().clone())
            fresh = torch.optim.AdamW(
                [fresh_parameter], lr=0.001, foreach=False, fused=False
            )
            install_state_residency(fresh, "npu_virtual")
            init_optim_state(fresh)
            fresh_chunks = {
                name: checkpoint_view(fresh.state[fresh_parameter][name])
                for name in moments
            }
            fresh_state = {
                "optimizer": {
                    "state": {
                        "weight": {
                            **fresh_chunks,
                            "step": fresh.state[fresh_parameter]["step"],
                        }
                    }
                }
            }
            dcp.load(
                fresh_state,
                checkpoint_id=directory,
                planner=VirtualOptimizerLoadPlanner(),
            )
            fresh.load_state_dict(fresh.state_dict())
            assert fresh.state[fresh_parameter]["step"].item() == 50
            parameter.grad = torch.full_like(parameter, rank + 1.375)
            fresh_parameter.grad = parameter.grad.clone()
            optimizer.step()
            fresh.step()
            torch.testing.assert_close(fresh_parameter, parameter, rtol=0, atol=0)
            for name in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    fresh.state[fresh_parameter][name].clone().cpu(),
                    optimizer.state[parameter][name].clone().cpu(),
                    rtol=0,
                    atol=0,
                )
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS exact nonzero virtual chunk IO: "
                    f"{mode}, {layout}, {dtype}, {world} ranks"
                )
        if mode == "async":
            failed_directory = root / "injected_failure"
            stager = VirtualOptimizerStager(FailingBlockingStager(), host_group)
            try:
                try:
                    dcp.async_save(
                        state,
                        checkpoint_id=failed_directory,
                        process_group=host_group,
                        async_stager=stager,
                    )
                except (OSError, RuntimeError) as error:
                    expected_error = (
                        "injected virtual CPU staging failure"
                        if rank == 0
                        else "Virtual optimizer snapshot failed on another rank"
                    )
                    assert expected_error in str(error), str(error)
                else:
                    raise AssertionError("Failed staging started checkpoint upload")
                dist.barrier(group=host_group)
                assert not (failed_directory / ".metadata").exists()
                if rank == 0:
                    print(
                        "PASS rank-local CPU staging failure coordinated before upload"
                    )
            finally:
                stager.close()
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
