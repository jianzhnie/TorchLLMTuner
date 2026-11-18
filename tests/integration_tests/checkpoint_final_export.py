"""Two CPU/NPU ranks: both backends export and reload actual FSDP model state.

Run with torchrun --standalone --nproc_per_node=2 -m
tests.integration_tests.checkpoint_final_export. Pass --backend both to exercise
native DCP and torch_checkpointing on CPU/NPU.
"""

import argparse
import importlib
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.backends.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)
from llmtuner.components.checkpointer.core.checkpoint_keys import MODEL
from llmtuner.config.checkpoint_config import CheckpointConfig
from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("dcp", "torch_checkpointing", "both"), default="dcp"
    )
    args = parser.parse_args()
    device = init_distributed(timeout=timedelta(seconds=90))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh(device.type, (2,))

        def build(manager_type, config, folder, seed):
            torch.manual_seed(seed)
            model = torch.nn.Linear(4, 4, device=device)
            model.register_buffer("counter", torch.tensor(7, device=device))
            fully_shard(model, mesh=mesh)
            assert isinstance(model.weight, DTensor)
            assert model.weight.to_local().numel() < model.weight.numel()
            optimizer = torch.optim.AdamW(
                model.parameters(), foreach=False, fused=False
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
            return manager_type(
                config,
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=scheduler,
                states={},
                folder=str(folder),
            )

        with TemporaryDirectory() if rank == 0 else nullcontext(None) as root:
            roots = [root]
            dist.broadcast_object_list(roots, src=0)
            for backend, manager_type in (
                ("dcp", DistributedCheckpointManager),
                ("torch_checkpointing", TorchCheckpointingManager),
            ):
                if args.backend not in (backend, "both"):
                    continue
                folder = Path(roots[0]) / backend
                config = CheckpointConfig(
                    backend=backend,
                    enable=True,
                    last_save_model_only=True,
                    export_dtype="float16",
                    keep_latest_k=0,
                )
                manager = build(manager_type, config, folder, 123)
                try:
                    reporting_logger = importlib.import_module(
                        manager_type.__module__
                    ).logger
                    original_info = reporting_logger.info
                    manager.last_save_model_only = False
                    for failing_rank in (0, 1):
                        failure = OSError("injected full final save reporting failure")

                        def fail_report(
                            message,
                            *args,
                            failing_rank=failing_rank,
                            failure=failure,
                            original_info=original_info,
                            **kwargs,
                        ):
                            if rank == failing_rank and str(message).startswith(
                                "Saving a full checkpoint"
                            ):
                                raise failure
                            return original_info(message, *args, **kwargs)

                        with patch.object(reporting_logger, "info", fail_report):
                            try:
                                manager.save(1, last_step=True)
                            except (OSError, RuntimeError) as error:
                                if rank == failing_rank:
                                    assert error is failure
                                else:
                                    assert "Checkpoint final state reporting" in str(
                                        error
                                    )
                            else:
                                raise AssertionError(
                                    "Final save reporting failure ignored"
                                )
                        dist.barrier()
                    manager.last_save_model_only = True
                    expected = {}
                    for key, value in manager.states[MODEL].state_dict().items():
                        local = (
                            value.to_local() if isinstance(value, DTensor) else value
                        )
                        expected[key] = (
                            local.half().to(local.dtype)
                            if local.is_floating_point()
                            else local.clone()
                        )
                    cast = manager._cast_export_state
                    for failing_rank in (0, 1):
                        failure = OSError("injected final export cast failure")

                        def fail_cast(
                            state, failing_rank=failing_rank, failure=failure, cast=cast
                        ):
                            if rank == failing_rank:
                                raise failure
                            return cast(state)

                        with patch.object(manager, "_cast_export_state", fail_cast):
                            try:
                                manager.save(1, last_step=True)
                            except (OSError, RuntimeError) as error:
                                if rank == failing_rank:
                                    assert error is failure
                                else:
                                    assert "Checkpoint final model preparation" in str(
                                        error
                                    )
                            else:
                                raise AssertionError("Final export failure was ignored")
                        dist.barrier()
                    assert manager.save(1, last_step=True)
                finally:
                    manager.close()
                loader = build(
                    manager_type,
                    CheckpointConfig(
                        enable=True,
                        load_only=True,
                        keep_latest_k=0,
                        initial_load_model_only=True,
                        initial_load_path=str(folder / "checkpoint" / "step-1"),
                    ),
                    folder / "fresh",
                    456,
                )
                try:
                    assert loader.load(-1)
                    for key, value in loader.states[MODEL].state_dict().items():
                        local = (
                            value.to_local() if isinstance(value, DTensor) else value
                        )
                        assert local.dtype == expected[key].dtype
                        torch.testing.assert_close(local, expected[key], atol=0, rtol=0)
                finally:
                    loader.close()
                dist.barrier()
                print(
                    f"rank={rank} {backend}: "
                    "final export, cast failures and reload passed"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
