"""Real asynchronous DCP publication failure, cleanup and retry on two ranks.

LLMTUNER_TEST_DEVICE=npu PYTHONPATH=. torchrun --standalone --nproc_per_node=2
tests/integration_tests/checkpoint_publication_failure.py --async-mode async
An actual directory blocks metadata replacement; no writer/collective is mocked.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import LRSchedulerConfig, OptimizerConfig
from tests.integration_tests.async_checkpoint_lifecycle_equivalence import (
    assert_group_released,
    manager_for,
    step,
)
from tests.integration_tests.harness import init_distributed, moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--async-mode", choices=("async", "async_with_pinned_mem"), default="async"
    )
    args = parser.parse_args()
    device = init_distributed()
    active = []
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("dp_shard",))

        def build():
            model = moved(torch.nn.Linear(3, 2))
            fully_shard(model, mesh=mesh)
            optimizer = OptimizersContainer(
                OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
                model_parts=[model],
            )
            scheduler = build_lr_scheduler(
                LRSchedulerConfig(decay_ratio=1.0),
                optimizers=optimizer.optimizers,
                training_steps=8,
            )
            return model, optimizer, scheduler

        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            folder = paths[0]
            torch.manual_seed(271)
            model, optimizer, scheduler = build()
            manager = manager_for(
                model, optimizer, scheduler, folder, async_mode=args.async_mode
            )
            active.append(manager)
            step(model, optimizer, scheduler)
            assert manager.save(1)
            manager.maybe_wait_for_staging()
            manager.maybe_wait_for_saving()
            previous = Path(folder) / "checkpoint/step-1/.metadata"
            assert previous.is_file()
            previous_bytes = previous.read_bytes()
            step(model, optimizer, scheduler)
            for inner in optimizer.optimizers:
                assert inner.state, "publication test has no optimizer state"
                for state in inner.state.values():
                    assert state["step"].item() == 2
                    for key in ("exp_avg", "exp_avg_sq"):
                        moment = state[key]
                        local = (
                            moment.to_local() if isinstance(moment, DTensor) else moment
                        )
                        assert local.numel() > 0, key
                        assert torch.isfinite(local).all().item(), key
                        assert local.count_nonzero().item() > 0, key
            before_failure = deepcopy(
                (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            )
            blocker = Path(folder) / "checkpoint/step-2/.metadata"
            if rank == 0:
                blocker.mkdir(parents=True)
            dist.barrier()
            group = manager.pg
            try:
                manager.save(2)
                manager.close()
            except BaseException as error:
                assert "IsADirectoryError" in str(error), str(error)
            else:
                raise AssertionError("metadata publication unexpectedly succeeded")
            manager.close()
            assert_group_released(group)
            assert manager.save_future is None
            assert previous.read_bytes() == previous_bytes
            assert blocker.is_dir()
            torch.testing.assert_close(
                model.state_dict(), before_failure[0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state_dict(), before_failure[1], rtol=0, atol=0
            )
            assert scheduler.state_dict() == before_failure[2]
            dist.barrier()
            if rank == 0:
                blocker.rmdir()
            dist.barrier()

            retry = manager_for(
                model, optimizer, scheduler, folder, async_mode=args.async_mode
            )
            active.append(retry)
            assert retry.save(2)
            retry.close()
            assert (blocker.parent / ".metadata").is_file()
            restored_model, restored_optimizer, restored_scheduler = build()
            restored = manager_for(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                folder,
                async_mode=args.async_mode,
            )
            active.append(restored)
            assert all(not inner.state for inner in restored_optimizer.optimizers)
            assert restored.load(2)
            for resumed_step in (False, True):
                if resumed_step:
                    step(model, optimizer, scheduler)
                    step(restored_model, restored_optimizer, restored_scheduler)
                torch.testing.assert_close(
                    restored_model.state_dict(), model.state_dict(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    restored_optimizer.state_dict(),
                    optimizer.state_dict(),
                    rtol=0,
                    atol=0,
                )
                assert restored_scheduler.state_dict() == scheduler.state_dict()
            restored.close()
            dist.barrier()
            print(
                f"PASS publication failure/cleanup/retry/resume "
                f"{args.async_mode} rank={rank}"
            )
    finally:
        for manager in active:
            manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
