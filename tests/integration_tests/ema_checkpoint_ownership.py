"""Two CPU ranks: manager restore and cold start with an empty EMA owner.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. Models are replicated and
updated identically; EMA deliberately owns no model parts on one rank. This
exercises manager lifecycle, not a pipeline schedule or target-device kernels.
"""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal


def runtime(seed, empty_rank):
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 2)
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=optimizer.optimizers, training_steps=4
    )
    ema = EMA(model_parts=[] if dist.get_rank() == empty_rank else [model], decay=0.5)
    return model, optimizer, scheduler, ema


def update(current):
    model, optimizer, scheduler, ema = current
    optimizer.zero_grad()
    model(torch.tensor([[0.2, -0.4], [0.6, 0.8]])).square().sum().backward()
    optimizer.step()
    scheduler.step()
    ema.step(scheduler.state_dict()["last_epoch"])


def snapshot(current):
    return [deepcopy(component.state_dict()) for component in current]


def manager(current, root, mode, *, exclude_ema=False):
    model, optimizer, scheduler, ema = current
    return DistributedCheckpointManager(
        CheckpointConfig(
            enable=True,
            interval=1,
            keep_latest_k=0,
            async_mode=mode,
            exclude_from_loading=["ema"] if exclude_ema else [],
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ema=ema,
        states={},
        folder=str(root),
    )


def check_case(empty_rank, mode, exclude_ema):
    folders = [mkdtemp(prefix="llmtuner-ema-owner-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(folders, src=0)
    root = Path(folders[0])
    original = runtime(11, empty_rank)
    initial_ema = deepcopy(original[3].state_dict())
    update(original)
    saved = snapshot(original)
    assert bool(saved[3]) == (dist.get_rank() != empty_rank)
    for name, average in saved[3].items():
        fqn = name.removeprefix("state.").removesuffix(".ema_params")
        assert_state_equal(average, initial_ema[name].lerp(saved[0][fqn], 0.5))
    writer = manager(original, root, mode)
    try:
        assert writer.save(curr_step=1)
        if mode == "async":
            assert writer.save_future is not None
        writer.maybe_wait_for_staging()
        update(original)
        continuous = snapshot(original)
    finally:
        writer.close()
    restored = runtime(99, empty_rank)
    assert any(
        not torch.equal(value, saved[0][key])
        for key, value in restored[0].state_dict().items()
    )
    reader = manager(restored, root, mode, exclude_ema=exclude_ema)
    try:
        assert reader.load(1)
        boundary = snapshot(restored)
        assert_state_equal(boundary[:3], saved[:3])
        expected_ema = (
            {
                name: saved[0][name.removeprefix("state.").removesuffix(".ema_params")]
                for name in saved[3]
            }
            if exclude_ema
            else saved[3]
        )
        assert_state_equal(boundary[3], expected_ema)
        update(restored)
        final = snapshot(restored)
        assert_state_equal(final[:3], continuous[:3])
        for name, average in final[3].items():
            fqn = name.removeprefix("state.").removesuffix(".ema_params")
            assert_state_equal(average, expected_ema[name].lerp(final[0][fqn], 0.5))
        if not exclude_ema:
            assert_state_equal(final[3], continuous[3])
    finally:
        reader.close()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"empty_rank={empty_rank} mode={mode} exclude_ema={exclude_ema}: {root}")


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    try:
        assert dist.get_world_size() == 2
        for empty_rank in (0, 1):
            for mode in ("disabled", "async"):
                for exclude_ema in (False, True):
                    check_case(empty_rank, mode, exclude_ema)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
