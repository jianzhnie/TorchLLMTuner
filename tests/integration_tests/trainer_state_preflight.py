"""Reject malformed Trainer counters before live DCP writes.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/trainer_state_preflight.py

CPU/Gloo DDP/native/custom FSDP; real Trainer state methods, manager and DCP.
The Trainer object is used only as a Stateful counter owner, not a full runtime.
Both ranks read the same deliberately malformed counter record.
"""

import random
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from itertools import product
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.checkpointer.checkpoint_keys import DATALOADER, TRAIN_STATE
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.checkpoint_data_preflight import loader
from tests.integration_tests.rng_checkpoint_equivalence import manager, runtime, update


def counters(step, tokens):
    trainer = Trainer.__new__(Trainer)
    trainer.step = step
    trainer.ntokens_seen = tokens
    return trainer


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        context = TemporaryDirectory() if dist.get_rank() == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            mesh = init_device_mesh("cpu", (2,))
            for backend, field, invalid in product(
                ("ddp", "native", "custom"),
                ("step", "ntokens_seen"),
                (-1, True, 1.5),
            ):
                selected_mesh = None if backend == "ddp" else mesh
                folder = f"{paths[0]}/{backend}_{field}_{invalid}"
                source = runtime(42, selected_mesh, custom=backend == "custom")
                writer = manager(source, folder)
                try:
                    saved = counters(2, 128)
                    setattr(saved, field, invalid)
                    writer.states[TRAIN_STATE] = saved
                    writer.states[DATALOADER] = loader(2)
                    next(iter(writer.states[DATALOADER]))
                    update(source)
                    update(source)
                    # Deliberate malformed fixture; do not replace the reader's
                    # validation or DCP callbacks with injected exceptions.
                    assert writer.save(2)
                finally:
                    writer.close()
                target = runtime(123, selected_mesh, custom=backend == "custom")
                reader = manager(target, folder)
                try:
                    reader.states[TRAIN_STATE] = counters(1, 64)
                    reader.states[DATALOADER] = loader(2)
                    update(target)
                    tracked = (
                        target[0],
                        target[2],
                        target[3],
                        target[4],
                        reader.states[DATALOADER],
                        reader.states[TRAIN_STATE],
                    )
                    before = [deepcopy(state.state_dict()) for state in tracked]
                    python_before = random.getstate()
                    numpy_before = np.random.get_state()
                    cpu_before = torch.get_rng_state().clone()
                    try:
                        reader.load(2)
                    except ValueError as error:
                        assert str(error) == (
                            f"trainer {field} must be a nonnegative integer"
                        )
                    else:
                        raise AssertionError(f"Accepted {field}={invalid!r}")
                    for state, expected in zip(tracked, before, strict=True):
                        torch.testing.assert_close(
                            state.state_dict(), expected, rtol=0, atol=0
                        )
                    assert random.getstate() == python_before
                    np.testing.assert_equal(np.random.get_state(), numpy_before)
                    assert torch.equal(torch.get_rng_state(), cpu_before)
                    dist.barrier()
                finally:
                    reader.close()
                if dist.get_rank() == 0:
                    print(f"PASS {backend} {field}={invalid!r}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
