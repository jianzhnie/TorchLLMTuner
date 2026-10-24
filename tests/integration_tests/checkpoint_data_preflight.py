"""Two CPU/Gloo ranks reject incompatible data or schedule before live writes.

Run with torchrun --nproc_per_node=2. Covers DDP/native/custom FSDP, including
rank-local loader, scheduler or RNG compatibility mismatches and healthy peers.
Model topology and world size stay fixed; no topology migration is tested.
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

from llmtuner.components.checkpointer.checkpoint_keys import DATALOADER, RNG_STATE
from llmtuner.datasets.synthetic import RandomTokenDataLoader
from tests.integration_tests.rng_checkpoint_equivalence import manager, runtime, update


def loader(degree):
    return RandomTokenDataLoader(
        seed=42,
        vocab_size=16,
        batch_size=4,
        seq_len=8,
        dp_rank=dist.get_rank() if degree == 2 else 0,
        dp_world_size=degree,
    )


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            mesh = init_device_mesh("cpu", (2,))
            for backend in ("ddp", "native", "custom"):
                selected_mesh = None if backend == "ddp" else mesh
                source = runtime(42, selected_mesh, custom=backend == "custom")
                writer = manager(source, f"{paths[0]}/{backend}")
                try:
                    writer.states[DATALOADER] = loader(2)
                    next(iter(writer.states[DATALOADER]))
                    update(source)
                    update(source)
                    assert writer.save(2)
                finally:
                    writer.close()
                for phase, failing_rank in product(
                    ("dataloader", "lr_scheduler", "rng_world", "rng_topology"),
                    (None, 0, 1),
                ):
                    target = runtime(123, selected_mesh, custom=backend == "custom")
                    reader = manager(target, f"{paths[0]}/{backend}")
                    try:
                        failing = failing_rank is None or rank == failing_rank
                        degree = 1 if phase == "dataloader" and failing else 2
                        reader.states[DATALOADER] = loader(degree)
                        if phase == "lr_scheduler" and failing:
                            target[3].total_steps = 1
                        # Exercise real RNG compatibility validation without
                        # changing the actual process group or mocking the loader.
                        # These are incompatible destination records, not proof
                        # of a successful topology migration.
                        rng = reader.states[RNG_STATE]
                        if failing:
                            if phase == "rng_world":
                                rng.world_size = 3
                            elif phase == "rng_topology":
                                rng.parallel_degrees = {"tp": 2}
                        # Nonempty target AdamW/EMA state differs from checkpoint.
                        update(target)
                        before = [
                            deepcopy(state.state_dict())
                            for state in (target[0], target[2], target[3], target[4])
                        ]
                        data_before = deepcopy(reader.states[DATALOADER].state_dict())
                        python_before = random.getstate()
                        numpy_before = np.random.get_state()
                        cpu_before = torch.get_rng_state().clone()
                        try:
                            reader.load(2)
                        except ValueError as error:
                            assert failing
                            expected = {
                                "dataloader": "dp_world_size=2",
                                "lr_scheduler": "exceeds total_steps",
                                "rng_world": "world size changed",
                                "rng_topology": "parallel topology changed",
                            }[phase]
                            assert expected in str(error), error
                        except RuntimeError as error:
                            assert not failing
                            validation_phase = (
                                "RNG" if phase.startswith("rng_") else phase
                            )
                            assert str(error) == (
                                f"Checkpoint {validation_phase} preflight validation "
                                "failed on another rank"
                            )
                        else:
                            raise AssertionError(f"Incompatible {phase} accepted")
                        for state, expected in zip(
                            (target[0], target[2], target[3], target[4]),
                            before,
                            strict=True,
                        ):
                            torch.testing.assert_close(
                                state.state_dict(), expected, atol=0, rtol=0
                            )
                        assert reader.states[DATALOADER].state_dict() == data_before
                        assert random.getstate() == python_before
                        numpy_after = np.random.get_state()
                        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
                        assert numpy_after[0] == numpy_before[0]
                        assert numpy_after[2:] == numpy_before[2:]
                        assert torch.equal(torch.get_rng_state(), cpu_before)
                        dist.barrier()
                    finally:
                        reader.close()
                if rank == 0:
                    print(
                        f"{backend}: data/scheduler/RNG mismatch preserves live state",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
