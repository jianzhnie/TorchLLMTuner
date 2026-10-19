"""DCP coordinates GC failures and retains live asynchronous save ownership.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/checkpoint_gc_coordination.py

CPU/Gloo; inject sync post-save GC or async pre/post-launch GC.
The actual manager, DCP worker, save Future and subsequent load remain intact.
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from itertools import product
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.utils.garbage_collection import GarbageCollection
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        with TemporaryDirectory() if rank == 0 else nullcontext(None) as root:
            roots = [root]
            dist.broadcast_object_list(roots, src=0)
            for (mode, gc_call), failing_rank in product(
                (("disabled", 1), ("async", 1), ("async", 2)), (0, 1)
            ):
                for error_type in (OSError, KeyboardInterrupt, SystemExit):
                    torch.manual_seed(123)
                    model = torch.nn.Linear(2, 2)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda _: 1
                    )
                    model(torch.ones(2, 2)).square().sum().backward()
                    optimizer.step()
                    scheduler.step()
                    states = (model, optimizer, scheduler)
                    before = deepcopy([state.state_dict() for state in states])
                    manager = DistributedCheckpointManager(
                        CheckpointConfig(
                            enable=True, interval=1, keep_latest_k=0, async_mode=mode
                        ),
                        model_parts=[model],
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        states={},
                        folder=f"{roots[0]}/{mode}-{gc_call}-{failing_rank}-{error_type.__name__}",
                    )
                    failure = error_type("post-launch GC failure")
                    original = GarbageCollection.collect
                    calls = 0

                    def collect(
                        reason,
                        *,
                        failing_rank=failing_rank,
                        failure=failure,
                        original=original,
                        gc_call=gc_call,
                        **kwargs,
                    ):
                        nonlocal calls
                        calls += 1
                        if rank == failing_rank and calls == gc_call:
                            raise failure
                        return original(reason, **kwargs)

                    try:
                        caught = None
                        with patch.object(
                            GarbageCollection, "collect", staticmethod(collect)
                        ):
                            try:
                                manager.save(1)
                            except BaseException as error:
                                caught = error
                        # Coordinate the observation even on the broken baseline,
                        # rather than leaving one rank to enter a later load.
                        observed = [None, None]
                        dist.all_gather_object(
                            observed,
                            (manager.save_future is not None, type(caught).__name__),
                        )
                        expected_future = mode == "async" and gc_call == 2
                        assert all(item[0] == expected_future for item in observed), (
                            observed
                        )
                        if rank == failing_rank:
                            assert caught is failure
                        else:
                            assert isinstance(caught, RuntimeError), caught
                            assert str(caught) == (
                                "Checkpoint garbage collection failed on another rank"
                            )
                        manager.maybe_wait_for_saving()
                        assert manager.save_future is None
                        assert_state_equal(
                            [state.state_dict() for state in states], before
                        )
                        if mode == "async" and gc_call == 1:
                            # GC failed before launch: publish a healthy recovery
                            # point before testing its actual read.
                            assert manager.save(1)
                            manager.maybe_wait_for_saving()
                        with torch.no_grad():
                            model.weight.add_(5)
                        assert manager.load(1)
                        assert_state_equal(
                            [state.state_dict() for state in states], before
                        )
                        assert manager.save(2)
                        manager.maybe_wait_for_saving()
                        dist.barrier()
                    finally:
                        manager.close()
                    if rank == 0:
                        print(
                            f"PASS {mode=} {gc_call=} {failing_rank=} "
                            f"{error_type.__name__}",
                            flush=True,
                        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
