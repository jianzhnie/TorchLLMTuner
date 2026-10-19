"""Two Gloo ranks exercise Trainer.train reporting failure before checkpointing.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/training_metric_coordination.py

Step computation and services are substituted; the real train loop must stop
both ranks before checkpoint save after a rank-local sink failure.
"""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

from llmtuner.trainer.trainer import Trainer


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        assert dist.get_world_size() == 2
        for failing_rank in (0, 1):
            for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                original = error_type("training metric sink failed")
                saves = []
                closes = []
                trainer = Trainer.__new__(Trainer)
                trainer.cfg = SimpleNamespace(
                    checkpoint=SimpleNamespace(load_step=-1),
                    profiler=None,
                    dump_folder="unused",
                    gc_freq=1,
                    steps=1,
                )
                trainer.step = 0
                trainer.ntokens_seen = 0
                trainer.parallel_dims = None
                trainer.data_iterator = lambda: iter(())
                trainer.batch_generator = lambda source: source
                trainer.train_step = lambda source: {
                    "loss": 1.0,
                    "max_loss": 1.0,
                    "grad_norm": 0.5,
                }
                trainer.should_validate = lambda step: False
                trainer.close = lambda closes=closes: closes.append(True)
                trainer.checkpointer = SimpleNamespace(
                    load=lambda step: False,
                    save=lambda *args, saves=saves, **kwargs: saves.append(args),
                )

                def log(*args, failing_rank=failing_rank, original=original, **kwargs):
                    if dist.get_rank() == failing_rank:
                        raise original

                trainer.metrics = SimpleNamespace(log=log)
                with (
                    patch(
                        "llmtuner.trainer.trainer.Profiler",
                        return_value=nullcontext(SimpleNamespace(step=lambda: None)),
                    ),
                    patch(
                        "llmtuner.trainer.trainer.GarbageCollection",
                        return_value=SimpleNamespace(run=lambda step: None),
                    ),
                ):
                    caught = None
                    try:
                        trainer.train()
                    except BaseException as error:
                        caught = error
                assert caught is not None, "peer continued after sink failure"
                if dist.get_rank() == failing_rank:
                    assert caught is original
                else:
                    assert isinstance(caught, RuntimeError)
                    assert "Training metric reporting" in str(caught)
                assert saves == [], "checkpoint entered before reporting agreed"
                assert closes == [True]
                assert trainer.step == 1  # The optimizer step already completed.
                dist.barrier()
        if dist.get_rank() == 0:
            print("Trainer reporting failures coordinated before checkpoint save.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
