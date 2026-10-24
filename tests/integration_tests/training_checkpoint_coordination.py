"""Two CPU/Gloo ranks: staging failure stops training and validation.

Run with torchrun --nproc_per_node=2. Uses the real train_step and optimizer;
checkpoint staging and batch preparation are substituted local operations.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.trainer.trainer import Trainer


def check_validation_staging():
    for failing_rank in (0, 1):
        for error_type in (OSError, KeyboardInterrupt, SystemExit):
            trainer = Trainer.__new__(Trainer)
            trainer.cfg = SimpleNamespace(validation=SimpleNamespace(dataset=None))
            trainer.device = torch.device("cpu")
            model = torch.nn.Linear(2, 2)
            trainer.model_parts = [model]
            failure = error_type("validation staging failed")

            def wait(*, failing_rank=failing_rank, failure=failure):
                if dist.get_rank() == failing_rank:
                    raise failure

            trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=wait)
            rng = torch.get_rng_state().clone()
            with patch.object(model, "eval") as evaluate:
                caught = None
                try:
                    trainer.validate(1)
                except BaseException as error:
                    caught = error
                assert caught is not None
                if dist.get_rank() == failing_rank:
                    assert caught is failure
                else:
                    assert isinstance(caught, RuntimeError)
                    assert str(caught) == (
                        "Validation checkpoint staging failed on another rank"
                    )
                evaluate.assert_not_called()
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            assert model.training
            dist.barrier()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        for failing_rank in (0, 1):
            for error_type in (OSError, KeyboardInterrupt, SystemExit):
                trainer = Trainer.__new__(Trainer)
                trainer.cfg = SimpleNamespace(gradient_accumulation_steps=1)
                trainer.model_parts = [torch.nn.Linear(2, 2)]
                trainer.optimizer = torch.optim.AdamW(
                    trainer.model_parts[0].parameters()
                )
                trainer.parallel_dims = None
                trainer.device = torch.device("cpu")
                trainer.lr_scheduler = SimpleNamespace(get_metrics=lambda: {})
                trainer.should_log = lambda: False
                trainer.microbatch = lambda batch: {
                    "num_valid_tokens": 2,
                    "num_routing_tokens": 2,
                }
                before = {
                    key: value.clone()
                    for key, value in trainer.model_parts[0].state_dict().items()
                }
                original = error_type("checkpoint staging failed")
                calls = []

                def wait(*, failing_rank=failing_rank, original=original):
                    if dist.get_rank() == failing_rank:
                        raise original

                def forward(*args, calls=calls, **kwargs):
                    calls.append(True)
                    raise AssertionError("Model executed after staging failure")

                trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=wait)
                trainer.forward_backward_step = forward
                caught = None
                try:
                    trainer.train_step(iter([None]))
                except BaseException as error:
                    caught = error
                assert caught is not None
                if dist.get_rank() == failing_rank:
                    assert caught is original
                else:
                    assert isinstance(caught, RuntimeError)
                    assert str(caught) == (
                        "Training checkpoint staging failed on another rank"
                    )
                assert not calls
                assert not trainer.optimizer.state
                torch.testing.assert_close(
                    trainer.model_parts[0].state_dict(), before, atol=0, rtol=0
                )
                dist.barrier()
        check_validation_staging()
        if dist.get_rank() == 0:
            print(
                "Staging failures stop training and validation before model execution."
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
