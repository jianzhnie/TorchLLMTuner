"""Real DP2 Trainer coordinates local step setup and state update failures.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. Optimizer updates are real;
Only the failing local call is replaced; committed updates are not rolled back.
"""

from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.config.ema_config import EMAConfig
from llmtuner.trainer.cli import parse_config
from llmtuner.trainer.trainer import Trainer


def check_failure(failing_rank, error_type, phase):
    with TemporaryDirectory() as root:
        config = parse_config(
            [
                "--model_name_or_path",
                "qwen3",
                "--vocab_size",
                "32",
                "--hidden_size",
                "16",
                "--intermediate_size",
                "32",
                "--num_hidden_layers",
                "2",
                "--num_attention_heads",
                "2",
                "--num_key_value_heads",
                "2",
                "--max_seq_len",
                "8",
                "--global_batch_size",
                "2",
                "--steps",
                "2",
                "--total_steps",
                "2",
                "--seed",
                "42",
                "--dump_folder",
                root,
            ]
        )
        config.parallel.data_parallel_shard_size = 2
        config.training.ema_config = EMAConfig(decay=0.75)
        trainer = Trainer(config)
        try:
            trainer.step = 1
            trainer.should_log = lambda: True
            averages = [
                value["ema_params"]
                for optimizer in trainer.ema.optimizers
                for value in optimizer.state.values()
            ]
            before = [value.to_local().clone() for value in averages]
            preparing = phase in ("lr snapshot", "log decision")
            parameters = [p for part in trainer.model_parts for p in part.parameters()]
            original_parameters = [p.to_local().clone() for p in parameters]
            initial_steps = [
                int(state["step"])
                for optimizer in trainer.optimizer.optimizers
                for state in optimizer.state.values()
            ]
            failure = error_type(f"injected {phase} failure")
            owner, method = {
                "scheduler": (trainer.lr_scheduler, "step"),
                "EMA": (trainer.ema, "step"),
                "lr snapshot": (trainer.lr_scheduler, "get_metrics"),
                "log decision": (trainer, "should_log"),
            }[phase]
            original_step = getattr(owner, method)
            calls = []

            def step(*args):
                calls.append(phase)
                if dist.get_rank() == failing_rank:
                    raise failure
                return original_step(*args)

            caught = None
            with patch.object(owner, method, step):
                try:
                    trainer.train_step(trainer.batch_generator(trainer.data_iterator()))
                except BaseException as error:
                    caught = error
            if dist.get_rank() == failing_rank:
                assert caught is failure, repr(caught)
            else:
                assert isinstance(caught, RuntimeError), repr(caught)
                boundary = (
                    "Training step reporting setup"
                    if preparing
                    else f"Training {phase} update"
                )
                assert boundary in str(caught)
            assert calls == [phase]
            for value, expected in zip(averages, before, strict=True):
                if preparing or phase == "scheduler" or dist.get_rank() == failing_rank:
                    torch.testing.assert_close(
                        value.to_local(), expected, rtol=0, atol=0
                    )
            if phase == "EMA" and dist.get_rank() != failing_rank:
                assert any(
                    not torch.equal(value.to_local(), expected)
                    for value, expected in zip(averages, before, strict=True)
                )
            actual_steps = [
                int(state["step"])
                for optimizer in trainer.optimizer.optimizers
                for state in optimizer.state.values()
            ]
            if preparing:
                assert actual_steps == initial_steps
                for parameter, expected in zip(
                    parameters, original_parameters, strict=True
                ):
                    torch.testing.assert_close(
                        parameter.to_local(), expected, rtol=0, atol=0
                    )
            else:
                # Post-update failures terminate; they cannot roll back AdamW.
                assert actual_steps and all(step == 1 for step in actual_steps)
            assert trainer.lr_scheduler.state_dict()["last_epoch"] == (
                0
                if preparing
                or (phase == "scheduler" and dist.get_rank() == failing_rank)
                else 1
            )
            dist.barrier()
        finally:
            trainer.close()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        for rank in (0, 1):
            for error_type in (OSError, KeyboardInterrupt, SystemExit):
                for phase in ("lr snapshot", "log decision", "scheduler", "EMA"):
                    check_failure(rank, error_type, phase)
        if dist.get_rank() == 0:
            print("Step setup and update failures stop both ranks together.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
