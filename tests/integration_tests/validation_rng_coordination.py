"""Two CPU ranks coordinate RNG entry and exit failures around real validation.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. Trainer/FSDP/RNG are real;
RNG calls are faulted only during capture or after the real loader closes.
"""

import pickle
import random
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from llmtuner.config.validation_config import ValidationConfig
from llmtuner.trainer.cli import parse_config
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed


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
        config.training.validation_config = ValidationConfig(freq=1, steps=1)
        trainer = Trainer(config)
        try:
            modes = {
                module: module.training
                for part in trainer.model_parts
                for module in part.modules()
            }
            host_state = pickle.dumps((random.getstate(), np.random.get_state()))
            cpu_state = torch.get_rng_state().clone()
            failure = error_type("validation RNG snapshot failed")
            from llmtuner.trainer.runtime import validation as validation_runtime

            get_rng_state = torch.get_rng_state
            set_rng_state = torch.set_rng_state
            close_input = validation_runtime.close_validation_dataloader
            input_closed = False
            restore_calls = []

            def snapshot():
                if phase == "setup" and dist.get_rank() == failing_rank:
                    raise failure
                return get_rng_state()

            def restore(state):
                if input_closed:
                    restore_calls.append(state)
                if (
                    phase == "cleanup"
                    and input_closed
                    and dist.get_rank() == failing_rank
                ):
                    raise failure
                set_rng_state(state)

            def close(*args, **kwargs):
                nonlocal input_closed
                close_input(*args, **kwargs)
                input_closed = True

            with (
                patch.object(torch, "get_rng_state", snapshot),
                patch.object(torch, "set_rng_state", restore),
                patch.object(validation_runtime, "close_validation_dataloader", close),
            ):
                caught = None
                try:
                    trainer.validate(1)
                except BaseException as error:
                    caught = error
            if dist.get_rank() == failing_rank:
                assert caught is failure, repr(caught)
            else:
                assert isinstance(caught, RuntimeError), repr(caught)
                assert f"Validation RNG {phase}" in str(caught)
            assert all(module.training == mode for module, mode in modes.items())
            assert (
                pickle.dumps((random.getstate(), np.random.get_state())) == host_state
            )
            if phase == "setup" or dist.get_rank() != failing_rank:
                torch.testing.assert_close(
                    torch.get_rng_state(), cpu_state, rtol=0, atol=0
                )
            assert input_closed == (phase == "cleanup")
            assert len(restore_calls) == (1 if phase == "cleanup" else 0)
            # The failed RNG source is deliberately restored only for test teardown.
            set_rng_state(cpu_state)
            dist.barrier()
        finally:
            trainer.close()


def main():
    init_distributed()
    try:
        assert dist.get_world_size() == 2
        for rank in (0, 1):
            for error_type in (OSError, KeyboardInterrupt, SystemExit):
                for phase in ("cleanup", "setup"):
                    check_failure(rank, error_type, phase)
        if dist.get_rank() == 0:
            print("Validation RNG setup and cleanup failures stop both ranks.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
