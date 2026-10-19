"""Two-rank finite accumulation checkpoint, resumed in a separate torchrun.

Run full then resume with the same directory. Rank zero exhausts after three
microbatches; its DP peer has four. Only one two-microbatch update may commit.
"""

import os
import sys
from copy import deepcopy
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.pp_checkpoint_equivalence import (
    _compare_loaded_state,
    _runtime_state,
)


def main():
    phase, directory = sys.argv[1:]
    os.environ["LLMTUNER_TEST_RNG_STATE"] = "1"
    os.environ["LLMTUNER_TEST_LOADER_STATE"] = "1"
    cfg = LLMTunerConfig(
        model=ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(data_parallel_shard_size=2),
        training=TrainingConfig(
            global_batch_size=2,
            max_seq_len=8,
            gradient_accumulation_steps=2,
            steps=8,
            dump_folder=directory,
            ema_config=EMAConfig(decay=0.75),
            checkpoint_config=CheckpointConfig(
                enable=True, interval=100, last_save_model_only=False
            ),
        ),
    )
    trainer = Trainer(cfg)
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    path = Path(directory) / f"finite-rank-{rank}.pt"
    try:
        if phase == "full":
            loader = trainer.dataloader
            trainer.data_iterator = lambda: islice(iter(loader), 3 + rank)
            original_close = trainer.close

            def capture_and_close():
                if trainer.dataloader is not None:
                    torch.save(deepcopy(_runtime_state(trainer)), path)
                original_close()

            trainer.close = capture_and_close
            trainer.train()
            assert trainer.step == 1
            assert trainer.ntokens_seen == 24
            assert loader.state_dict()["steps"] == 3 + rank
            assert (Path(directory) / "checkpoint/step-1/.metadata").exists()
            assert not (Path(directory) / "checkpoint/step-2").exists()
        elif phase == "resume":
            expected = torch.load(path, weights_only=False)
            assert trainer.checkpointer.load(1)
            failures = []
            _compare_loaded_state(_runtime_state(trainer), expected, failures, "finite")
            assert not failures, failures
            print(f"rank={rank}: finite checkpoint complete state restored", flush=True)
        else:
            raise ValueError("Expected full or resume")
        dist.barrier()
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
