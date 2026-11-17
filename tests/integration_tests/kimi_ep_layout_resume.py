"""Resume a real Kimi Trainer across EP layout changes with RNG excluded.

Run full then resume WORKDIR in separate two-rank torchrun invocations,
changing LLMTUNER_TEST_EP_SIZE between phases. This checks full weights,
optimizer/EMA, trainer/scheduler before update. RNG topology rejection remains
active; explicit RNG exclusion forfeits exact random stream continuity.
"""

import json
import os
import sys

import torch
import torch.distributed as dist

from llmtuner.models.kimi.kimi_k3.checkpoint import (
    convert_kimi_expert_state,
    convert_kimi_optimizer_state,
)
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.checkpoint_scenarios import (
    EXTRA_STEPS,
    SPLIT_STEP,
    kimi_config,
    run_phase,
    train_steps,
)
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.trainer_ema_process_resume import snapshot


def run(workdir):
    cfg = kimi_config(workdir)
    cfg.checkpoint.exclude_from_loading = ["rng_state"]
    trainer = Trainer(cfg)
    try:
        if trainer.world_size != 2:
            raise ValueError(
                "Kimi cross-layout snapshot reference currently requires two ranks"
            )
        assert trainer.checkpointer.load(-1)
        expected = torch.load(
            os.path.join(workdir, f"split-state-rank-{trainer.rank}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        target_grouped = cfg.parallel.ep_size > 1
        # The current checkpoint snapshot already contains full EP/FSDP state.
        source_model = expected["model"]
        expected_model = convert_kimi_expert_state(
            source_model,
            num_experts=4,
            to_hf=not target_grouped,
        )
        expected_optim = convert_kimi_optimizer_state(
            expected["optimizer"],
            num_experts=4,
            to_hf=not target_grouped,
        )
        expected_ema = convert_kimi_optimizer_state(
            expected["ema"],
            num_experts=4,
            to_hf=not target_grouped,
        )
        loaded = snapshot(trainer)
        for label, actual, wanted in (
            (
                "model",
                loaded["model"],
                expected_model,
            ),
            ("optimizer", loaded["optimizer"], expected_optim),
            ("ema", loaded["ema"], expected_ema),
            ("trainer", trainer.progress.state_dict(), expected["train_state"]),
            ("scheduler", trainer.lr_scheduler.state_dict(), expected["lr_scheduler"]),
        ):
            assert_state_equal(actual, wanted, label)
        assert_state_equal(
            trainer.dataloader.state_dict(), expected["dataloader"], "dataloader"
        )
        losses = train_steps(trainer, trainer.data_iterator(), EXTRA_STEPS)
        with open(os.path.join(workdir, f"losses-rank-{trainer.rank}.json")) as stream:
            reference = json.load(stream)[SPLIT_STEP:]
        torch.testing.assert_close(
            torch.tensor(losses), torch.tensor(reference), atol=1e-6, rtol=1e-6
        )
        print(
            f"rank {trainer.rank}: exact pre-update state, "
            "later loss passed; RNG excluded",
            flush=True,
        )
    finally:
        try:
            trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1] == "full":
        run_phase(kimi_config)
    elif sys.argv[1] == "resume":
        run(sys.argv[2])
    else:
        raise ValueError("Expected full|resume WORKDIR")
