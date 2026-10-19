"""Resume a real Kimi Trainer across EP layout changes with RNG excluded.

The source is pp_checkpoint_equivalence.py full. Run this script instead of its
resume phase with a different LLMTUNER_TEST_EP_SIZE. This checks full weights,
optimizer/EMA, trainer/scheduler before update. RNG topology rejection remains
active; explicit RNG exclusion forfeits exact random stream continuity.
"""

import json
import os
import sys

import torch
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.models.kimi.checkpoint import (
    convert_kimi_expert_state,
    convert_kimi_optimizer_state,
)
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.pp_checkpoint_equivalence import (
    EXTRA_STEPS,
    SPLIT_STEP,
    _cfg,
    _compare_loaded_state,
    _train_steps,
)


def full(values):
    return {
        key: value.full_tensor().cpu() if isinstance(value, DTensor) else value
        for key, value in values.items()
    }


def run(workdir):
    cfg = _cfg(workdir)
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
            weights_only=True,
            map_location="cpu",
        )
        target_grouped = cfg.parallel.ep_size > 1
        source_model = full(expected["model"])
        # The independent runtime snapshot stores each source EP rank's
        # local experts; unlike optimizer/EMA snapshots it is not gathered.
        if not target_grouped:
            peers = [
                torch.load(
                    os.path.join(workdir, f"split-state-rank-{rank}.pt"),
                    weights_only=True,
                    map_location="cpu",
                )["model"]
                for rank in range(trainer.world_size)
            ]
            peer_models = [full(peer) for peer in peers]
            for key in source_model:
                if ".routed_experts.inner_experts." in key:
                    source_model[key] = torch.cat([peer[key] for peer in peer_models])
        expected_model = convert_kimi_expert_state(
            source_model,
            num_experts=4,
            to_hf=not target_grouped,
        )
        expected_optim = convert_kimi_optimizer_state(
            full(expected["optimizer"]),
            num_experts=4,
            to_hf=not target_grouped,
        )
        expected_ema = convert_kimi_optimizer_state(
            full(expected["ema"]),
            num_experts=4,
            to_hf=not target_grouped,
        )
        failures = []
        for label, actual, wanted in (
            (
                "model",
                full(ModelCheckpointState(trainer.model_parts).state_dict()),
                expected_model,
            ),
            ("optimizer", full(trainer.optimizer.state_dict()), expected_optim),
            ("ema", full(trainer.ema.state_dict()), expected_ema),
            ("trainer", trainer.state_dict(), expected["trainer"]),
            ("scheduler", trainer.lr_scheduler.state_dict(), expected["scheduler"]),
        ):
            _compare_loaded_state(actual, wanted, failures, label)
        if os.environ.get("LLMTUNER_TEST_LOADER_STATE") == "1":
            _compare_loaded_state(
                trainer.dataloader.state_dict(),
                expected["dataloader"],
                failures,
                "dataloader",
            )
        assert not failures, failures
        losses = _train_steps(trainer, trainer.data_iterator(), EXTRA_STEPS)
        with open(os.path.join(workdir, "losses_full.json")) as stream:
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
        trainer.close()


if __name__ == "__main__":
    run(sys.argv[1])
