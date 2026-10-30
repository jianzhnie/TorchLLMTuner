"""Actual 30B DP8: continuous 100 versus fresh-process 50 -> 100 recovery.

Run each phase with eight ranks, using the same captured baseline config:
continuous, initial, then resume. The latter two share a checkpoint directory.
Compare continuous/checkpoint/step-100 and resumed/checkpoint/step-100 with
compare_training_checkpoints.py after all three processes exit successfully.
This tests candidate recovery, not 100-step cross-framework numerical alignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.npu_trainer_alignment_config import (
    alignment_runtime_manifest,
    baseline_allow_hf32,
    build_candidate_30b_config,
)


def recovery_config(baseline, output: Path, phase: str, world: int):
    config = build_candidate_30b_config(baseline, output, world)
    # Both arms have the full horizon, including the interrupted process.
    config.training.steps = 100
    config.optimizer.lr_scheduler_config.total_steps = 100
    checkpoint = config.training.checkpoint_config
    checkpoint.load_only = False
    checkpoint.interval = 50
    checkpoint.last_save_model_only = False
    checkpoint.keep_latest_k = 0
    if phase == "resume":
        checkpoint.load_step = 50
        # Retain the HF initialization contract so assembly constructs on meta.
        # The existing step-50 DCP takes precedence over the initial HF fallback;
        # checked_load rejects any fallback that does not restore exactly 50.
    elif phase == "verify-continuous":
        checkpoint.load_step = 100
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("continuous", "initial", "resume", "verify-continuous")
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_config.read_text())
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Requires real NPU hardware")
    hf32 = baseline_allow_hf32(baseline)
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = hf32
    folder = args.output / (
        "continuous" if args.phase in ("continuous", "verify-continuous") else "resumed"
    )
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    config = recovery_config(baseline, folder, args.phase, world)
    expected_start = (
        100
        if args.phase == "verify-continuous"
        else 50
        if args.phase == "resume"
        else 0
    )
    stop = 50 if args.phase == "initial" else 100
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        trainer = Trainer(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    try:
        trainer.model_parts[0].model.config.router_aux_loss_coef = 0.0
        trainer.model_parts[0].model.config.output_router_logits = False
        load = trainer.checkpointer.load

        def checked_load(*positional, **keywords):
            loaded = load(*positional, **keywords)
            if not loaded or trainer.step != expected_start:
                raise AssertionError(
                    f"Expected loaded step {expected_start}, got {trainer.step}; "
                    f"loaded={loaded}"
                )
            return loaded

        trainer.checkpointer.load = checked_load
        checkpoint_components = sorted(trainer.checkpointer.states)
        ema_enabled = trainer.ema is not None
        # Only the interruption boundary changes, never the schedule horizon.
        trainer.should_continue_training = lambda: trainer.step < stop
        trainer.train()
        if trainer.step != stop:
            raise AssertionError(f"Expected final step {stop}, got {trainer.step}")
        record = {
            "phase": args.phase,
            "start": expected_start,
            "stop": trainer.step,
            "schedule_horizon": config.optimizer.lr_scheduler_config.total_steps,
            "ntokens_seen": trainer.ntokens_seen,
            "checkpoint_components": checkpoint_components,
            "ema_enabled": ema_enabled,
            "runtime": alignment_runtime_manifest(("llmtuner",)),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.phase}-rank{rank}.json").write_text(
            json.dumps(record, indent=2)
        )
        print(f"PASS phase={args.phase} rank={rank} stop={stop}", flush=True)
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
