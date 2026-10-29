"""Real optional checkpoint backend resume against uninterrupted NPU training.

PYTHONPATH=. python tests/integration_tests/npu_checkpoint_backend_equivalence.py
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from examples.train_qwen3 import qwen3_config
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.trainer.trainer import Trainer
from tests.caps import require_env


def _assert_equal(actual, expected) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, tuple | list):
        assert len(actual) == len(expected)
        for observed, reference in zip(actual, expected, strict=True):
            _assert_equal(observed, reference)
    elif isinstance(expected, torch.Tensor):
        if isinstance(expected, DTensor):
            assert isinstance(actual, DTensor)
            assert actual.placements == expected.placements
            assert actual.shape == expected.shape
            actual, expected = actual.to_local(), expected.to_local()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        assert actual == expected, (actual, expected)


def _train(root: Path, steps: int, *, ema: str, ac: str) -> dict:
    config = qwen3_config()
    config.training.steps = steps
    config.training.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))
    config.training.dump_folder = str(root)
    config.training.activation_checkpoint_mode = ac
    config.optimizer.lr_scheduler_config = LRSchedulerConfig(total_steps=4)
    if ema != "none":
        config.training.ema_config = EMAConfig(
            decay=0.5 if ema == "fixed" else None,
            **({"half_life_fraction": 0.5} if ema == "dynamic" else {}),
        )
    config.training.checkpoint_config = CheckpointConfig(
        enable=True,
        backend="torch_checkpointing",
        interval=2,
        last_save_model_only=False,
        keep_latest_k=0,
    )
    trainer = Trainer(config)
    manager = trainer.checkpointer
    try:
        assert type(manager).__name__ == "TorchCheckpointingManager"
        if ac != "none":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointWrapper,
            )

            assert any(
                isinstance(module, CheckpointWrapper)
                for part in trainer.model_parts
                for module in part.modules()
            ), "Requested activation checkpoint wrappers were not applied"
        initial_ema = (
            copy.deepcopy(trainer.ema.state_dict()) if trainer.ema is not None else None
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.fsdp import FSDPModule

            assert isinstance(trainer.model_parts[0], FSDPModule)
            assert trainer.parallel_dims.dp_shard == dist.get_world_size()
            assert any(
                isinstance(parameter, DTensor)
                and parameter.to_local().numel() < parameter.numel()
                for parameter in trainer.model_parts[0].parameters()
            ), "No genuinely sharded parameter found"
        trainer.train()
        assert trainer.step == steps
        states = {
            name: copy.deepcopy(value.state_dict())
            for name, value in manager.states.items()
        }
        if ema != "none":
            assert "ema" in states and states["ema"] and initial_ema
            assert any(
                not torch.equal(
                    value.to_local() if isinstance(value, DTensor) else value,
                    initial_ema[name].to_local()
                    if isinstance(initial_ema[name], DTensor)
                    else initial_ema[name],
                )
                for name, value in states["ema"].items()
                if isinstance(value, torch.Tensor)
            ), "EMA did not update during training"
        return states
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema", choices=("none", "fixed", "dynamic"), default="none")
    parser.add_argument("--ac", choices=("none", "full", "selective"), default="none")
    args = parser.parse_args()
    require_env("dtensor")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Requires a real NPU")
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()
    reference = _train(args.output / "continuous", 4, ema=args.ema, ac=args.ac)
    interrupted = _train(args.output / "resumed", 2, ema=args.ema, ac=args.ac)
    assert interrupted, "No state captured after interrupted training"
    resumed = _train(args.output / "resumed", 4, ema=args.ema, ac=args.ac)
    torch.save(reference, args.output / f"reference-rank{rank}.pt")
    torch.save(resumed, args.output / f"resumed-rank{rank}.pt")
    assert reference.keys() == resumed.keys()
    for name in reference:
        expected, actual = reference[name], resumed[name]
        if name == "rng_state":
            expected = {key: pickle.loads(value) for key, value in expected.items()}
            actual = {key: pickle.loads(value) for key, value in actual.items()}
            for states in (expected, actual):
                for state in states.values():
                    state["numpy"] = (
                        state["numpy"][0],
                        torch.from_numpy(state["numpy"][1].copy()),
                        *state["numpy"][2:],
                    )
        _assert_equal(actual, expected)
        print(f"PASS rank={rank} exact checkpoint state: {name}", flush=True)
    print("PASS real torch_checkpointing NPU save/load/continued training", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
