"""Ascend NPU: virtual moments and activation swap survive checkpoint resume."""

import sys
from pathlib import Path

import torch

from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer


def _config(directory: Path, combined: bool, steps: int) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        optimizer=OptimizerConfig(
            implementation="for-loop",
            state_residency="npu_virtual" if combined else "device",
        ),
        training=TrainingConfig(
            steps=steps,
            dump_folder=str(directory),
            activation_checkpoint_mode="none",
            activation_swap=combined,
            checkpoint_config=CheckpointConfig(
                enable=True,
                interval=2,
                async_mode="disabled",
                last_save_model_only=False,
            ),
        ),
    )


def _run(directory: Path, combined: bool, steps: int):
    trainer = Trainer(_config(directory, combined, steps))
    try:
        trainer.train()
        model = {
            name: value.detach().cpu().clone()
            for name, value in trainer.example_model.state_dict().items()
        }
        optimizer = {
            name: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for name, value in trainer.optimizer.state_dict().items()
        }
        return model, optimizer
    finally:
        trainer.close()


def _compare(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        other = expected[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=5e-3, atol=5e-3, msg=name)
        else:
            assert value == other, name


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("combined memory-policy test requires a visible NPU")
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=False)
    directory = root / "combined"
    _run(directory, combined=True, steps=2)
    assert (directory / "checkpoint/step-2/.metadata").is_file()
    combined_model, combined_optimizer = _run(directory, combined=True, steps=4)
    device_model, device_optimizer = _run(root / "device", combined=False, steps=4)
    _compare(combined_model, device_model)
    _compare(combined_optimizer, device_optimizer)
    print("PASS virtual moments + activation swap checkpoint resume")


if __name__ == "__main__":
    main()
