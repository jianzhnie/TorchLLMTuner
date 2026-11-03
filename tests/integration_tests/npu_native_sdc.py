"""Ascend NPU: native eager SDC wraps modules and permits real training."""

import sys
from pathlib import Path

import torch

from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("native SDC integration requires a visible NPU")
    assert torch.nn.Module.__call__.__module__ == "torch_npu.utils._step", (
        "NPU_ASD_CONFIG did not install the native eager SDC module wrapper"
    )
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(
        LLMTunerConfig(
            model=ModelConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
            training=TrainingConfig(
                steps=2,
                dump_folder=str(directory),
                checkpoint_config=CheckpointConfig(enable=False),
            ),
        )
    )
    initial = {
        name: value.detach().cpu().clone()
        for name, value in trainer.example_model.state_dict().items()
    }
    try:
        trainer.train()
        updated = trainer.example_model.state_dict()
        assert any(
            not torch.equal(value.detach().cpu(), initial[name])
            for name, value in updated.items()
        )
    finally:
        trainer.close()
    print("PASS native eager SDC: module wrapper installed and two steps updated")


if __name__ == "__main__":
    main()
