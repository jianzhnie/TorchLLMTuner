"""Resume the full Qwen3.5 recipe into a separate output directory.

Use the same model/data/batch/sequence length and total training horizon as
continuous training. Compare the two final DCP snapshots with
compare_training_checkpoints.py after both jobs exit successfully.
"""

from __future__ import annotations

from pathlib import Path

import torch

from examples.qwen3_5.train import build_argument_parser, build_config
from llmtuner import Trainer
from llmtuner.errors import ConfigError


def main() -> None:
    parser = build_argument_parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=50)
    args = parser.parse_args()
    if args.initial_weights or args.load_step != -1:
        raise ConfigError("Use --checkpoint alone to select the full resume state")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Requires real Ascend devices")
    if not (args.checkpoint / ".metadata").is_file():
        raise FileNotFoundError(args.checkpoint / ".metadata")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Resume verification requires a fresh output directory")
    if not 0 < args.expected_step < args.steps:
        raise ValueError("Expected restored step must be inside the training horizon")
    config = build_config(args)
    checkpoint = config.training.checkpoint_config
    checkpoint.initial_load_path = str(args.checkpoint.resolve())
    checkpoint.initial_load_model_only = False
    torch.set_default_dtype(torch.bfloat16)
    trainer = Trainer(config)
    original_load = trainer.checkpointer.load

    def checked_load(*positional, **keywords):
        loaded = original_load(*positional, **keywords)
        if not loaded or trainer.step != args.expected_step:
            raise AssertionError(
                f"Expected restored step {args.expected_step}, got {trainer.step}; "
                f"loaded={loaded}"
            )
        return loaded

    trainer.checkpointer.load = checked_load
    trainer.train()


if __name__ == "__main__":
    main()
