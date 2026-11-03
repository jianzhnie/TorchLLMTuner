"""Collect one steady-state Qwen3 EP trace using the shared recipe."""

import torch

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner import Trainer


def main() -> None:
    config = qwen3_30b_npu_config()
    config.profiler.enable_profiling = True
    config.profiler.profile_freq = config.steps
    config.profiler.profiler_active = 1
    config.profiler.profiler_warmup = max(0, config.steps - 3)
    config.profiler.profiler_repeat = 1
    torch.set_default_dtype(torch.bfloat16)
    Trainer(config).train()


if __name__ == "__main__":
    main()
