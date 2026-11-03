"""Ascend NPU: activation exchange and AC numerical compatibility."""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    TrainingConfig,
)
from llmtuner.memory import activation_swap
from llmtuner.trainer.trainer import Trainer


def _run(
    directory: Path, swap: bool, ac_mode: str
) -> tuple[dict[str, torch.Tensor], int, int, dict[str, float | int]]:
    activation_swap.reset_activation_swap_metrics()
    config = LLMTunerConfig(
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
            activation_checkpoint_mode=ac_mode,
            activation_swap=swap,
            checkpoint_config=CheckpointConfig(enable=False),
        ),
    )
    trainer = Trainer(config)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in trainer.example_model.state_dict().items()
    }
    torch.npu.reset_peak_memory_stats()
    released_count = 0
    original_release = activation_swap._ForwardSwapState.release

    def count_release(state, outputs):
        nonlocal released_count
        original_release(state, outputs)
        released_count += sum(packed.released for packed in state.packed)

    if swap:
        activation_swap._ForwardSwapState.release = count_release
    try:
        profiling = swap and os.environ.get("NPU_SWAP_PROFILE") == "1"
        if profiling:
            from torch_npu import profiler as npu_profiler

            context = npu_profiler.profile(
                activities=[
                    npu_profiler.ProfilerActivity.CPU,
                    npu_profiler.ProfilerActivity.NPU,
                ]
            )
        else:
            context = nullcontext()
        with context as profiler:
            trainer.train()
        if profiling:
            profiler.export_chrome_trace(str(directory / "trace.json"))
        metrics = activation_swap.activation_swap_metrics()
        updated = {
            name: value.detach().cpu().clone()
            for name, value in trainer.example_model.state_dict().items()
        }
        assert any(
            not torch.equal(value, initial[name]) for name, value in updated.items()
        )
        return (
            updated,
            torch.npu.max_memory_allocated(),
            released_count,
            metrics,
        )
    finally:
        activation_swap._ForwardSwapState.release = original_release
        trainer.close()


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("activation swap integration requires a visible NPU")
    root = Path(sys.argv[1])
    # Every torchrun rank uses the same output root; creation is idempotent.
    root.mkdir(parents=True, exist_ok=True)
    for ac_mode in ("none", "full"):
        baseline, baseline_memory, _, _ = _run(
            root / ac_mode / "baseline", swap=False, ac_mode=ac_mode
        )
        swapped, swapped_memory, released_count, metrics = _run(
            root / ac_mode / "swapped", swap=True, ac_mode=ac_mode
        )
        if ac_mode == "none":
            assert released_count > 0, (
                "activation swap did not release any saved tensor: "
                f"released={released_count}, metrics={metrics}"
            )
            assert metrics["d2h_bytes"] > 0
            assert metrics["h2d_bytes"] > 0
            assert metrics["host_bytes_peak"] > 0
        else:
            assert metrics["d2h_bytes"] == metrics["h2d_bytes"] == 0
            assert metrics["host_bytes_peak"] == metrics["host_bytes_current"], metrics
        assert baseline.keys() == swapped.keys()
        for name, value in swapped.items():
            torch.testing.assert_close(value, baseline[name], rtol=0, atol=0, msg=name)
        print(
            f"PASS activation swap AC={ac_mode}: two-step updates match bitwise; "
            f"released={released_count}; "
            f"peak_allocated_baseline={baseline_memory}, swapped={swapped_memory}; "
            f"transfers={metrics}"
        )


if __name__ == "__main__":
    main()
