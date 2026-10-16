"""Compare fresh CLI processes: continuous training versus checkpoint resume.

PYTHONPATH=. python tests/integration_tests/cli_checkpoint_equivalence.py

Single CPU process per run, eager Qwen3, synthetic data and native DCP.
Artifacts and logs remain in the printed temporary directory for inspection.
This does not exercise distributed layouts, EMA, compile or device kernels.
"""

import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from llmtuner.accelerator.device import device_type


def assert_state_equal(actual, expected, path="checkpoint"):
    """Compare decoded tensors and metadata with a useful first-difference path."""
    assert type(actual) is type(expected), f"{path}: types differ"
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, msg=lambda message: f"{path}: {message}"
        )
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in actual:
            assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list | tuple):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_state_equal(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def main():
    if device_type != "cpu":
        raise RuntimeError("This CLI checkpoint check requires a CPU-only environment")
    root = Path(mkdtemp(prefix="llmtuner-cli-resume-"))
    print(f"Artifacts: {root}", flush=True)
    command = [sys.executable, "-m", "llmtuner"]
    options = {
        "model_name_or_path": "qwen3",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_seq_len": 8,
        "global_batch_size": 2,
        "seed": 42,
        "deterministic": "true",
        "enable": "true",
        "last_save_model_only": "false",
        # Both runs use the same curve even when the first process stops early.
        "total_steps": 3,
        "decay_ratio": 1.0,
        "log_freq": 1,
    }
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    for name, steps, folder in (
        ("continuous", 3, "continuous"),
        ("initial", 2, "resumed"),
        ("resume", 3, "resumed"),
    ):
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command + ["--steps", str(steps), "--dump_folder", str(root / folder)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, f"{name} failed: {root / f'{name}.log'}"
    assert "Resuming from step 2" in (root / "resume.log").read_text()
    states = []
    for name in ("continuous", "resumed"):
        destination = root / f"{name}.pt"
        dcp_to_torch_save(root / name / "checkpoint" / "step-3", destination)
        state = torch.load(destination, weights_only=False)
        # Pickle storage identifiers need not match across fresh processes.
        # Compare the saved streams themselves, not their opaque encoding.
        rng = pickle.loads(state["rng_state"]["rank_0"])
        numpy_state = rng["numpy"]
        rng["numpy"] = (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            *numpy_state[2:],
        )
        state["rng_state"]["rank_0"] = rng
        states.append(state)
    assert states[0]["rng_state"]["rank_0"]["device_type"] == "cpu"
    assert states[0]["train_state"] == {"step": 3, "ntokens_seen": 48}
    assert states[0]["lr_scheduler"] == {"last_epoch": 3}
    assert_state_equal(states[0], states[1])
    print("Full semantic checkpoint state matches exactly after CLI resume.")


if __name__ == "__main__":
    main()
