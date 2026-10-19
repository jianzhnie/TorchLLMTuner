"""Compare fresh CLI processes: continuous training versus checkpoint resume.

PYTHONPATH=. python tests/integration_tests/cli_checkpoint_equivalence.py

Default: one CPU process per run, eager Qwen3, synthetic data and native DCP.
Artifacts and logs remain in the printed temporary directory for inspection.
Append dp2 or pp2 for DP2 or DP2 x PP2; add custom, async, ac and/or profile.
Append first_fit for local JSONL data and production FirstFit packing, with one
document chunk per row so CPU SDPA needs no cross-document attention mask.
Append seed for a single-process step-0 export before the initial training run
and a fresh loading boundary (single process, DP2 or DP2 x PP2).
PP uses Interleaved1F1B with four microbatches and six layers.
EMA, validation and compile-backend configuration are programmatic-only;
this script does not expose those or exercise target-device kernels.
"""

import gzip
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.tensor import DTensor

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


def check_seed_loading(args, seed_file):
    """Check the load boundary with deliberately different initial weights."""
    from llmtuner.components.checkpointer.checkpoint_keys import MODEL
    from llmtuner.trainer.cli import parse_config
    from llmtuner.trainer.trainer import Trainer

    trainer = Trainer(parse_config(args + ["--seed", "43"]))
    try:
        expected = torch.load(seed_file, weights_only=False)
        model = trainer.checkpointer.states[MODEL]
        before = {
            key: (value.full_tensor() if isinstance(value, DTensor) else value).clone()
            for key, value in model.state_dict().items()
        }
        assert any(
            not torch.equal(value, expected[key]) for key, value in before.items()
        )
        services = {
            key: deepcopy(value.state_dict())
            for key, value in trainer.checkpointer.states.items()
            if key not in (MODEL, "rng_state")
        }
        rng = torch.get_rng_state().clone()
        assert trainer.checkpointer.load(0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value.full_tensor() if isinstance(value, DTensor) else value,
                expected[key],
                rtol=0,
                atol=0,
            )
        for key, value in services.items():
            assert_state_equal(
                trainer.checkpointer.states[key].state_dict(), value, key
            )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert trainer.step == 0 and trainer.ntokens_seen == 0
        print(
            "Different initial weights replaced; training services and RNG unchanged."
        )
    finally:
        try:
            trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--seed-boundary":
        check_seed_loading(json.loads(sys.argv[2]), sys.argv[3])
        return
    if device_type != "cpu":
        raise RuntimeError("This CLI checkpoint check requires a CPU-only environment")
    modes = set(sys.argv[1:])
    if modes - {"dp2", "pp2", "custom", "async", "ac", "profile", "seed", "first_fit"}:
        raise ValueError(f"Unknown options: {sorted(modes)}")
    world_size = 4 if "pp2" in modes else 2 if "dp2" in modes else 1
    dp_size = 2 if world_size > 1 else 1
    global_batch = 8 if "pp2" in modes else 2
    root = Path(mkdtemp(prefix="llmtuner-cli-resume-"))
    print(f"Artifacts: {root}", flush=True)
    command = [sys.executable, "-m", "llmtuner"]
    options = {
        "model_name_or_path": "qwen3",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 6 if "pp2" in modes else 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_seq_len": 8,
        "global_batch_size": global_batch,
        "seed": 42,
        "deterministic": "true",
        "enable": "true",
        "last_save_model_only": "false",
        # Both runs use the same curve even when the first process stops early.
        "total_steps": 3,
        "decay_ratio": 1.0,
        "log_freq": 1,
    }
    if "first_fit" in modes:
        from tests.data_fixtures import VOCAB, write_tokenizer

        assets = root / "assets"
        write_tokenizer(str(assets))
        corpus = assets / "rows.jsonl"
        # 31 words + BOS/EOS yield 32 shifted targets per document. Every
        # context-sized chunk fills a row, keeping the token-count oracle exact.
        corpus.write_text(
            "".join(
                json.dumps({"text": f"w{index} " + "lorem " * 30}) + "\n"
                for index in range(32)
            )
        )
        options.update(
            dataset="local_jsonl",
            dataset_path=str(corpus),
            tokenizer_path=str(assets),
            packing="first_fit",
            max_num_documents=1,
            vocab_size=len(VOCAB),
        )
    if world_size > 1:
        options["data_parallel_shard_size"] = dp_size
    if "pp2" in modes:
        options.update(
            pipeline_parallel_size=2,
            pipeline_parallel_schedule="Interleaved1F1B",
            num_pp_microbatches=4,
        )
    if "custom" in modes:
        options["fsdp_implementation"] = "custom"
    if "async" in modes:
        options.update(async_mode="async", interval=1)
    if "ac" in modes:
        options["activation_checkpoint_mode"] = "full"
    if "profile" in modes:
        options.update(
            enable_profiling="true",
            profile_freq=1,
            profiler_active=1,
            profiler_warmup=0,
        )
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    if "seed" in modes:
        seed_command = [sys.executable, "-m", "llmtuner"]
        for name, value in options.items():
            if name not in {
                "data_parallel_shard_size",
                "pipeline_parallel_size",
                "pipeline_parallel_schedule",
                "num_pp_microbatches",
            }:
                seed_command.extend((f"--{name}", str(value)))
        with (root / "seed.log").open("w") as log:
            result = subprocess.run(
                seed_command
                + [
                    "--create_seed_checkpoint",
                    "true",
                    "--steps",
                    "3",
                    "--dump_folder",
                    str(root / "resumed"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, f"Seed export failed: {root / 'seed.log'}"
        assert "Created seed checkpoint at step 0" in (root / "seed.log").read_text()
        assert (root / "resumed/checkpoint/step-0/.metadata").is_file()
        assert not (root / "resumed/checkpoint/step-1").exists()
        seed_file = root / "seed.pt"
        dcp_to_torch_save(root / "resumed/checkpoint/step-0", seed_file)
        boundary_command = [sys.executable]
        if world_size > 1:
            boundary_port = (
                int(os.environ.get("LLMTUNER_TEST_MASTER_PORT", "30160")) + 3
            )
            boundary_command.extend(
                [
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={world_size}",
                    "--master_addr=127.0.0.1",
                    f"--master_port={boundary_port}",
                ]
            )
        boundary_command.extend(
            [
                __file__,
                "--seed-boundary",
                json.dumps(
                    command[3:]
                    + ["--steps", "3", "--dump_folder", str(root / "resumed")]
                ),
                str(seed_file),
            ]
        )
        with (root / "seed-boundary.log").open("w") as log:
            result = subprocess.run(
                boundary_command, stdout=log, stderr=subprocess.STDOUT
            )
        assert result.returncode == 0, (
            f"Seed boundary failed: {root / 'seed-boundary.log'}"
        )

        # PP initializes stages with distinct seeds. Both trajectories must
        # start from the exported weights, rather than comparing unrelated starts.
        command.extend(["--initial_load_path", str(root / "resumed/checkpoint/step-0")])

    for index, (name, steps, folder) in enumerate(
        (
            ("continuous", 3, "continuous"),
            ("initial", 2, "resumed"),
            ("resume", 3, "resumed"),
        )
    ):
        run_command = command
        if world_size > 1:
            port = int(os.environ.get("LLMTUNER_TEST_MASTER_PORT", "30160")) + index
            run_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={world_size}",
                "--master_addr=127.0.0.1",
                f"--master_port={port}",
                "--module",
                "llmtuner",
                *command[3:],
            ]
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(
                run_command
                + ["--steps", str(steps), "--dump_folder", str(root / folder)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, f"{name} failed: {root / f'{name}.log'}"
    if "seed" in modes:
        assert "Resuming from step 0" in (root / "initial.log").read_text()
    assert "Resuming from step 2" in (root / "resume.log").read_text()
    states = []
    for name in ("continuous", "resumed"):
        destination = root / f"{name}.pt"
        dcp_to_torch_save(root / name / "checkpoint" / "step-3", destination)
        state = torch.load(destination, weights_only=False)
        # Pickle storage identifiers need not match across fresh processes.
        # Compare the saved streams themselves, not their opaque encoding.
        assert set(state["rng_state"]) == {f"rank_{rank}" for rank in range(world_size)}
        for rank in range(world_size):
            rng = pickle.loads(state["rng_state"][f"rank_{rank}"])
            assert rng["world_size"] == world_size
            numpy_state = rng["numpy"]
            rng["numpy"] = (
                numpy_state[0],
                torch.from_numpy(numpy_state[1].copy()),
                *numpy_state[2:],
            )
            state["rng_state"][f"rank_{rank}"] = rng
        states.append(state)
    assert states[0]["rng_state"]["rank_0"]["device_type"] == "cpu"
    assert states[0]["train_state"] == {
        "step": 3,
        "ntokens_seen": 3 * global_batch // dp_size * 8,
    }
    assert states[0]["lr_scheduler"] == {"last_epoch": 3}
    assert_state_equal(states[0], states[1])
    if "profile" in modes:
        for folder in ("continuous", "resumed"):
            traces = root / folder / "profiling/traces"
            for rank in range(world_size):
                filename = f"rank{rank}_trace.json.gz"
                completed = traces / "iteration_3" / filename
                events = json.loads(gzip.decompress(completed.read_bytes()))[
                    "traceEvents"
                ]
                assert any(event.get("name") == "aten::linear" for event in events)
                tail = traces / "iteration_3_exit" / filename
                assert isinstance(
                    json.loads(gzip.decompress(tail.read_bytes()))["traceEvents"], list
                )
    print("Full semantic checkpoint state matches exactly after CLI resume.")


if __name__ == "__main__":
    main()
