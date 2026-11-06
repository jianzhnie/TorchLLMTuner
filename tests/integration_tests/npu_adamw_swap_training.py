"""Real NPU Trainer acceptance for explicit AdamW swap and fresh-process DCP.

Run from the project root with PYTHONPATH=.; snapshots are self-generated trusted
artifacts containing only tensors and primitive containers (weights_only load).
Every train/resume phase starts a new process, including under torchrun/FSDP.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from tests.caps import require_env

require_env("torch_npu")

from llmtuner.components.checkpointer.core.chunked_tensor import ChunkedCheckpointTensor
from llmtuner.components.optimizer.adamw_swap import AdamWSwap
from llmtuner.config import (
    CheckpointConfig,
    FSDPConfig,
    LLMTunerConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer


def snapshot(value):
    if isinstance(value, ChunkedCheckpointTensor):
        return {
            "shape": list(value.shape),
            "chunks": [
                [list(offset), snapshot(payload)] for offset, payload in value.chunks
            ],
        }
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [snapshot(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | bytes | bool | int | float):
        return value
    raise TypeError(f"Unsupported snapshot type {type(value)}")


def train(args) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    stop_after = args.steps if args.stop_after is None else args.stop_after
    if not 0 < stop_after <= args.steps:
        raise ValueError("stop-after must be between 1 and the configured steps")
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe" if args.ep > 1 else "llama",
            arch_overrides=(
                {
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32,
                    "norm_topk_prob": True,
                }
                if args.ep > 1
                else {}
            ),
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=world // (args.tp * args.cp * args.pp),
            tensor_parallel_size=args.tp,
            expert_parallel_size=args.ep,
            pipeline_parallel_size=args.pp,
            context_parallel_ulysses_size=args.cp,
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule="1F1B", num_pp_microbatches=2
            ),
            fsdp_config=FSDPConfig(fsdp_cpu_offload=args.cpu_offload and args.swap),
        ),
        optimizer=OptimizerConfig(
            optimizer_type="AdamWSwap" if args.swap else "AdamW",
            implementation=args.implementation,
            swap_prefetch=args.prefetch,
            swap_max_device_bytes=256,
            swap_max_host_bytes=1024 * 1024,
            lr_scheduler_config=LRSchedulerConfig(
                warmup_steps=1,
                total_steps=4,
                decay_ratio=0.5,
            ),
        ),
        training=TrainingConfig(
            steps=args.steps,
            dump_folder=str(args.directory),
            global_batch_size=4,
            max_seq_len=16,
            max_norm=args.max_norm,
            checkpoint_config=CheckpointConfig(
                enable=True,
                interval=2,
                last_save_model_only=False,
                sharded_tensor_parallel=args.tp > 1,
                sharded_expert_parallel=args.ep > 1,
                async_mode=args.async_mode,
                backend=args.checkpoint_backend,
            ),
        ),
    )
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["amsgrad"] = args.amsgrad
    trainer = Trainer(config)
    if stop_after < args.steps:
        should_continue = trainer.should_continue_training
        trainer.should_continue_training = lambda: (
            trainer.step < stop_after and should_continue()
        )
    if args.tp > 1:
        assert any(
            getattr(part, "_tp_checkpoint_layouts", None)
            or any(
                getattr(module, "_tp_checkpoint_layouts", None)
                for module in part.modules()
            )
            for part in trainer.model_parts
        ), "TP must actually shard projections"
    if world // (args.tp * args.cp * args.pp) > 1:
        assert any(
            isinstance(parameter, DTensor)
            for optimizer in trainer.optimizer.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ), "distributed acceptance must actually use sharded parameters"
    gradient_trace = []
    for optimizer in trainer.optimizer.optimizers:
        optimizer.register_step_pre_hook(
            lambda owner, _args, _kwargs: gradient_trace.append(
                [
                    snapshot(parameter.grad)
                    for group in owner.param_groups
                    for parameter in group["params"]
                ]
            )
        )

    def model_snapshot():
        return {
            str(index): snapshot(part.state_dict())
            for index, part in enumerate(trainer.model_parts)
        }

    initial = model_snapshot()
    load = trainer.checkpointer.load
    starts = []

    def load_and_record(*load_args, **load_kwargs):
        result = load(*load_args, **load_kwargs)
        starts.append(trainer.step)
        return result

    trainer.checkpointer.load = load_and_record
    save = trainer.checkpointer.save
    auxiliary = {}

    def save_and_record(*save_args, **save_kwargs):
        result = save(*save_args, **save_kwargs)
        if trainer.step == stop_after:
            if stop_after < args.steps and args.async_mode != "disabled":
                assert result and not save_kwargs.get("last_step", False)
                assert trainer.checkpointer.save_future is not None
                print(
                    f"PASS asynchronous checkpoint submitted at step={trainer.step}",
                    flush=True,
                )
            for key in ("rng_state", "dataloader"):
                owner = trainer.checkpointer.states[key]
                state = owner.state_dict()
                # Decode our own live RNG state: opaque serialization contains
                # process-specific storage IDs which are not generator state.
                if key == "rng_state":
                    state = owner.validate_state_dict(state)
                auxiliary[key] = snapshot(state)
        return result

    trainer.checkpointer.save = save_and_record
    try:
        trainer.train()
        final = model_snapshot()
        assert starts == [args.expected_start], starts
        assert trainer.step == stop_after
        assert any(
            not torch.equal(value, initial[part][name])
            for part, state in final.items()
            for name, value in state.items()
        )
        states = {
            "model": final,
            "optimizer": snapshot(trainer.optimizer.state_dict()),
            "scheduler": snapshot(trainer.lr_scheduler.state_dict()),
            "trainer": snapshot(trainer.state_dict()),
        }
        assert auxiliary.keys() == {"rng_state", "dataloader"}
        states.update(auxiliary)
        if args.cpu_offload and args.swap:
            assert all(
                (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                ).device.type
                == "cpu"
                for optimizer in trainer.optimizer.optimizers
                for group in optimizer.param_groups
                for parameter in group["params"]
            ), "CPU-offload acceptance must actually retain CPU parameters"
        metrics = []
        for optimizer in trainer.optimizer.optimizers:
            if args.implementation == "fused":
                assert all(group["fused"] for group in optimizer.param_groups)
            if args.swap:
                assert isinstance(optimizer, AdamWSwap)
                assert 0 < optimizer.staging_peak_payload_bytes <= 256
                assert optimizer.h2d_bytes >= optimizer.d2h_bytes > 0
                if args.prefetch:
                    assert optimizer.prefetch_count > 0
                assert 0 < optimizer.host_state_bytes <= 1024 * 1024
                for state in optimizer.state.values():
                    for key in (
                        ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
                        if args.amsgrad
                        else ("exp_avg", "exp_avg_sq")
                    ):
                        value = state[key]
                        local = (
                            value.to_local() if isinstance(value, DTensor) else value
                        )
                        assert local.device.type == "cpu"
                        assert local.numel() == 0 or local.is_pinned()
                metrics.append(
                    {
                        "host_state_bytes": optimizer.host_state_bytes,
                        "prefetch_count": optimizer.prefetch_count,
                        "staging_peak_payload_bytes": (
                            optimizer.staging_peak_payload_bytes
                        ),
                        "h2d_bytes": optimizer.h2d_bytes,
                        "d2h_bytes": optimizer.d2h_bytes,
                    }
                )
        rank = int(os.environ.get("RANK", "0"))
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(states, args.output / f"rank-{rank}.pt")
        torch.save(gradient_trace, args.output / f"gradients-{rank}.pt")
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "start": starts[0],
                    "step": trainer.step,
                    "world_size": world,
                    "swap": args.swap,
                    "implementation": args.implementation,
                    "cpu_offload": args.cpu_offload and args.swap,
                    "metrics": metrics,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"PASS Trainer swap={args.swap} rank={rank}: {starts[0]}->{trainer.step}",
            flush=True,
        )
    finally:
        trainer.close()


def compare(actual, expected, path="root", *, rtol=0, atol=0) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=path)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            compare(actual[key], expected[key], f"{path}.{key}", rtol=rtol, atol=atol)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for index, (value, other) in enumerate(zip(actual, expected, strict=True)):
            compare(value, other, f"{path}.{index}", rtol=rtol, atol=atol)
    else:
        assert actual == expected, f"{path}: scalar/opaque value differs"


def suite(args) -> None:
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for label, swap, steps, directory, start in (
        ("device-full", False, 4, "device-full", 0),
        ("swap-full", True, 4, "swap-full", 0),
        ("device-half", False, 2, "device-resume", 0),
        ("device-resume", False, 4, "device-resume", 2),
        ("swap-half", True, 2, "swap-resume", 0),
        ("swap-resume", True, 4, "swap-resume", 2),
    ):
        asynchronous_half = (
            label.endswith("-half")
            and args.checkpoint_backend == "dcp"
            and args.async_mode != "disabled"
        )
        command = (
            [sys.executable]
            if args.world_size == 1
            else [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.world_size}",
            ]
        ) + [
            __file__,
            "train",
            "--directory",
            str(root / directory),
            "--output",
            str(root / label / "snapshot"),
            "--steps",
            str(4 if asynchronous_half else steps),
            "--expected-start",
            str(start),
        ]
        if asynchronous_half:
            # A configured final step forces synchronous DCP saving. Keep the
            # full training plan and stop after an ordinary interval save instead.
            command += ["--stop-after", str(steps)]
        command += [
            "--ep",
            str(args.ep),
            "--tp",
            str(args.tp),
            "--cp",
            str(args.cp),
            "--pp",
            str(args.pp),
            "--checkpoint-backend",
            args.checkpoint_backend,
            "--async-mode",
            args.async_mode,
        ]
        command += [
            "--implementation",
            args.implementation,
            "--max-norm",
            str(args.max_norm),
        ]
        if args.prefetch:
            command.append("--prefetch")
        if args.cpu_offload:
            command.append("--cpu-offload")
        if args.amsgrad:
            command.append("--amsgrad")
        if swap:
            command.append("--swap")
        with (root / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False
            )
        reports.append(
            {"label": label, "command": command, "exit_code": result.returncode}
        )
        (root / "runs.json").write_text(json.dumps(reports, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed: see {root / (label + '.log')}")
    for rank in range(args.world_size):
        reference = torch.load(
            root / "device-full/snapshot" / f"rank-{rank}.pt", weights_only=True
        )
        swapped = torch.load(
            root / "swap-full/snapshot" / f"rank-{rank}.pt", weights_only=True
        )
        # CPU-offload moves gradient clipping arithmetic to CPU, changing its
        # rounding. Keep exact independent-resume checks, but declare an FP32
        # tolerance for the cross-device clipping + fused update comparison.
        compare(
            swapped,
            reference,
            f"rank{rank}.swap-full",
            rtol=1e-5 if args.cpu_offload else 0,
            atol=1e-7 if args.cpu_offload else 0,
        )
        for label, expected in (("device-resume", reference), ("swap-resume", swapped)):
            actual = torch.load(
                root / label / "snapshot" / f"rank-{rank}.pt", weights_only=True
            )
            compare(actual, expected, f"rank{rank}.{label}")
        if args.cpu_offload:
            native_gradients = torch.load(
                root / "device-full/snapshot" / f"gradients-{rank}.pt",
                weights_only=True,
            )
            swapped_gradients = torch.load(
                root / "swap-full/snapshot" / f"gradients-{rank}.pt", weights_only=True
            )
            compare(
                swapped_gradients,
                native_gradients,
                f"rank{rank}.clipped-gradients",
                rtol=1e-5,
                atol=1e-7,
            )
    (root / "PASS").write_text(
        "Resume bitwise equal; cross-device offload uses FP32 tolerance\n"
        if args.cpu_offload
        else "All local states bitwise equal; fresh-process resume verified\n"
    )
    print(
        f"PASS AdamWSwap Trainer + independent {args.checkpoint_backend} "
        f"resume, world_size={args.world_size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("suite")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    worker = commands.add_parser("train")
    worker.add_argument("--directory", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument(
        "--stop-after", type=int, help="Stop without changing the configured step plan"
    )
    worker.add_argument("--expected-start", type=int, required=True)
    worker.add_argument("--swap", action="store_true")
    for subparser in (run, worker):
        subparser.add_argument(
            "--checkpoint-backend",
            choices=("dcp", "torch_checkpointing"),
            default="dcp",
        )
        subparser.add_argument("--ep", type=int, default=1)
        subparser.add_argument("--tp", type=int, default=1)
        subparser.add_argument("--cp", type=int, default=1)
        subparser.add_argument("--pp", type=int, default=1)
        subparser.add_argument(
            "--async-mode",
            choices=("disabled", "async", "async_with_pinned_mem"),
            default="disabled",
        )
        subparser.add_argument(
            "--implementation", choices=("for-loop", "fused"), default="for-loop"
        )
        subparser.add_argument("--max-norm", type=float, default=1.0)
        subparser.add_argument("--prefetch", action="store_true")
        subparser.add_argument("--cpu-offload", action="store_true")
        subparser.add_argument("--amsgrad", action="store_true")
    args = parser.parse_args()
    train(args) if args.command == "train" else suite(args)


if __name__ == "__main__":
    main()
