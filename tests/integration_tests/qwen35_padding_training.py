"""Observe padding in the actual full Qwen3.5 recipe/Trainer, not a new loop.

Run on eight idle NPUs with --execute. Per-rank reports must all finish and
contain real padding in aggregate before claiming padded-path coverage.
This short-run observer does not establish cross-framework alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from examples.qwen3_5.train import build_config
from llmtuner import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Require --execute; no runtime acceptance otherwise")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        parser.error("Require all eight ranks")
    start = 0
    if args.resume is not None:
        suffix = args.resume.name.removeprefix("step-")
        if (
            not args.resume.name.startswith("step-")
            or not suffix.isdigit()
            or not (args.resume / ".metadata").is_file()
        ):
            parser.error("Resume requires a complete step-N DCP directory")
        start = int(suffix)
    if Path(args.output).exists():
        parser.error("Require a fresh output directory; preserve the continuous run")
    if args.steps <= start or args.checkpoint_interval <= 0:
        parser.error("Require positive work and checkpoint interval")
    options = argparse.Namespace(
        **vars(args),
        batch=8,
        sequence_length=2048,
        layers=None,
        initial_weights=None,
        compile_backend="none",
        load_step=-1,
        router_compute_dtype="float32",
        experts_implementation="pre_down",
        rotary_compute_dtype="float32",
        rotary_cache_dtype="float32",
        linear_attention_backend="npu_triton",
        gdn_numerical_policy="split_qkv_fp32_norm",
        gdn_deterministic_reduction=True,
        gdn_allow_right_padding=True,
    )
    rank = int(os.environ["RANK"])
    args.report.mkdir(parents=True, exist_ok=True)
    destination = args.report / f"rank-{rank:05d}.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = {
        "status": "failed",
        "acceptance": "short actual Trainer only; NOT cross-framework acceptance",
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rank": rank,
        "recipe": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()
        },
        "forwards": [],
    }
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in (
            "examples/qwen3_5/train.py",
            "llmtuner/models/hf/model.py",
            "llmtuner/models/hf/padding_metadata.py",
            "llmtuner/models/hf/linear_attention.py",
            "llmtuner/models/qwen3_5/linear_attention.py",
            "llmtuner/models/hf/gdn_numerics.py",
            "llmtuner/models/qwen3_5/gdn_numerics.py",
            "llmtuner/accelerator/gated_delta.py",
        )
    }
    trainer = handle = None
    try:
        torch.set_default_dtype(torch.bfloat16)
        cfg = build_config(options)
        assert cfg.model.arch_overrides["num_hidden_layers"] == 40
        if args.resume is not None:
            cfg.training.checkpoint_config.initial_load_path = str(
                args.resume.resolve()
            )
            cfg.training.checkpoint_config.initial_load_model_only = False
        trainer = Trainer(cfg)
        assert len(trainer.model_parts) == 1
        model = trainer.model_parts[0]
        assert sum(p.numel() for p in model.parameters()) == 34_660_610_688
        original_load = trainer.checkpointer.load

        def checked_load(*values, **kwargs):
            loaded = original_load(*values, **kwargs)
            assert bool(loaded) == bool(args.resume) and trainer.step == start
            report["loaded_step"] = trainer.step
            return loaded

        trainer.checkpointer.load = checked_load

        def observe(_module, inputs, kwargs):
            tokens = inputs[0] if inputs else kwargs["input_ids"]
            padding = kwargs.get("padding_mask")
            assert tokens.ndim == 1 and tokens.numel() == 2048
            padded = 0
            if padding is not None:
                assert padding.dtype == torch.bool and padding.shape == tokens.shape
                padded = int(padding.count_nonzero().item())
                assert padded < tokens.numel(), "Vacuous all-padding input"
                if padded:
                    assert not padding[:-padded].any() and padding[-padded:].all()
            report["forwards"].append(
                {"step": trainer.step, "tokens": tokens.numel(), "padded": padded}
            )

        handle = model.register_forward_pre_hook(observe, with_kwargs=True)
        trainer.train()
        assert trainer.step == args.steps
        assert len(report["forwards"]) == args.steps - start
        checkpoint = Path(args.output) / "checkpoint" / f"step-{args.steps}"
        assert (checkpoint / ".metadata").is_file()
        report.update(
            status="completed_candidate_only",
            step=trainer.step,
            padded_tokens=sum(item["padded"] for item in report["forwards"]),
            checkpoint=str(checkpoint),
        )
        for relative, expected in report["source_sha256"].items():
            assert (
                hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
            ), f"Source changed during training: {relative}"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    finally:
        if handle is not None:
            handle.remove()
        operations = ([trainer.close] if trainer is not None else []) + [
            lambda: dist.destroy_process_group() if dist.is_initialized() else None
        ]
        for operation in operations:
            try:
                operation()
            except Exception as error:
                report["status"] = "failed"
                report.setdefault("error", f"Cleanup {type(error).__name__}: {error}")
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if report["status"] == "failed":
        raise SystemExit(report.get("error", "Padding training failed"))


if __name__ == "__main__":
    main()
