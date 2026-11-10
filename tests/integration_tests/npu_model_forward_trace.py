"""Capture layer-level Qwen3-30B forward traces without backward or optimizer."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.integration_tests.alignment_attention_control import (
    batched_attention_control,
)
from tests.integration_tests.alignment_history import load_alignment_history
from tests.integration_tests.alignment_module_trace import trace_modules
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    alignment_runtime_manifest,
    baseline_allow_hf32,
    baseline_cli_args,
    build_candidate_30b_config,
    captured_attention,
    configure_baseline_30b,
    effective_forward_positions,
    input_fingerprints,
    mesh_manifest,
    target_parallelism,
    validate_baseline_30b_dict,
)


def snapshot(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: snapshot(item) for name, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(snapshot(item) for item in value)
    return value


def build_baseline(output: Path, world: int):
    importlib.import_module("torchtitan_npu")
    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args(baseline_cli_args([]))
    configure_baseline_30b(config, world)
    config.dump_folder = str(output)
    config.comm.init_timeout_seconds = 1800
    config.comm.train_timeout_seconds = 1800
    trainer = config.build()
    assert trainer.checkpointer.load(config.checkpoint.load_step)
    return trainer


def build_candidate(baseline, output: Path, world: int):
    config = build_candidate_30b_config(baseline, output, world)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        trainer = Trainer(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert trainer.checkpointer.load(trainer.cfg.checkpoint.load_step)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--grad-enabled", action="store_true")
    parser.add_argument(
        "--batched-attention-control",
        action="store_true",
        help="Diagnostic only: restore captured batch geometry in every SDPA layer",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Observe selected decoder layers; execution still runs the full model",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batched_attention_control and args.framework != "candidate":
        parser.error("batched attention control is candidate-only")
    if args.layers is not None and any(index < 0 for index in args.layers):
        parser.error("layer indices must be non-negative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["LLMTUNER_ALIGNMENT_MODEL"] = QWEN3_30B
    set_deterministic_algorithms()
    init_distributed(timeout=timedelta(hours=2))
    root = args.capture / f"rank{rank}"
    args.output = args.output / f"rank{rank}"
    baseline_config = json.loads((root / "config.json").read_text())
    validate_baseline_30b_dict(baseline_config, world)
    baseline = args.framework == "baseline"
    if baseline:
        requested = target_parallelism()
        captured = baseline_config["parallelism"]
        if any(captured[name] != degree for name, degree in requested.items()):
            raise ValueError(
                "Baseline trace layout differs from capture; set "
                "LLMTUNER_ALIGNMENT_LAYOUT to the captured layout"
            )
    allow_hf32 = baseline_allow_hf32(baseline_config)
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = allow_hf32
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = (
        build_baseline(args.output, world)
        if baseline
        else build_candidate(baseline_config, args.output, world)
    )
    close_dataloader = getattr(trainer.dataloader, "close", None)
    if close_dataloader is not None:
        close_dataloader()
    model = trainer.model_parts[0]
    model.train()
    trace = {}
    handles = []

    def observe(name, module, inputs, output):
        trace[name] = snapshot(output)
        if name.endswith("inner_attention") or name.endswith("output_projection"):
            trace[f"{name}.inputs"] = snapshot(inputs)

    from functools import partial

    modules = trace_modules(trainer, baseline)
    if args.layers is not None:
        available = {
            int(name[5:])
            for name in modules
            if name.startswith("layer") and "." not in name
        }
        if not set(args.layers) <= available:
            raise ValueError(f"Unknown layer indices: {set(args.layers) - available}")
        modules = {
            name: module
            for name, module in modules.items()
            if not name.startswith("layer")
            or int(name.split(".", 1)[0][5:]) in args.layers
        }
    source_names = (
        (
            "torchtitan.trainer",
            "torchtitan.models.llama4.parallelize",
            "torchtitan_npu.converters.kernels.gmm",
            "torchtitan_npu.converters.kernels.moe_dispatch",
            "torchtitan_npu.patches.torchtitan.expert_parallel",
        )
        if baseline
        else (
            "llmtuner.trainer.trainer",
            "llmtuner.parallel.tensor_parallel.tensor_parallel",
        )
    )
    source_names = (
        *source_names,
        "tests.integration_tests.alignment_attention_control",
    )
    runtime = alignment_runtime_manifest((*source_names, __name__))
    forward = load_alignment_history(root, "forward-batches")[0]
    batch_geometry = tuple(forward["args"][0].shape)
    if args.batched_attention_control:
        if len(batch_geometry) != 2:
            raise ValueError("Attention control requires rectangular captured tokens")
        positions = effective_forward_positions(forward).reshape(batch_geometry)
        expected = torch.arange(batch_geometry[1], device=positions.device)
        if not torch.equal(positions, expected.expand_as(positions)):
            raise ValueError("Attention control requires full unpadded documents")
        if captured_attention(baseline_config)["mask_type"] != "causal":
            raise ValueError("Attention control requires causal attention")
    batch = torch.load(root / "consumed-batches.pt", weights_only=False)[0]
    captured_hashes = input_fingerprints([batch], [forward])
    expected_hashes = json.loads((root / "input-hashes.json").read_text())
    if captured_hashes != expected_hashes[:1]:
        raise ValueError("Forward trace input hashes differ from capture")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "runtime": runtime,
                "layers": args.layers,
                "framework": args.framework,
                "input_hashes": captured_hashes,
                "actual_mesh": mesh_manifest(trainer.parallel_dims),
                "grad_enabled": args.grad_enabled,
                "batched_attention_control": args.batched_attention_control,
                "captured_batch_geometry": batch_geometry,
            },
            indent=2,
        )
    )
    for name, module in modules.items():
        handles.append(module.register_forward_hook(partial(observe, name)))
    if not baseline:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import AttentionInterface

        def traced_sdpa(module, query, key, value, attention_mask, **kwargs):
            selected = args.layers is None or module.layer_idx in args.layers
            if not selected and not args.batched_attention_control:
                return sdpa_attention_forward(
                    module, query, key, value, attention_mask, **kwargs
                )
            name = f"layer{module.layer_idx}.inner_attention"
            if selected:
                trace[f"{name}.inputs"] = snapshot(
                    (query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
                )
            if args.batched_attention_control:
                output, _ = batched_attention_control(
                    query, key, value, attention_mask, batch_geometry, **kwargs
                )
                weights = None
            else:
                output, weights = sdpa_attention_forward(
                    module, query, key, value, attention_mask, **kwargs
                )
            if selected:
                trace[name] = snapshot(output)
            return output, weights

        AttentionInterface.register("sdpa", traced_sdpa)
    original_grouped_mm = torch._grouped_mm
    grouped_calls = 0

    def trace_grouped_mm(input, weight, *, offs):
        nonlocal grouped_calls
        output = original_grouped_mm(input, weight, offs=offs)
        if grouped_calls < 3:
            trace[f"grouped_{grouped_calls}"] = {
                "input": snapshot(input),
                "weight_shape": tuple(weight.shape),
                "weight_stride": weight.stride(),
                "offsets": snapshot(offs),
                "output": snapshot(output),
            }
        grouped_calls += 1
        return output

    torch._grouped_mm = trace_grouped_mm
    try:
        with torch.set_grad_enabled(args.grad_enabled):
            if captured_attention(baseline_config)["mask_type"] != "causal":
                raise ValueError("This trace currently requires captured causal SDPA")
            tokens = forward["args"][0].to("npu")
            if baseline:
                logits = model(tokens)
            else:
                positions = effective_forward_positions(forward).reshape(-1).to("npu")
                tokens = tokens.reshape(-1)
                if trainer.parallel_dims.tp > 1:
                    from llmtuner.parallel.context_parallel.input_sharding import (
                        shard_batch_for_tp,
                    )

                    # Match Trainer's sequence-parallel input boundary. Keep
                    # full positions: TP projections gather tokens internally.
                    tokens, _ = shard_batch_for_tp(
                        tokens, tokens, trainer.parallel_dims.get_mesh("tp")
                    )
                logits = model(tokens, positions=positions)
        trace["logits"] = snapshot(logits)
        if alignment_runtime_manifest((*source_names, __name__)) != runtime:
            raise RuntimeError("Forward trace sources/runtime changed during execution")
        torch.save(trace, args.output / "forward-trace.pt")
        print(f"PASS {args.framework} forward trace at {args.output}")
    finally:
        torch._grouped_mm = original_grouped_mm
        for handle in handles:
            handle.remove()
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
