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
from tests.integration_tests.alignment_history import load_alignment_history
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    baseline_allow_hf32,
    baseline_cli_args,
    build_candidate_30b_config,
    captured_attention,
    configure_baseline_30b,
    effective_forward_positions,
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


def trace_modules(trainer, baseline: bool):
    model = trainer.model_parts[0]
    modules = {
        "embedding": model.tok_embeddings,
        "final_norm": model.norm,
        "lm_head": (model.output if baseline else model.lm_head),
    }
    layers = model.layers.values() if baseline else model.layers
    for index, layer in enumerate(layers):
        attention = layer.attention if baseline else layer.self_attn
        moe = layer.moe if baseline else layer.mlp
        modules[f"layer{index}.attention"] = attention
        modules[f"layer{index}.moe"] = moe
        modules[f"layer{index}.router"] = (
            moe.router if baseline or hasattr(moe, "router") else moe.gate
        )
        router_gate = getattr(modules[f"layer{index}.router"], "gate", None)
        if isinstance(router_gate, torch.nn.Module):
            modules[f"layer{index}.router_gate"] = router_gate
        modules[f"layer{index}"] = layer
        modules[f"layer{index}.input_norm"] = (
            layer.attention_norm if baseline else layer.input_layernorm
        )
        modules[f"layer{index}.post_norm"] = (
            layer.ffn_norm if baseline else layer.post_attention_layernorm
        )
        modules[f"layer{index}.q_norm"] = attention.q_norm
        modules[f"layer{index}.k_norm"] = attention.k_norm
        modules[f"layer{index}.output_projection"] = (
            attention.wo if baseline else attention.o_proj
        )
        if baseline:
            for name in ("wq", "wk", "wv"):
                modules[f"layer{index}.{name}"] = getattr(attention, name)
            modules[f"layer{index}.inner_attention"] = attention.inner_attention
        else:
            for name, attribute in (
                ("wq", "q_proj"),
                ("wk", "k_proj"),
                ("wv", "v_proj"),
            ):
                projection = getattr(attention, attribute, None)
                if isinstance(projection, torch.nn.Module):
                    modules[f"layer{index}.{name}"] = projection
    if not baseline:
        modules["rotary_cache"] = model.model.model.rotary_emb
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--grad-enabled", action="store_true")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["LLMTUNER_ALIGNMENT_MODEL"] = QWEN3_30B
    set_deterministic_algorithms()
    init_distributed(timeout=timedelta(hours=2))
    root = args.capture / f"rank{rank}"
    args.output = args.output / f"rank{rank}"
    baseline_config = json.loads((root / "config.json").read_text())
    validate_baseline_30b_dict(baseline_config, world)
    allow_hf32 = baseline_allow_hf32(baseline_config)
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = allow_hf32
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.framework == "baseline"
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

    for name, module in trace_modules(trainer, baseline).items():
        handles.append(module.register_forward_hook(partial(observe, name)))
    if not baseline:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import AttentionInterface

        def traced_sdpa(module, query, key, value, attention_mask, **kwargs):
            name = f"layer{module.layer_idx}.inner_attention"
            trace[f"{name}.inputs"] = snapshot(
                (query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
            )
            output, weights = sdpa_attention_forward(
                module, query, key, value, attention_mask, **kwargs
            )
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
    forward = load_alignment_history(root, "forward-batches")[0]
    try:
        with torch.set_grad_enabled(args.grad_enabled):
            if captured_attention(baseline_config)["mask_type"] != "causal":
                raise ValueError("This trace currently requires captured causal SDPA")
            tokens = forward["args"][0].to("npu")
            if baseline:
                logits = model(tokens)
            else:
                positions = effective_forward_positions(forward).reshape(-1).to("npu")
                logits = model(tokens.reshape(-1), positions=positions)
        trace["logits"] = snapshot(logits)
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
