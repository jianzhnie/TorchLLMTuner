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
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    baseline_cli_args,
    build_candidate_30b_config,
    configure_baseline_30b,
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
        "lm_head": model.lm_head,
    }
    layers = model.layers.values() if baseline else model.layers
    for index, layer in enumerate(layers):
        attention = layer.attention if baseline else layer.self_attn
        moe = layer.moe if baseline else layer.mlp
        modules[f"layer{index}.attention"] = attention
        modules[f"layer{index}.moe"] = moe
        modules[f"layer{index}.router"] = moe.router if baseline else moe.gate
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
            modules[f"layer{index}.projection"] = attention.qkv_linear
            modules[f"layer{index}.rope"] = attention.rope
            if index == 0:
                modules[f"layer{index}.inner_experts"] = (
                    moe.routed_experts.inner_experts
                )
    if not baseline:
        modules["rotary_cache"] = model.model.model.rotary_emb
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("baseline", "candidate"), required=True)
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
    allow_hf32 = baseline_config["training"]["extension"]["allow_hf32"]
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

    from functools import partial

    for name, module in trace_modules(trainer, baseline).items():
        handles.append(module.register_forward_hook(partial(observe, name)))
    if not baseline:
        from transformers.modeling_utils import AttentionInterface

        from llmtuner.models.hf.attention_backend import (
            ATTENTION_BACKEND,
            flex_attention_adapter,
        )

        def trace_attention_inputs(module, query, key, value, attention_mask, **kwargs):
            index = module.layer_idx
            trace[f"layer{index}.inner_query"] = snapshot(query.transpose(1, 2))
            trace[f"layer{index}.inner_key"] = snapshot(key.transpose(1, 2))
            trace[f"layer{index}.inner_value"] = snapshot(value.transpose(1, 2))
            return flex_attention_adapter(
                module, query, key, value, attention_mask, **kwargs
            )

        AttentionInterface.register(ATTENTION_BACKEND, trace_attention_inputs)
        for index, layer in enumerate(model.layers):
            dispatcher = layer.mlp._llmtuner_local_dispatcher
            original_combine = dispatcher.combine

            def trace_combine(
                routed_output,
                metadata,
                hidden_states,
                *,
                index=index,
                original_combine=original_combine,
            ):
                if index == 0:
                    trace[f"layer{index}.inner_experts"] = snapshot(routed_output)
                    trace[f"layer{index}.sorted_scores"] = snapshot(
                        metadata.topk_scores_experts_sorted_N
                    )
                    trace[f"layer{index}.token_indices"] = snapshot(
                        metadata.token_indices_experts_sorted_N
                    )
                return original_combine(routed_output, metadata, hidden_states)

            dispatcher.combine = trace_combine
    original_grouped_mm = torch._grouped_mm
    grouped_calls = 0

    def trace_grouped_mm(input, weight, *, offs):
        nonlocal grouped_calls
        output = original_grouped_mm(input, weight, offs=offs)
        if grouped_calls < 3:
            trace[f"grouped_{grouped_calls}"] = {
                "input": snapshot(input),
                "weight": snapshot(weight),
                "weight_stride": weight.stride(),
                "offsets": snapshot(offs),
                "output": snapshot(output),
            }
        grouped_calls += 1
        return output

    torch._grouped_mm = trace_grouped_mm
    batch = torch.load(root / "consumed-batches.pt", weights_only=False)[0]
    forward = torch.load(root / "forward-batches.pt", weights_only=False)[0]
    try:
        with torch.no_grad():
            if baseline:
                from torch.nn.attention.flex_attention import and_masks
                from torchtitan.models.common.attention import (
                    create_attention_mask,
                    get_causal_mask_mod,
                    get_efficient_causal_mask_mod_for_packed_document,
                )

                tokens = forward["args"][0].to("npu")
                positions = forward["kwargs"]["positions"].to("npu")
                masks = create_attention_mask(
                    and_masks(
                        get_causal_mask_mod(),
                        get_efficient_causal_mask_mod_for_packed_document(positions),
                    ),
                    1,
                    None,
                    positions.shape[1],
                    positions.shape[1],
                    device=positions.device,
                    BLOCK_SIZE=128,
                    separate_full_blocks=True,
                )
                logits = model(tokens, positions, masks)
            else:
                values = batch["kwargs"]
                tokens = values["input_dict"]["input"].reshape(-1).to("npu")
                positions = forward["kwargs"]["positions"].reshape(-1).to("npu")
                logits = model(tokens, positions=positions)
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
