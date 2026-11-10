"""Bounded, explicit layer selection for full-training backward diagnostics."""

import os
import re

import torch


def trace_modules(trainer, baseline: bool):
    model = trainer.model_parts[0]
    output_projection = (
        model.output if baseline and not hasattr(model, "lm_head") else model.lm_head
    )
    modules = {
        "embedding": model.tok_embeddings,
        "final_norm": model.norm,
        "lm_head": output_projection,
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
            qkv = getattr(attention, "qkv_linear", None)
            if qkv is None:
                for name in ("wq", "wk", "wv"):
                    modules[f"layer{index}.{name}"] = getattr(attention, name)
            else:
                modules[f"layer{index}.qkv"] = qkv
                projections = {
                    name: projection
                    for name in ("wqkv", "wq", "wk", "wv")
                    if isinstance(
                        projection := getattr(qkv, name, None), torch.nn.Module
                    )
                }
                if not projections:
                    raise ValueError(
                        f"Baseline layer {index} has no recognized QKV projection"
                    )
                for name, projection in projections.items():
                    modules[f"layer{index}.projection.{name}"] = projection
            modules[f"layer{index}.inner_attention"] = attention.inner_attention
            if isinstance(getattr(attention, "rope", None), torch.nn.Module):
                modules[f"layer{index}.rope"] = attention.rope
        else:
            if getattr(attention.config, "qkv_projection", "separate") == "fused":
                # These modules retain parameter ownership but their forwards
                # are bypassed by the actual fused projection.
                continue
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


def selected_trace_modules(trainer, *, baseline, selection):
    if selection == "final":
        return {"final_norm": trainer.model_parts[0].norm}
    if not re.fullmatch(r"\d+(,\d+)*", selection):
        raise ValueError("trace layers must be comma-separated nonnegative indices")
    indices = {int(value) for value in selection.split(",")}
    modules = trace_modules(trainer, baseline)
    available = {
        int(match.group(1))
        for name in modules
        if (match := re.fullmatch(r"layer(\d+)", name))
    }
    if not indices <= available:
        raise ValueError(
            f"trace layer indices unavailable: {sorted(indices - available)}"
        )
    # Exclude logits and embedding snapshots to bound host memory and transfers.
    return {
        name: module
        for name, module in modules.items()
        if name == "final_norm"
        or any(
            name == f"layer{i}"
            or (
                os.environ.get("LLMTUNER_ALIGNMENT_TRACE_BOUNDARIES_ONLY") != "1"
                and name.startswith(f"layer{i}.")
            )
            for i in indices
        )
    }
