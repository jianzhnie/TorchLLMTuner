"""Diagnose captured attention VJPs without replacing Trainer acceptance tests."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.integration_tests.alignment_history import (
    capture_artifact_path,
    load_alignment_history,
)


def replay_attention(
    inputs: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cache: torch.Tensor,
    mask: torch.Tensor,
    *,
    fused: bool,
    gradients: dict[str, torch.Tensor],
    attention: str,
    transposed_weight: bool,
) -> torch.Tensor:
    prefix = "model.layers.7.self_attn."
    projections = [weights[f"{prefix}{name}_proj.weight"] for name in ("q", "k", "v")]
    if fused:
        query_weight, key_weight, value_weight = projections
        packed = torch.cat(
            (
                query_weight.reshape(8, 2, 128, 256),
                key_weight.reshape(8, 1, 128, 256),
                value_weight.reshape(8, 1, 128, 256),
            ),
            dim=1,
        ).reshape(4096, 256)
        if transposed_weight:
            packed = packed.T.contiguous().T
        projected = functional.linear(inputs, packed).reshape(1, 32, 8, 4, 128)
        projected = projected.split((2, 1, 1), dim=-2)
        query, key, value = [
            tensor.reshape(1, 32, -1, 128).contiguous() for tensor in projected
        ]
    else:
        query, key, value = [
            functional.linear(inputs, weight).reshape(1, 32, -1, 128)
            for weight in projections
        ]

    def normalize(tensor, name):
        def capture_input(gradient):
            gradients[f"{name}_norm.input_gradient"] = gradient.detach().cpu()

        tensor.register_hook(capture_input)
        variance = tensor.float().pow(2).mean(-1, keepdim=True)
        return weights[f"{prefix}{name}_norm.weight"] * (
            tensor.float() * torch.rsqrt(variance + 1e-6)
        )

    def capture_value(gradient):
        gradients["inner_attention.input2_gradient"] = gradient.detach().cpu()

    value.register_hook(capture_value)
    query, key = normalize(query, "q"), normalize(key, "k")

    def capture_query(gradient):
        gradients["q_norm"] = gradient.detach().cpu()

    def capture_key(gradient):
        gradients["k_norm"] = gradient.detach().cpu()

    query.register_hook(capture_query)
    key.register_hook(capture_key)
    cosine, sine = cache.split(128, dim=-1)
    cosine, sine = cosine[None, :, None], sine[None, :, None]

    def rotate(tensor):
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cosine + torch.cat((-second, first), dim=-1) * sine

    query, key = rotate(query).transpose(1, 2), rotate(key).transpose(1, 2)
    value = value.transpose(1, 2)
    if attention == "flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def mask_mod(batch, head, query_index, key_index):
            return (
                (query_index < 32)
                & (key_index < 32)
                & mask[batch, 0, query_index.clamp(max=31), key_index.clamp(max=31)]
            )

        block_mask = create_block_mask(mask_mod, 1, None, 32, 32, device=inputs.device)
        output = flex_attention(
            query, key, value, block_mask=block_mask, scale=128**-0.5, enable_gqa=True
        )
    else:
        key, value = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
        scores = (query @ key.transpose(-1, -2)) * (128**-0.5)
        probabilities = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        output = probabilities @ value
    output = output.transpose(1, 2).reshape(1, 32, 2048)
    return functional.linear(output, weights[f"{prefix}o_proj.weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", choices=("eager", "flex"), default="eager")
    parser.add_argument("--require-input-gradients", action="store_true")
    parser.add_argument("--transposed-weight", action="store_true")
    parser.add_argument("--require-exact-fused", action="store_true")
    args = parser.parse_args()
    if args.attention == "flex":
        importlib.import_module("torchtitan_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Requires a real NPU")
    set_deterministic_algorithms()
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = False
    report = []
    for rank in (0, 1):
        root = args.baseline / f"rank{rank}"
        baseline = torch.load(root / "module-trace-step-1.pt", weights_only=True)
        if args.require_input_gradients:
            for name in (
                "q_norm.input_gradient",
                "k_norm.input_gradient",
                "inner_attention.input2_gradient",
            ):
                assert f"layer7.{name}" in baseline, name
        candidate = torch.load(
            args.candidate / f"rank{rank}" / "module-trace-step-1.pt",
            weights_only=True,
        )
        for name in ("layer7.input_norm.output", "layer7.attention.gradient"):
            assert torch.equal(
                candidate[name].reshape_as(baseline[name]), baseline[name]
            )
        weights = {
            name: tensor.to("npu")
            for name, tensor in torch.load(
                capture_artifact_path(args.baseline, root, "initial-hf.pt"),
                map_location="cpu",
                mmap=True,
                weights_only=True,
            ).items()
            if name.startswith("model.layers.7.self_attn.")
        }
        forward = load_alignment_history(root, "forward-batches")[0]
        positions = forward["kwargs"]["positions"].flatten()
        cache = baseline["layer7.rope.cache"][positions].to("npu")
        mask = forward["kwargs"]["attention_masks"]["dense_token_mask"]
        mask = mask[..., :32, :32].to("npu")
        for fused in (False, True):
            gradients = {}
            inputs = baseline["layer7.input_norm.output"].to("npu").requires_grad_()
            output = replay_attention(
                inputs,
                weights,
                cache,
                mask,
                fused=fused,
                gradients=gradients,
                attention=args.attention,
                transposed_weight=args.transposed_weight,
            )
            upstream = (
                baseline["layer7.attention.gradient"].to("npu").reshape_as(output)
            )
            gradient = torch.autograd.grad(output, inputs, upstream)[0].cpu()
            assert torch.isfinite(output).all().item()
            assert torch.isfinite(gradient).all().item()
            result = {
                "rank": rank,
                "fused": fused,
                "attention": args.attention,
                "transposed_weight": args.transposed_weight,
                "output_max_ae": (
                    output.detach()
                    .cpu()
                    .reshape_as(baseline["layer7.attention.output"])
                    - baseline["layer7.attention.output"]
                )
                .abs()
                .max()
                .item(),
            }
            for label, trace in (("baseline", baseline), ("candidate", candidate)):
                reference = trace["layer7.input_norm.gradient"]
                result[f"{label}_gradient_max_ae"] = (
                    (gradient.reshape_as(reference) - reference).abs().max().item()
                )
                for name, intermediate in gradients.items():
                    trace_key = (
                        f"layer7.{name}"
                        if name.endswith("_gradient")
                        else f"layer7.{name}.gradient"
                    )
                    if trace_key not in trace:
                        continue
                    reference = trace[trace_key]
                    result[f"{label}_{name}_gradient_max_ae"] = (
                        (intermediate.reshape_as(reference) - reference)
                        .abs()
                        .max()
                        .item()
                    )
            if args.require_exact_fused and fused:
                assert result["output_max_ae"] == 0.0, result
                assert all(
                    value == 0.0
                    for name, value in result.items()
                    if name.startswith("baseline_") and name.endswith("max_ae")
                ), result
            report.append(result)
            print(result, flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
