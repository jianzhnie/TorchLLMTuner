"""Actual layer-0 shared FFN/gate VJPs; not Trainer or 100-step acceptance.

Only the published shared modules are materialized. Native is the default.
The reference is never patched. --execute is required for NPU execution;
--cpu-check builds meta modules and verifies adapter/key geometry only.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from torchtitan.models.common.feed_forward import FeedForward

from tests.integration_tests.qwen35_cross_framework import boundary, compare
from tests.integration_tests.qwen35_decoder_layer_trace import descriptor, file_hash
from tests.integration_tests.qwen35_moe_module_trace import build

ATOL, RTOL = 0.005, 0.02
PREFIX = "model.layers.0.mlp."
KEYS = {
    "shared_expert.gate_proj.weight": "w1.weight",
    "shared_expert.up_proj.weight": "w3.weight",
    "shared_expert.down_proj.weight": "w2.weight",
    "shared_expert_gate.weight": "gate.weight",
}


def checked(value):
    assert value.numel() and torch.isfinite(value).all(), "empty/nonfinite tensor"
    assert torch.count_nonzero(value), "entirely zero tensor"
    return value.detach().cpu()


def source_info(callable_):
    path = Path(inspect.getsourcefile(callable_))
    return {
        "name": callable_.__qualname__,
        "file": str(path),
        "sha256": file_hash(path),
        "source": inspect.getsource(callable_),
    }


def evaluate(ffn, gate, *, reference, x, dy, mode, fp32_gate=False):
    """Compose exactly the original HF shared expression, not a new model."""
    x = x.detach().clone().requires_grad_(True)
    # Separate identity edges expose branch VJPs without altering arithmetic.
    xf, xg = x.clone(), x.clone()
    parameters = dict(ffn.named_parameters())
    if not reference:
        parameters = {"shared_expert." + k: v for k, v in parameters.items()}
        parameters.update(
            {"shared_expert_gate." + k: v for k, v in gate.named_parameters()}
        )
    selected = {
        k: p
        for k, p in parameters.items()
        if mode == "combined"
        or (
            ("gate.weight" == k if reference else k.startswith("shared_expert_gate."))
            == (mode == "gate")
        )
    }
    raw = logits = sigmoid = None
    if mode != "gate":
        raw = FeedForward.forward(ffn, xf) if reference else ffn(xf)
    if mode != "ffn":
        # Explicit candidate-only exploratory intervention; not asserted to be
        # the reference's implementation. Parameter identity/dtype stay intact.
        logits = (
            torch.nn.functional.linear(xg.float(), gate.weight.float())
            if fp32_gate
            else gate(xg)
        )
        sigmoid = logits.sigmoid()
    output = raw if mode == "ffn" else sigmoid if mode == "gate" else sigmoid * raw
    upstream = dy[..., :1].contiguous() if mode == "gate" else dy
    grads = torch.autograd.grad(
        output, (x, *selected.values()), upstream.to(output.dtype), retain_graph=True
    )
    result = {"output": checked(output), "dx": checked(grads[0])}
    result.update(
        {"param:" + k: checked(g) for k, g in zip(selected, grads[1:], strict=True)}
    )
    if mode == "combined":
        edge_grads = torch.autograd.grad(output, (xf, xg, raw, logits), upstream)
        result.update(
            {
                k: checked(v)
                for k, v in zip(
                    ("ffn_input.vjp", "gate_input.vjp", "raw.vjp", "gate_logits.vjp"),
                    edge_grads,
                    strict=True,
                )
            }
        )
        result["branch_sum_fp32"] = checked(
            edge_grads[0].float() + edge_grads[1].float()
        )
        result["branch_sum_bf16"] = checked(edge_grads[0] + edge_grads[1])
    for name, value in (("raw", raw), ("gate_logits", logits), ("sigmoid", sigmoid)):
        if value is not None:
            result[name] = checked(value)
    if mode == "combined":
        # Cloning each input edge changes BF16 accumulation grouping: the raw
        # FFN's two projection gradients are summed before the gate branch.
        # Original shared forward feeds all three projections from one leaf.
        # Thus replay dx is evidence, not an exactness prerequisite.
        actual_x = x.detach().clone().requires_grad_(True)
        if reference:
            actual = ffn(actual_x)
        else:
            actual_raw = ffn(actual_x)
            actual_logits = (
                torch.nn.functional.linear(actual_x.float(), gate.weight.float())
                if fp32_gate
                else gate(actual_x)
            )
            actual = actual_logits.sigmoid() * actual_raw
        actual_grads = torch.autograd.grad(
            actual, (actual_x, *selected.values()), dy.to(actual.dtype)
        )
        actual_values = {"output": checked(actual)}
        actual_values.update(
            {
                name: checked(gradient)
                for name, gradient in zip(
                    ("dx", *("param:" + k for k in selected)),
                    actual_grads,
                    strict=True,
                )
            }
        )
        for name, value in actual_values.items():
            result["replay:" + name] = result[name]
            result[name] = value
    if reference:

        def native_key(key):
            prefix = "replay:" if key.startswith("replay:") else ""
            key = key.removeprefix(prefix) if prefix else key
            if key.startswith("param:"):
                key = "param:" + next(h for h, r in KEYS.items() if r == key[6:])
            return prefix + key

        result = {native_key(k): v for k, v in result.items()}
    return result


def run(args, report):
    # Explicit complete build namespace; routed modules remain meta and unused.
    args.hf_score_position = "native"
    args.reference_score_position = "actual"
    args.hf_expert_projections = "native"
    hf, reference, adapter = build(args)
    ffn, gate, ref = hf.shared_expert, hf.shared_expert_gate, reference.shared_experts
    del hf, reference
    assert {k: tuple(v.shape) for k, v in ffn.state_dict().items()} == {
        "gate_proj.weight": (512, 2048),
        "up_proj.weight": (512, 2048),
        "down_proj.weight": (2048, 512),
    }
    assert tuple(gate.weight.shape) == (1, 2048)
    assert set(ref.state_dict()) == set(KEYS.values())
    meta_native = {
        "layers.0.moe.shared_experts." + k: v for k, v in ref.state_dict().items()
    }
    assert set(boundary(adapter.to_hf(meta_native), to_reference=False)) == {
        PREFIX + k for k in KEYS
    }
    report["sources"] = {
        "hf_ffn": source_info(type(ffn).forward),
        "hf_gate": source_info(type(gate).forward),
        "reference_combined": source_info(type(ref).forward),
        "reference_ffn": source_info(FeedForward.forward),
        "reference_gate": source_info(type(ref.gate).forward),
        "build": source_info(build),
        "script_sha256": file_hash(Path(__file__)),
    }
    report["gate_classes"] = {"hf": str(type(gate)), "reference": str(type(ref.gate))}
    if args.cpu_check:
        report["status"] = "META_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    assert args.execute, "choose --cpu-check or --execute"
    paths = {
        name: args.artifacts / name
        for name in ("initial.json", "initial-0.pt", "initial-outer.pt")
    }
    manifest = json.loads(paths["initial.json"].read_text())
    assert manifest["original_step"] == 0 and not manifest["resume"]
    assert file_hash(args.model / "config.json") == manifest["config_sha256"]
    report["initial_sha256"] = {k: file_hash(p) for k, p in paths.items()}
    for name in ("initial-0.pt", "initial-outer.pt"):
        assert report["initial_sha256"][name] == manifest["files"][name]
    initial = torch.load(
        paths["initial-0.pt"], weights_only=True, mmap=True, map_location="cpu"
    )
    state = {PREFIX + k: initial[PREFIX + k] for k in KEYS}
    for value in state.values():
        checked(value)
        assert value.dtype == torch.bfloat16
    outer = torch.load(
        paths["initial-outer.pt"], weights_only=True, mmap=True, map_location="cpu"
    )
    converted = boundary(state, to_reference=True)
    converted["lm_head.weight"] = outer["lm_head.weight"]
    native = adapter.from_hf(converted)
    native.pop("lm_head.weight")
    restored = boundary(adapter.to_hf(native), to_reference=False)
    assert restored.keys() == state.keys()
    assert all(torch.equal(restored[k], v) for k, v in state.items())
    trace_path = args.trace / "hf-stages.pt"
    trace = torch.load(trace_path, weights_only=True, mmap=True, map_location="cpu")
    x, dy = trace["ffn_norm"], trace["moe_output.vjp"]
    assert x.shape == dy.shape and x.shape[-1] == 2048 and x.numel() // 2048 == 128
    assert x.dtype == dy.dtype == torch.bfloat16
    report["trace_sha256"] = file_hash(trace_path)
    report["inputs"] = {
        k: descriptor(checked(v)) for k, v in {"x": x, "dy": dy}.items()
    }
    assert torch.npu.is_available()
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    for module in (ffn, gate, ref):
        module.to_empty(device=device)
        module.train()
    ffn.load_state_dict(
        {
            k.removeprefix(PREFIX + "shared_expert."): v
            for k, v in state.items()
            if k.startswith(PREFIX + "shared_expert.")
        },
        strict=True,
    )
    gate.load_state_dict(
        {"weight": state[PREFIX + "shared_expert_gate.weight"]}, strict=True
    )
    ref.load_state_dict(
        {k.removeprefix("layers.0.moe.shared_experts."): v for k, v in native.items()},
        strict=True,
    )
    x, dy = x.to(device), dy.to(device)
    results = {}
    for side in (
        "reference",
        "native",
        *(("fp32-gate",) if args.fp32_shared_gate else ()),
    ):
        results[side] = {}
        for mode in ("ffn", "gate", "combined"):
            results[side][mode] = evaluate(
                ref if side == "reference" else ffn,
                ref.gate if side == "reference" else gate,
                reference=side == "reference",
                x=x,
                dy=dy,
                mode=mode,
                fp32_gate=side == "fp32-gate",
            )
            torch.save(results[side], args.output / f"{side}.pt")
    report["comparisons"] = {}
    for side in results.keys() - {"reference"}:
        report["comparisons"][side] = {
            mode: {
                k: compare(v, results["reference"][mode][k], ATOL, RTOL)
                for k, v in tensors.items()
            }
            for mode, tensors in results[side].items()
        }
    report["graph_replay_comparisons"] = {
        side: {
            key.removeprefix("replay:"): {
                **compare(values[key.removeprefix("replay:")], value, ATOL, RTOL),
                "exact": torch.equal(values[key.removeprefix("replay:")], value),
            }
            for key, value in values.items()
            if key.startswith("replay:")
        }
        for side, modes in results.items()
        for values in (modes["combined"],)
    }
    report["tensor_descriptors"] = {
        side: {
            mode: {k: descriptor(v) for k, v in values.items()}
            for mode, values in modes.items()
        }
        for side, modes in results.items()
    }
    report["passed"] = all(
        metric["passed"]
        for modes in report["comparisons"].values()
        for mode, tensors in modes.items()
        for key, metric in tensors.items()
        if mode != "combined" or key in ("output", "dx") or key.startswith("param:")
    )
    report["status"] = "PASS" if report["passed"] else "NUMERICAL_MISMATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New artifact directory"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--fp32-shared-gate",
        action="store_true",
        help="Candidate-only exploratory FP32 linear+sigmoid; reference unchanged",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-check", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "atol": ATOL,
        "rtol": RTOL,
        "options": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "scope": "shared modules only; not full MoE/Trainer/100-step acceptance",
        "cotangents": (
            "FFN/combined: trace moe_output.vjp; isolated sigmoid: "
            "same trace [...,:1] on both sides"
        ),
        "branch_sum": (
            "Isolated two-edge observations; does not establish "
            "routed+shared accumulation order. Combined output/dx/param VJPs "
            "use the original shared-input graph; replay:* and intermediate "
            "edges describe the cloned-edge graph only. Replay differences "
            "are reported separately and do not replace native evidence"
        ),
        "intervention": (
            "Native evidence retained; optional FP32 shared linear+sigmoid "
            "is exploratory, not a reference-equivalence claim"
        ),
    }
    try:
        run(args, report)
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
