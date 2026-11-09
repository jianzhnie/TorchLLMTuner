"""Same-instance original layer VJP repeats; no NPU work without --execute.

Compare production, manual equivalent expressions and production with only its
owned validation hook removed. Never alter reference, tolerances or gradients.
Every repeat saves full output/dx/parameter gradients (substantial disk usage).
Exit zero means capture completed; inspect numerical results in report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from llmtuner.models.hf import gdn_numerics
from tests.integration_tests import qwen35_decoder_layer_trace as layer
from tests.integration_tests import qwen35_moe_module_trace as moe
from tests.integration_tests.qwen35_moe_merge_trace import forward_reference_order


def comparisons(left, right):
    assert left.keys() == right.keys()
    return {key: moe.compare_branch(value, right[key]) for key, value in left.items()}


def run(args, report):
    options = SimpleNamespace(
        model=args.model,
        hf_experts_implementation="pre_down",
        hf_router_compute_dtype="production-float32",
        hf_moe_graph="native",
        hf_gdn_numerical_policy="split_qkv_fp32_norm",
        hf_gdn_qkv_projections="native",
        hf_gdn_norm_cast="native",
        hf_gdn_deterministic_reduction=args.hf_gdn_deterministic_reduction,
    )
    hf, reference, adapter = layer.models(options)
    hook_ids = [
        key
        for key, hook in hf.linear_attn._forward_pre_hooks.items()
        if hook is gdn_numerics._validate_forward
    ]
    assert len(hook_ids) == 1
    report["sources"] = {
        str(Path(module.__file__).resolve()): layer.file_hash(Path(module.__file__))
        for module in (layer, moe, gdn_numerics)
    }
    report["cases"] = ["production", "manual", "production-no-validation-hook"]
    report["hf_gdn_deterministic_reduction"] = args.hf_gdn_deterministic_reduction
    if args.hf_gdn_deterministic_reduction:
        kernel_path = (
            Path(__file__).resolve().parents[2] / "llmtuner/accelerator/gated_delta.py"
        )
        report["sources"][str(kernel_path)] = layer.file_hash(kernel_path)
    if args.cpu_check:
        report["status"] = "CPU_META_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    assert args.artifacts is not None
    manifest = json.loads((args.artifacts / "initial.json").read_text())
    assert manifest["original_step"] == 0 and not manifest["resume"]
    assert manifest["config_sha256"] == layer.file_hash(args.model / "config.json")
    for name in ("initial-0.pt", "initial-outer.pt"):
        assert layer.file_hash(args.artifacts / name) == manifest["files"][name]
    source = torch.load(
        args.artifacts / "initial-0.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    outer = torch.load(
        args.artifacts / "initial-outer.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    source["lm_head.weight"] = outer["lm_head.weight"]
    native = layer.convert(adapter, source)
    restored = layer.boundary(
        adapter.to_hf({"layers.0." + k: v for k, v in native.items()}),
        to_reference=False,
    )
    assert restored.keys() == source.keys() - {"lm_head.weight"}
    assert all(torch.equal(value, source[key]) for key, value in restored.items())
    hf_state = {
        k.removeprefix("model.layers.0."): v
        for k, v in source.items()
        if k != "lm_head.weight"
    }
    x = outer["model.embed_tokens.weight"][torch.arange(100, 228)].unsqueeze(0).clone()
    dy = torch.randn(x.shape, generator=torch.Generator().manual_seed(42)).bfloat16()
    report["input"] = layer.descriptor(x)
    report["cotangent"] = layer.descriptor(dy)
    report["manifest_sha256"] = layer.file_hash(args.artifacts / "initial.json")
    assert torch.npu.is_available()
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    x, dy = x.to(device), dy.to(device)
    report["repeats"] = {}

    def capture_case(model, state, *, is_hf, case):
        model.load_state_dict(state, strict=True)
        model.train()
        first_path = None
        for repeat in range(args.repeats):
            observations, gradients = layer.capture_original(
                model, hf=is_hf, x=x, cotangent=dy
            )
            if not is_hf:
                gradients = layer.boundary(
                    adapter.to_hf({"layers.0." + k: v for k, v in gradients.items()}),
                    to_reference=False,
                )
            else:
                gradients = {"model.layers.0." + k: v for k, v in gradients.items()}
            payload = {"observations": observations, "gradients": gradients}
            path = args.output / f"{case}-repeat{repeat}.pt"
            torch.save(payload, path)
            entry = {
                "finite": all(
                    bool(torch.isfinite(v).all())
                    for group in payload.values()
                    for v in group.values()
                ),
                "observations": {
                    k: layer.descriptor(v) for k, v in observations.items()
                },
                "parameter_descriptors": {
                    k: layer.descriptor(v) for k, v in gradients.items()
                },
            }
            targets = {}
            if first_path is not None:
                targets["vs_same_case_repeat0"] = first_path
            if is_hf:
                targets["vs_reference_repeat0"] = args.output / "reference-repeat0.pt"
                targets["vs_reference_same_repeat"] = (
                    args.output / f"reference-repeat{repeat}.pt"
                )
                if case != "production":
                    targets["vs_production_same_repeat"] = (
                        args.output / f"production-repeat{repeat}.pt"
                    )
            for label, target in targets.items():
                baseline = torch.load(
                    target, map_location="cpu", weights_only=True, mmap=True
                )
                entry[label] = {
                    group: comparisons(values, baseline[group])
                    for group, values in payload.items()
                }
                entry[label]["all_exact"] = all(
                    v["exact"]
                    for group in payload
                    for v in entry[label][group].values()
                )
                entry[label]["all_passed"] = all(
                    v["passed"]
                    for group in payload
                    for v in entry[label][group].values()
                )
                del baseline
            report["repeats"][f"{case}:{repeat}"] = entry
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            first_path = first_path or path
            del payload, observations, gradients

    reference.to_empty(device=device)
    reference.moe._init_self_buffers(buffer_device=device)
    capture_case(reference, native, is_hf=False, case="reference")
    reference.to("cpu")
    hf.to_empty(device=device)
    parameters = dict(hf.named_parameters())
    gdn = hf.linear_attn
    production_qkv, production_norm = gdn.in_proj_qkv.forward, gdn.norm.forward
    production_moe = hf.mlp.forward
    candidate_recurrence = gdn.chunk_gated_delta_rule
    hook = gdn._forward_pre_hooks.pop(hook_ids[0])
    # Keep the same hook ID/kwargs bookkeeping when restoring it.
    gdn._forward_pre_hooks[hook_ids[0]] = hook
    for case in report["cases"]:
        hf.mlp.forward = production_moe
        gdn.in_proj_qkv.forward, gdn.norm.forward = production_qkv, production_norm
        gdn._forward_pre_hooks.pop(hook_ids[0], None)
        if case == "production":
            gdn._forward_pre_hooks[hook_ids[0]] = hook
        elif case == "manual":
            hf.mlp.forward = MethodType(forward_reference_order, hf.mlp)
            gdn.in_proj_qkv._trace_qkv_sizes = (gdn.key_dim, gdn.key_dim, gdn.value_dim)
            gdn.in_proj_qkv.forward = MethodType(
                layer.forward_split_qkv, gdn.in_proj_qkv
            )
            gdn.norm.forward = MethodType(
                layer.forward_gated_norm_reference_cast, gdn.norm
            )
        assert all(dict(hf.named_parameters())[k] is p for k, p in parameters.items())
        assert gdn.chunk_gated_delta_rule is candidate_recurrence
        capture_case(hf, hf_state, is_hf=True, case=case)
    report["status"] = "repeat_capture_complete_not_training_acceptance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--hf-gdn-deterministic-reduction",
        action="store_true",
        help="Use candidate fixed-order recurrence backward for every HF case/repeat.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 2 and not args.output.exists()
    args.output.mkdir(parents=True)
    report = {
        "entry_sha256": layer.file_hash(Path(__file__)),
        "status": "started",
        "tolerances": {"atol": layer.ATOL, "rtol": layer.RTOL},
        "design": (
            "same model instance per side, weights reloaded per case; "
            "no internal observation hooks/retain_grad/cloned edges; "
            "fixed case order, no allocator reset"
        ),
    }
    try:
        run(args, report)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
