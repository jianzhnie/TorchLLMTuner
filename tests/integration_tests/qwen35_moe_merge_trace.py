"""Original-graph MoE branch-order/view interventions, not training acceptance.

Reference, production experts/router and tolerances remain unchanged. No internal
hooks, retain_grad, cloned branch inputs or custom backward are used. Compare
native HF, reference construction order with the shared HF flat view, then
branch-local views with shared computation on the original 3D input. Actual
post-attention norms propagate the measured MoE cotangents to the next boundary;
this is a local amplification probe, not a full-layer or full40 acceptance.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

from tests.integration_tests import qwen35_moe_module_trace as base


def forward_reference_order(module, hidden_states, *, branch_local_views=False):
    """Change graph construction only; use original HF modules/parameters."""
    batch, length, dim = hidden_states.shape
    flat = hidden_states.view(-1, dim)
    router_input = hidden_states if branch_local_views else flat
    _, scores, ids = module.gate(router_input)
    expert_input = hidden_states.view(-1, dim) if branch_local_views else flat
    routed = module.experts(expert_input, ids, scores)
    shared_input = hidden_states if branch_local_views else flat
    # Match reference SigmoidGatedFeedForward: FFN first, gate second.
    shared = module.shared_expert(shared_input)
    shared = torch.sigmoid(module.shared_expert_gate(shared_input)) * shared
    if branch_local_views:
        return routed.view(batch, length, dim) + shared
    return (routed + shared).view(batch, length, dim)


def forward_reference_views(module, hidden_states):
    return forward_reference_order(module, hidden_states, branch_local_views=True)


def norm_vjp(norm, residual, cotangent, outer_cotangent):
    """Actual norm, original graph, measured local cotangent; no hooks."""
    norm.zero_grad(set_to_none=True)
    leaf = residual.detach().clone().requires_grad_()
    output = norm(leaf)
    output.backward(cotangent)
    return {
        "output": output.detach().cpu(),
        "dx": leaf.grad.detach().cpu(),
        "dw": norm.weight.grad.detach().cpu(),
        "attention_residual_dx": (leaf.grad + outer_cotangent).detach().cpu(),
    }


def run(args, report):
    hf, reference, adapter = base.build(args)
    with torch.device("meta"):
        hf_norm = Qwen3_5MoeRMSNorm(2048, eps=1e-6).to(torch.bfloat16)
        reference_norm = base.model_registry("35B-A3B").model.layers[0].ffn_norm.build()
        reference_norm = reference_norm.to(torch.bfloat16)
    before = dict(hf.named_parameters())
    original_forward = hf.forward
    for forward in (forward_reference_order, forward_reference_views):
        hf.forward = MethodType(forward, hf)
        assert before.keys() == dict(hf.named_parameters()).keys()
        assert all(dict(hf.named_parameters())[key] is p for key, p in before.items())
    hf.forward = original_forward
    report["reference_absorb_router_scores"] = (
        reference.routed_experts.token_dispatcher.absorb_router_scores
    )
    assert report["reference_absorb_router_scores"]
    report["sources"] = {}
    for name, target in (
        ("orchestration", base.run_module),
        ("hf_forward", type(hf)),
        ("reference_forward", type(reference)),
        ("production_experts", hf.experts.forward),
        ("router", hf.gate.forward),
        ("reference_norm", type(reference_norm)),
    ):
        path = Path(inspect.getsourcefile(target))
        report["sources"][name] = {"path": str(path), "sha256": base.file_hash(path)}
    if args.cpu_check:
        report["status"] = "CPU_META_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    device = torch.device("npu", args.device)
    assert torch.npu.is_available()
    torch.npu.set_device(device)
    manifest = json.loads((args.artifacts / "initial.json").read_text())
    assert manifest["original_step"] == 0 and not manifest["resume"]
    assert manifest["config_sha256"] == base.file_hash(args.model / "config.json")
    for filename in ("initial-0.pt", "initial-outer.pt"):
        assert base.file_hash(args.artifacts / filename) == manifest["files"][filename]
    source = torch.load(
        args.artifacts / "initial-0.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    outer = torch.load(
        args.artifacts / "initial-outer.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    source_moe = {
        k: v for k, v in source.items() if k.startswith("model.layers.0.mlp.")
    }
    converted = base.boundary(source_moe, to_reference=True)
    converted["lm_head.weight"] = outer["lm_head.weight"]
    native = adapter.from_hf(converted)
    native.pop("lm_head.weight")
    restored = base.boundary(adapter.to_hf(native), to_reference=False)
    assert restored.keys() == source_moe.keys()
    assert all(torch.equal(restored[key], value) for key, value in source_moe.items())
    trace = torch.load(
        args.trace / "hf-stages.pt", weights_only=True, map_location="cpu", mmap=True
    )
    x, dy = trace["ffn_norm"], trace["moe_output.vjp"]
    report["input"] = {
        key: base.descriptor(value)
        for key, value in {
            "x": x,
            "dy": dy,
            "residual": trace["attention_residual"],
        }.items()
    }
    report["trace_sha256"] = base.file_hash(args.trace / "hf-stages.pt")
    report["manifest_sha256"] = base.file_hash(args.artifacts / "initial.json")
    assert torch.equal(dy, trace["layer_output.vjp"])
    reference.to_empty(device=device)
    reference._init_self_buffers(buffer_device=device)
    reference.load_state_dict(
        {k.removeprefix("layers.0.moe."): v for k, v in native.items()}, strict=True
    )
    reference.train()
    common = dict(
        x=x.to(device),
        dy=dy.to(device),
        scores=trace["router_weights"].to(device),
        ids=trace["router_ids"].to(device),
        routed_only=False,
    )
    ref_result, ref_grads = base.run_module(reference, hf=False, **common)
    ref_grads = base.boundary(
        adapter.to_hf({"layers.0.moe." + k: v for k, v in ref_grads.items()}),
        to_reference=False,
    )
    reference.to("cpu")
    norm_weight = source["model.layers.0.post_attention_layernorm.weight"]
    reference_norm.to_empty(device=device)
    reference_norm.load_state_dict({"weight": norm_weight}, strict=True)
    ref_norm = norm_vjp(
        reference_norm,
        trace["attention_residual"].to(device),
        ref_result["dx"].to(device),
        common["dy"],
    )
    assert torch.equal(ref_norm["output"], x)
    reference_norm.to("cpu")
    torch.save(
        {"observations": ref_result, "gradients": ref_grads, "norm": ref_norm},
        args.output / "reference-original.pt",
    )
    hf.to_empty(device=device)
    hf.load_state_dict(
        {k.removeprefix("model.layers.0.mlp."): v for k, v in source_moe.items()},
        strict=True,
    )
    hf.train()
    hf_norm.to_empty(device=device)
    hf_norm.load_state_dict({"weight": norm_weight}, strict=True)
    report["cases"] = {}
    baseline = None
    for name, forward in (
        ("native", None),
        ("reference-order", forward_reference_order),
        ("reference-views", forward_reference_views),
    ):
        hf.forward = original_forward if forward is None else MethodType(forward, hf)
        result, grads = base.run_module(hf, hf=True, **common)
        grads = {"model.layers.0.mlp." + k: v for k, v in grads.items()}
        assert grads.keys() == ref_grads.keys()
        norm = norm_vjp(
            hf_norm,
            trace["attention_residual"].to(device),
            result["dx"].to(device),
            common["dy"],
        )
        assert torch.equal(norm["output"], x)
        if baseline is None:
            baseline = result
        report["cases"][name] = {
            "original_graph": {
                k: base.compare_branch(v, ref_result[k]) for k, v in result.items()
            },
            "parameters": {
                k: base.compare_branch(v, ref_grads[k]) for k, v in grads.items()
            },
            "norm_amplification": {
                k: base.compare_branch(v, ref_norm[k]) for k, v in norm.items()
            },
            "vs_native": {
                k: base.compare_branch(v, baseline[k]) for k, v in result.items()
            },
        }
        case = report["cases"][name]
        primary = [*case["original_graph"].values(), *case["parameters"].values()]
        case["original_graph_all_passed"] = all(value["passed"] for value in primary)
        case["original_graph_all_exact"] = all(value["exact"] for value in primary)
        torch.save(
            {"observations": result, "gradients": grads, "norm": norm},
            args.output / f"hf-{name}-original.pt",
        )
        del grads
    report["status"] = "original_graph_diagnostic_complete_not_full_layer_acceptance"
    report["exit_code_note"] = (
        "Exit 0 means capture completed; read each case's original_graph_all_passed "
        "and original_graph_all_exact. Norm propagation is a local probe, not "
        "full-layer acceptance."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.hf_score_position = "production-pre-down"
    args.hf_expert_projections = "native"
    args.reference_score_position = "actual"
    if args.execute:
        assert args.artifacts is not None and args.trace is not None
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    report = {
        "entry_sha256": base.file_hash(Path(__file__)),
        "tolerances": {"atol": base.ATOL, "rtol": base.RTOL},
        "scope": "original graph branch-order/view interventions; no internal probes",
        "status": "started_not_accepted",
    }
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
