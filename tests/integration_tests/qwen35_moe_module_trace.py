"""Actual HF/reference layer-0 MoE local VJP, not training acceptance.

Read original initial-0/outer plus decoder trace hf-stages.pt. Both sides use
exactly its ffn_norm input and moe_output.vjp local cotangent. Run actual MoE
first, then actual routed experts with identical detached leaf scores/IDs to
isolate dscore. --reference-score-position post-down is a diagnostic config
intervention; default actual preserves the installed reference config.
--hf-score-position pre-down preserves the HF expert loop/parameters and
only moves score multiplication before down projection, using reference casts.
--cpu-check constructs only meta modules and checks adapter geometry.
Input-gradient decomposition is a separate replay with identity-cloned branch
inputs; its output/dx are checked against the uninstrumented actual MoE run.
--hf-expert-projections separate-gate-up changes only candidate pre-down
projections, keeping original parameters and the actual reference unchanged.
local-grouped also replaces per-expert indexing with one LocalTokenDispatcher
gather, three grouped GEMMs and an unweighted deterministic scatter combine.
"""

from __future__ import annotations

# ruff: noqa: I001
import argparse
import json
from pathlib import Path
from types import MethodType

import torch
import torchtitan_npu  # noqa: F401
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan_npu.models.qwen3_5 import model_registry

from llmtuner.models.hf.expert_kernels import (
    forward_fp32_topk_router,
    stabilize_moe_experts,
)
from llmtuner.models.hf.model_setup import configure_experts_implementation
from llmtuner.models.common.scatter_add import deterministic_scatter_add
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher

from tests.integration_tests.qwen35_cross_framework import boundary, compare
from tests.integration_tests.qwen35_decoder_layer_trace import descriptor, file_hash

ATOL, RTOL = 0.005, 0.02


def forward_hf_pre_down(
    module, hidden_states, top_k_index, top_k_weights, *, separate_gate_up=False
):
    """Diagnostic HF expert loop with only the score position/casts changed."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=module.num_experts
        ).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == module.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        weight = module.gate_up_proj[expert_idx]
        if separate_gate_up:
            gate_weight, up_weight = weight.chunk(2, dim=0)
            gate = torch.nn.functional.linear(current_state, gate_weight)
            up = torch.nn.functional.linear(current_state, up_weight)
        else:
            gate, up = torch.nn.functional.linear(current_state, weight).chunk(
                2, dim=-1
            )
        current_hidden_states = module.act_fn(gate) * up
        # Match actual _ClampGroupedExperts: FP32 product, then BF16 before W2.
        current_hidden_states = (
            current_hidden_states.float()
            * top_k_weights[token_idx, top_k_pos, None].float()
        ).to(current_hidden_states.dtype)
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, module.down_proj[expert_idx]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    return final_hidden_states


def forward_hf_separate_gate_up(module, hidden_states, top_k_index, top_k_weights):
    return forward_hf_pre_down(
        module, hidden_states, top_k_index, top_k_weights, separate_gate_up=True
    )


def forward_hf_local_grouped(module, hidden_states, top_k_index, top_k_weights):
    """Diagnostic local grouped experts over original HF parameter views.

    No new parameters, cached weight views or per-expert GEMM fallback. Score
    multiplication is FP32 before the BF16 down projection; dispatcher.combine
    is deliberately bypassed because it would multiply the scores again.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("local-grouped diagnostic requires BF16 activations")
    counts = torch.nn.functional.one_hot(top_k_index, module.num_experts).sum((0, 1))
    routed, counts, metadata = module._llmtuner_trace_dispatcher.dispatch(
        hidden_states, top_k_weights, top_k_index, counts
    )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    # Resolve views after materialization/FSDP unshard, on every invocation.
    gate_weight, up_weight = module.gate_up_proj.chunk(2, dim=1)
    gate = torch._grouped_mm(routed, gate_weight.transpose(-2, -1), offs=offsets)
    up = torch._grouped_mm(routed, up_weight.transpose(-2, -1), offs=offsets)
    hidden = module.act_fn(gate) * up
    hidden = (
        hidden.float() * metadata.topk_scores_experts_sorted_N.float().reshape(-1, 1)
    ).to(hidden.dtype)
    output = torch._grouped_mm(hidden, module.down_proj.transpose(-2, -1), offs=offsets)
    return deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(
            -1, hidden_states.shape[-1]
        ),
        output,
    )


def build(args):
    set_spmd_backend("partial_dtensor")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    if args.hf_score_position == "production-pre-down":
        config.experts_implementation = "pre_down"
        configure_experts_implementation(config, model_cls=object)
    assert (config.hidden_size, config.num_experts, config.moe_intermediate_size) == (
        2048,
        256,
        512,
    )
    spec = model_registry("35B-A3B")
    moe = spec.model.layers[0].moe
    moe.load_balance_coeff = None
    if args.reference_score_position == "post-down":
        moe.routed_experts.token_dispatcher.absorb_router_scores = False
    with torch.device("meta"):
        hf = Qwen3_5MoeSparseMoeBlock(config).to(torch.bfloat16)
        reference = moe.build().to(torch.bfloat16)
    hf.gate.forward = MethodType(forward_fp32_topk_router, hf.gate)
    if args.hf_score_position == "pre-down":
        before = dict(hf.named_parameters())
        hf.experts.forward = MethodType(forward_hf_pre_down, hf.experts)
        after = dict(hf.named_parameters())
        assert before.keys() == after.keys()
        assert all(after[key] is value for key, value in before.items())
    elif args.hf_score_position == "production-pre-down":
        assert args.reference_score_position == "actual", (
            "Production helper comparison requires unchanged actual reference"
        )
        holder = torch.nn.Module()
        holder.config = config
        holder.add_module("moe", hf)
        before = dict(hf.named_parameters())
        stabilize_moe_experts(holder)
        after = dict(hf.named_parameters())
        assert before.keys() == after.keys()
        assert all(after[key] is value for key, value in before.items())
    if args.hf_expert_projections in ("separate-gate-up", "local-grouped"):
        assert args.hf_score_position in ("pre-down", "production-pre-down")
        assert args.reference_score_position == "actual"
        before = dict(hf.named_parameters())
        if args.hf_expert_projections == "local-grouped":
            hf.experts._llmtuner_trace_dispatcher = LocalTokenDispatcher(
                hf.experts.num_experts, hf.gate.top_k
            )
            forward = forward_hf_local_grouped
        else:
            forward = forward_hf_separate_gate_up
        hf.experts.forward = MethodType(forward, hf.experts)
        after = dict(hf.named_parameters())
        assert before.keys() == after.keys()
        assert all(after[key] is value for key, value in before.items())
    return hf, reference, Qwen35StateDictAdapter(spec.model, None)


def input_gradient_decomposition(model, *, hf, x, dy):
    """Replay actual forward with isolated identity edges, no parameter grads.

    HF shared expert and shared gate have separate input edges; reference's
    shared module contains both. FP32 branch sums are observations, not an
    alternative backward implementation. BF16 order probes expose rounding;
    none is presumed to reproduce autograd's internal accumulation schedule.
    """
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    edges, handles = {}, []

    def isolate(name):
        def hook(module, inputs):
            assert name not in edges, f"Repeated branch {name}"
            edge = inputs[0].clone()
            edges[name] = edge
            return (edge, *inputs[1:])

        return hook

    branches = (
        {
            "router": model.gate,
            "routed": model.experts,
            "shared_raw": model.shared_expert,
            "shared_gate": model.shared_expert_gate,
        }
        if hf
        else {
            "router": model.router,
            "routed": model.routed_experts,
            "shared": model.shared_experts,
        }
    )
    try:
        for name, module in branches.items():
            handles.append(module.register_forward_pre_hook(isolate(name)))
        output = model(x)
        names = list(edges)
        grads = torch.autograd.grad(output, [x, *edges.values()], dy)
        result = {"output": output.detach().cpu(), "dx": grads[0].detach().cpu()}
        for name, grad in zip(names, grads[1:], strict=True):
            result["dx_" + name] = grad.detach().cpu().reshape(x.shape)
        if hf:
            result["dx_shared"] = result["dx_shared_raw"] + result["dx_shared_gate"]
            shared_fp32 = (
                result["dx_shared_raw"].float() + result["dx_shared_gate"].float()
            )
        else:
            shared_fp32 = result["dx_shared"].float()
        routed, router = result["dx_routed"], result["dx_router"]
        result["dx_sum_fp32"] = routed.float() + shared_fp32 + router.float()
        result["dx_sum_fp32_to_bf16"] = result["dx_sum_fp32"].to(torch.bfloat16)
        result["dx_sum_bf16_routed_shared_router"] = (
            routed + result["dx_shared"]
        ) + router
        result["dx_sum_bf16_router_routed_shared"] = (router + routed) + result[
            "dx_shared"
        ]
        return result
    finally:
        for handle in handles:
            handle.remove()


def compare_branch(a, b):
    """Keep original tolerances and expose exactness/elementwise failures."""
    result = compare(a, b, ATOL, RTOL)
    if a.shape != b.shape or a.dtype != b.dtype:
        return result | {"exact": False, "failed_elements": None}
    left, right = a.detach().float().cpu(), b.detach().float().cpu()
    invalid = ~torch.isfinite(left) | ~torch.isfinite(right)
    outside = (left - right).abs() > ATOL + RTOL * left.abs()
    return result | {
        "exact": torch.equal(a, b),
        "failed_elements": int((invalid | outside).sum()),
    }


def run_module(model, *, hf, x, dy, scores, ids, routed_only, observe_router=False):
    """Original graph by default; router retain-grad is a separate replay.

    Cloning the external input creates a fresh leaf, not internal branch edges.
    No hook or retain_grad is installed for primary whole-MoE acceptance.
    """
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    scores = scores.detach().clone().requires_grad_()
    observations = {}
    handles = []
    if routed_only:
        if hf:
            output = model.experts(
                x.reshape(-1, 2048), ids.reshape(-1, 8), scores.reshape(-1, 8)
            ).reshape_as(x)
        else:
            counts = torch.nn.functional.one_hot(ids, 256).sum((0, 1, 2))
            output = model.routed_experts(x, scores, ids, counts)
    elif observe_router:
        router = model.gate if hf else model.router

        def observe(module, inputs, result):
            logits, weights, indices = result if hf else (None, result[0], result[1])
            weights.retain_grad()
            observations.update(weights=weights, ids=indices)
            if logits is not None:
                observations["logits"] = logits

        handles.append(router.register_forward_hook(observe))
    try:
        if not routed_only:
            output = model(x)
        output.backward(dy)
        result = {"output": output.detach().cpu(), "dx": x.grad.detach().cpu()}
        if routed_only:
            result["dscore"] = scores.grad.detach().cpu()
        elif observe_router:
            for name, value in observations.items():
                result[name] = value.detach().cpu()
            result["dscore"] = observations["weights"].grad.detach().cpu()
        gradients = {
            name: p.grad.detach().cpu()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        return result, gradients
    finally:
        for handle in handles:
            handle.remove()


def run(args, report):
    hf, reference, adapter = build(args)
    report["reference_actual_classes"] = {
        "moe": type(reference).__name__,
        "experts": type(reference.routed_experts.inner_experts).__name__,
        "dispatcher": type(reference.routed_experts.token_dispatcher).__name__,
    }
    report["absorb_router_scores"] = (
        reference.routed_experts.token_dispatcher.absorb_router_scores
    )
    meta = {"layers.0.moe." + k: v for k, v in reference.state_dict().items()}
    hf_meta = boundary(adapter.to_hf(meta), to_reference=False)
    assert {
        k.removeprefix("model.layers.0.mlp.") for k in hf_meta
    } == hf.state_dict().keys()
    if args.cpu_check:
        report["status"] = "CPU_META_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    assert args.execute and torch.npu.is_available()
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    manifest = json.loads((args.artifacts / "initial.json").read_text())
    assert manifest["original_step"] == 0 and not manifest["resume"]
    assert manifest["config_sha256"] == file_hash(args.model / "config.json")
    for name in ("initial-0.pt", "initial-outer.pt"):
        assert file_hash(args.artifacts / name) == manifest["files"][name]
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
    source = {k: v for k, v in source.items() if k.startswith("model.layers.0.mlp.")}
    converted = boundary(source, to_reference=True)
    converted["lm_head.weight"] = outer["lm_head.weight"]
    native = adapter.from_hf(converted)
    native.pop("lm_head.weight")
    restored = boundary(adapter.to_hf(native), to_reference=False)
    assert restored.keys() == source.keys()
    assert all(torch.equal(restored[k], v) for k, v in source.items())
    states = {
        "hf": {k.removeprefix("model.layers.0.mlp."): v for k, v in source.items()},
        "reference": {k.removeprefix("layers.0.moe."): v for k, v in native.items()},
    }
    trace = torch.load(
        args.trace / "hf-stages.pt", map_location="cpu", weights_only=True, mmap=True
    )
    x, dy = trace["ffn_norm"], trace["moe_output.vjp"]
    ids, scores = trace["router_ids"], trace["router_weights"]
    assert x.shape == dy.shape and x.shape[-1] == 2048
    report["inputs"] = {
        k: descriptor(v) for k, v in dict(x=x, dy=dy, ids=ids, scores=scores).items()
    }
    report["trace_sha256"] = file_hash(args.trace / "hf-stages.pt")
    captured = {}
    for side, model, is_hf in (("hf", hf, True), ("reference", reference, False)):
        model.to_empty(device=device)
        if not is_hf:
            model._init_self_buffers(buffer_device=device)
        model.load_state_dict(states[side], strict=True)
        model.train()
        captured[side] = {}
        for mode in (
            "actual_moe",
            "observed_actual_moe",
            "leaf_scores_actual_routed_experts",
        ):
            result, grads = run_module(
                model,
                hf=is_hf,
                x=x.to(device),
                dy=dy.to(device),
                scores=scores.to(device),
                ids=ids.to(device),
                routed_only=mode == "leaf_scores_actual_routed_experts",
                observe_router=mode == "observed_actual_moe",
            )
            if not is_hf:
                grads = boundary(
                    adapter.to_hf({"layers.0.moe." + k: v for k, v in grads.items()}),
                    to_reference=False,
                )
            else:
                grads = {"model.layers.0.mlp." + k: v for k, v in grads.items()}
            torch.save(
                {"observations": result, "gradients": grads},
                args.output / f"{side}-{mode}.pt",
            )
            captured[side][mode] = (result, grads)
        decomposition = input_gradient_decomposition(
            model, hf=is_hf, x=x.to(device), dy=dy.to(device)
        )
        torch.save(decomposition, args.output / f"{side}-input-gradient-branches.pt")
        captured[side]["input_gradient_branches"] = decomposition
        model.to("cpu")
    report["comparisons"] = {}
    for mode in captured["hf"]:
        if mode == "input_gradient_branches":
            continue
        a, ag = captured["hf"][mode]
        b, bg = captured["reference"][mode]
        assert ag.keys() == bg.keys()
        report["comparisons"][mode] = {
            "observations": {
                k: compare_branch(a[k].reshape_as(b[k]), b[k])
                for k in (
                    ("output", "dx")
                    if mode == "actual_moe"
                    else ("output", "dx", "dscore")
                )
            },
            "parameters": {k: compare(v, bg[k], ATOL, RTOL) for k, v in ag.items()},
            "dscore_note": (
                "Original graph has no dscore probe. Observed replay slots need "
                "equal IDs; leaf mode has identical IDs/scores."
            ),
            "primary_acceptance": mode == "actual_moe",
        }
        if mode == "observed_actual_moe" and not torch.equal(
            a["ids"].reshape_as(b["ids"]), b["ids"]
        ):
            report["comparisons"][mode]["observations"]["dscore"] = {
                "comparable": False,
                "reason": "Ordered expert IDs differ",
            }
        if mode == "observed_actual_moe":
            report["comparisons"][mode]["ordered_ids_exact"] = torch.equal(
                a["ids"].reshape_as(b["ids"]), b["ids"]
            )
    branches = {side: captured[side]["input_gradient_branches"] for side in captured}
    report["input_gradient_decomposition"] = {
        "cross_framework": {
            key: compare_branch(value, branches["reference"][key])
            for key, value in branches["hf"].items()
            if key in branches["reference"]
        },
        "replay_and_composition": {},
        "metadata": {
            side: {key: descriptor(value) for key, value in values.items()}
            for side, values in branches.items()
        },
        "note": (
            "Branch input clones preserve values but can affect backward scheduling. "
            "Check replay against baseline first; FP32 and BF16 sums are diagnostic "
            "observations, not assertions about the actual accumulation order. "
            "HF shared_raw/shared_gate are separate edges; "
            "reference shared is combined."
        ),
    }
    for side, values in branches.items():
        actual = captured[side]["actual_moe"][0]
        report["input_gradient_decomposition"]["replay_and_composition"][side] = {
            "replay_" + key: compare_branch(values[key], actual[key])
            for key in ("output", "dx")
        } | {
            key + "_vs_actual_dx": compare_branch(value, actual["dx"])
            for key, value in values.items()
            if key.startswith("dx_sum_")
        }
        observed, observed_grads = captured[side]["observed_actual_moe"]
        original_grads = captured[side]["actual_moe"][1]
        report.setdefault("router_observation_replay_vs_original", {})[side] = {
            "observations": {
                key: compare_branch(observed[key], actual[key])
                for key in ("output", "dx")
            },
            "parameters": {
                key: compare_branch(value, original_grads[key])
                for key, value in observed_grads.items()
            },
        }
    primary = report["comparisons"]["actual_moe"]
    report["original_graph_all_passed"] = all(
        value["passed"]
        for group in (primary["observations"], primary["parameters"])
        for value in group.values()
    )
    report["acceptance_basis"] = (
        "actual_moe: original forward/backward without internal hooks, retain_grad "
        "or cloned edges; router observations, leaf scores and branch decomposition "
        "are separate diagnostic replays and do not determine acceptance."
    )
    report["status"] = (
        "isolated_actual_MoE_local_VJP_diagnostic_not_training_acceptance"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--hf-score-position",
        choices=("actual", "pre-down", "production-pre-down"),
        default="actual",
    )
    parser.add_argument(
        "--hf-expert-projections",
        choices=("native", "separate-gate-up", "local-grouped"),
        default="native",
        help="Candidate-only pre-down intervention; actual reference required.",
    )
    parser.add_argument(
        "--reference-score-position", choices=("actual", "post-down"), default="actual"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    if args.execute:
        assert args.trace is not None and args.artifacts is not None
    args.output.mkdir(parents=True)
    report = {
        "scope": "same local input/cotangent actual MoE and leaf dscore",
        "entry_sha256": file_hash(Path(__file__)),
        "reference_score_position": args.reference_score_position,
        "hf_score_position": args.hf_score_position,
        "hf_expert_projections": args.hf_expert_projections,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
    }
    try:
        run(args, report)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
