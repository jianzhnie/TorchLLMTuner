"""Isolated original layer-0 forward/VJP trace, never reduced-layer training.

Requires existing HF5.13/reference dependencies. No installation is performed.
Use --cpu-check for meta/API validation only. --execute explicitly runs one NPU
without distributed/FSDP, optimizer, tokenizer, or training. Keep original
initial-0.pt/initial-outer.pt and initial.json from cross_framework preparation.
Output is a NEW directory. Original tolerances remain atol=.005, rtol=.02.
Router logits/dtypes, expert IDs, weights and MoE outputs are captured before
interpreting downstream differences. The same reference recurrence is installed
in the HF instance; other HF/reference arithmetic remains unchanged.
"""

from __future__ import annotations

# ruff: noqa: I001
# Match the shared-kernel whole-model diagnostic's import order.
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import MethodType

import torch
import torchtitan_npu  # noqa: F401
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan_npu.models.qwen3_5 import model_registry

from llmtuner.models.hf.expert_kernels import (
    stabilize_moe_experts,
    stabilize_moe_router,
)
from llmtuner.models.hf.model_setup import configure_experts_implementation
from llmtuner.models.qwen3_5.gdn_numerics import configure_gdn_numerics

from tests.integration_tests.qwen35_cross_framework import boundary, compare

ATOL, RTOL = 0.005, 0.02
STAGES = (
    "input_norm",
    "gdn",
    "attention_residual",
    "ffn_norm",
    "router_logits",
    "router_probs",
    "router_ids",
    "router_weights",
    "routed_output",
    "shared_output",
    "moe_output",
    "layer_output",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def descriptor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all()),
        "sha256": hashlib.sha256(
            tensor.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def convert(adapter, state):
    native = adapter.from_hf(boundary(state, to_reference=True))
    native.pop("lm_head.weight")
    return {key.removeprefix("layers.0."): value for key, value in native.items()}


def fp32_router_diagnostic(module, hidden_states):
    """Diagnostic-only precision intervention; preserve HF sorted top-k."""
    hidden_states = hidden_states.reshape(-1, module.hidden_dim)
    logits = torch.nn.functional.linear(hidden_states.float(), module.weight.float())
    probabilities = logits.softmax(-1, dtype=torch.float32)
    weights, indices = probabilities.topk(module.top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return logits, weights.to(logits.dtype), indices


def configure_moe_graph(layer, mode):
    """Candidate-only construction-order intervention; no parameter changes."""
    if mode == "native":
        return
    if mode != "reference-order":
        raise ValueError(f"Unsupported diagnostic MoE graph: {mode}")
    # Delay this import: merge_trace reuses module_trace, which imports this file.
    diagnostic = importlib.import_module(
        "tests.integration_tests.qwen35_moe_merge_trace"
    )
    before = dict(layer.named_parameters())
    shared, router = layer.mlp.shared_expert, layer.mlp.gate
    layer.mlp.forward = MethodType(diagnostic.forward_reference_order, layer.mlp)
    after = dict(layer.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[key] is parameter for key, parameter in before.items())
    assert layer.mlp.shared_expert is shared and layer.mlp.gate is router


def forward_split_qkv(module, hidden_states):
    """Three candidate GEMMs with original parameter slices, no cached views."""
    weights = module.weight.split(module._trace_qkv_sizes, dim=0)
    return torch.cat(
        [torch.nn.functional.linear(hidden_states, weight) for weight in weights], -1
    )


def forward_gated_norm_reference_cast(module, hidden_states, gate=None):
    """Candidate-only cast order; retain HF normalization and SiLU expression."""
    dtype = hidden_states.dtype
    hidden = hidden_states.float()
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + module.variance_epsilon)
    hidden = (module.weight.float() * hidden).to(dtype)
    hidden = hidden * torch.nn.functional.silu(gate.float())
    return hidden.to(dtype)


def configure_gdn_graph(layer, projections, norm_cast):
    """Independent candidate-only diagnostics; no reference/parameter mutation."""
    if projections not in ("native", "separate") or norm_cast not in (
        "native",
        "reference-order",
    ):
        raise ValueError("Unsupported diagnostic GDN graph")
    before = dict(layer.named_parameters())
    gdn = layer.linear_attn
    if projections == "separate":
        assert gdn.in_proj_qkv.bias is None
        sizes = (gdn.key_dim, gdn.key_dim, gdn.value_dim)
        assert sum(sizes) == gdn.in_proj_qkv.weight.shape[0]
        gdn.in_proj_qkv._trace_qkv_sizes = sizes
        gdn.in_proj_qkv.forward = MethodType(forward_split_qkv, gdn.in_proj_qkv)
    if norm_cast == "reference-order":
        # Reject optional fused implementations rather than silently replacing them.
        assert type(gdn.norm).__name__ == "Qwen3_5MoeRMSNormGated"
        assert hasattr(gdn.norm, "variance_epsilon")
        gdn.norm.forward = MethodType(forward_gated_norm_reference_cast, gdn.norm)
    after = dict(layer.named_parameters())
    assert before.keys() == after.keys()
    assert all(after[key] is parameter for key, parameter in before.items())


def models(args):
    assert importlib.metadata.version("transformers").startswith("5.13.")
    set_spmd_backend("partial_dtensor")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert (config.num_hidden_layers, config.hidden_size, config.num_experts) == (
        40,
        2048,
        256,
    )
    assert config.layer_types[0] == "linear_attention"
    config._attn_implementation = "eager"
    config.experts_implementation = args.hf_experts_implementation
    config.gdn_numerical_policy = getattr(args, "hf_gdn_numerical_policy", "native")
    if config.gdn_numerical_policy != "native" and (
        getattr(args, "hf_gdn_qkv_projections", "native") != "native"
        or getattr(args, "hf_gdn_norm_cast", "native") != "native"
    ):
        raise ValueError(
            "Production GDN policy cannot combine with manual interventions"
        )
    configure_experts_implementation(config, model_cls=object)
    spec = model_registry("35B-A3B")
    spec.model.layers[0].moe.load_balance_coeff = None
    apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.qwen3_5.gated_delta.npu"]),
        spec.model,
    )
    with torch.device("meta"):
        hf = Qwen3_5MoeDecoderLayer(config, 0).to(torch.bfloat16)
        reference = spec.model.layers[0].build().to(torch.bfloat16)
    holder = torch.nn.Module()
    holder.config = config
    holder.add_module("layer", hf)
    configure_gdn_numerics(holder)
    parameters_before = dict(hf.named_parameters())
    shared_before = hf.mlp.shared_expert
    router_before = hf.mlp.gate.forward.__func__
    stabilize_moe_experts(holder)
    parameters_after = dict(hf.named_parameters())
    assert parameters_before.keys() == parameters_after.keys()
    assert all(parameters_after[k] is p for k, p in parameters_before.items())
    assert hf.mlp.shared_expert is shared_before
    assert hf.mlp.gate.forward.__func__ is router_before
    if args.hf_router_compute_dtype == "float32":
        hf.mlp.gate.forward = MethodType(fp32_router_diagnostic, hf.mlp.gate)
    kernel = spec.model.layers[0].delta_net.kernel.build()
    deterministic = getattr(args, "hf_gdn_deterministic_reduction", False)

    def recurrence(q, k, v, *, g, beta, **kwargs):
        assert kwargs.get("initial_state") is None
        assert kwargs.get("output_final_state", False) is False
        assert kwargs.get("use_qk_l2norm_in_kernel") is True
        assert kwargs.get("cu_seqlens") is None
        if deterministic:
            candidate = importlib.import_module("llmtuner.accelerator.gated_delta")
            return candidate.gated_delta_rule(
                q, k, v, g, beta, deterministic_reduction=True
            ), None
        return kernel(q, k, v, g, beta), None

    hf.linear_attn.chunk_gated_delta_rule = recurrence
    if args.hf_router_compute_dtype == "production-float32":
        config.router_compute_dtype = "float32"
        stabilize_moe_router(holder)
    configure_moe_graph(hf, getattr(args, "hf_moe_graph", "native"))
    configure_gdn_graph(
        hf,
        getattr(args, "hf_gdn_qkv_projections", "native"),
        getattr(args, "hf_gdn_norm_cast", "native"),
    )
    return hf, reference, Qwen35StateDictAdapter(spec.model, None)


def forward_layer(model, *, hf, x):
    positions = torch.arange(x.shape[1], device=x.device)[None]
    if hf:
        return model(
            x,
            position_embeddings=(None, None),
            attention_mask=None,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
        )
    return model(x, attention_masks=None, positions=positions)


def capture_original(model, *, hf, x, cotangent):
    """Primary full-layer VJP with no internal hooks, retain-grad or clones."""
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    try:
        output = forward_layer(model, hf=hf, x=x)
        output.backward(cotangent)
        observations = {"output": output.detach().cpu(), "dx": x.grad.detach().cpu()}
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().cpu()
        return observations, gradients
    finally:
        model.zero_grad(set_to_none=True)


def capture(model, *, hf, x, cotangent, folder):
    """Diagnostic replay only: retain-grad can change BF16 accumulation order."""
    stages, handles = {}, []

    def remember(name, tensor):
        assert name not in stages, f"Repeated stage: {name}"
        stages[name] = tensor
        if tensor.requires_grad:
            tensor.retain_grad()

    def hook(module, name):
        handles.append(
            module.register_forward_hook(lambda m, a, out: remember(name, out))
        )

    hook(model.input_layernorm if hf else model.attention_norm, "input_norm")
    hook(model.linear_attn if hf else model.attn, "gdn")
    norm = model.post_attention_layernorm if hf else model.ffn_norm
    handles.append(
        norm.register_forward_pre_hook(
            lambda m, a: remember("attention_residual", a[0])
        )
    )
    hook(norm, "ffn_norm")
    moe = model.mlp if hf else model.moe
    router = moe.gate if hf else moe.router

    def router_hook(module, inputs, output):
        if hf:
            logits, weights, ids = output
            remember("router_logits", logits)
            # Exact expression used by the actual HF router. This observation
            # is detached, not substituted into the forward graph.
            remember("router_probs", logits.detach().softmax(-1, dtype=torch.float32))
        else:
            weights, ids, probs = output
            remember("router_probs", probs)
        remember("router_weights", weights)
        remember("router_ids", ids)

    handles.append(router.register_forward_hook(router_hook))
    if not hf:
        hook(router.gate, "router_logits")
    hook(moe.experts if hf else moe.routed_experts, "routed_output")
    if hf:
        hook(moe.shared_expert, "shared_raw")
        hook(moe.shared_expert_gate, "shared_gate_logits")
    else:
        hook(moe.shared_experts, "shared_output")
    hook(moe, "moe_output")
    x = x.detach().clone().requires_grad_()
    try:
        output = forward_layer(model, hf=hf, x=x)
        remember("layer_output", output)
        output.backward(cotangent)
        if hf:
            stages["shared_output"] = (
                stages["shared_raw"].detach()
                * stages["shared_gate_logits"].detach().sigmoid()
            )
        assert set(STAGES) <= stages.keys()
        observations = {}
        for name, value in stages.items():
            observations[name] = value.detach().cpu().reshape(1, x.shape[1], -1)
            if value.grad is not None:
                observations[name + ".vjp"] = (
                    value.grad.detach().cpu().reshape(1, x.shape[1], -1)
                )
        observations["input.vjp"] = x.grad.detach().cpu()
        torch.save(
            observations, folder / ("hf-stages.pt" if hf else "reference-stages.pt")
        )
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().cpu()
        return observations, gradients
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def run(args, report):
    hf, reference, adapter = models(args)
    meta_source = {
        "model.layers.0." + key: value for key, value in hf.state_dict().items()
    }
    meta_source["lm_head.weight"] = torch.empty(
        (248320, 2048), device="meta", dtype=torch.bfloat16
    )
    native_meta = convert(adapter, meta_source)
    assert native_meta.keys() == reference.state_dict().keys()
    assert all(
        native_meta[k].shape == v.shape for k, v in reference.state_dict().items()
    )
    report["sources"] = {}
    for name, cls in (
        ("hf_layer", type(hf)),
        ("reference_layer", type(reference)),
        ("reference_router", type(reference.moe.router)),
        ("reference_experts", type(reference.moe.routed_experts.inner_experts)),
        ("adapter", type(adapter)),
        ("hf_moe_forward", hf.mlp.forward),
        ("hf_gdn_norm_forward", hf.linear_attn.norm.forward),
        ("hf_gdn_qkv_forward", hf.linear_attn.in_proj_qkv.forward),
    ):
        path = Path(inspect.getsourcefile(cls))
        report["sources"][name] = {"path": str(path), "sha256": file_hash(path)}
    report["versions"] = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    report["router_config"] = {
        "hf_moe_graph": getattr(args, "hf_moe_graph", "native"),
        "hf_gdn_deterministic_reduction": getattr(
            args, "hf_gdn_deterministic_reduction", False
        ),
        "hf_gdn_numerical_policy": getattr(args, "hf_gdn_numerical_policy", "native"),
        "hf_gdn_qkv_projections": getattr(args, "hf_gdn_qkv_projections", "native"),
        "hf_gdn_norm_cast": getattr(args, "hf_gdn_norm_cast", "native"),
        "hf_experts_implementation": args.hf_experts_implementation,
        "hf_compute_dtype_diagnostic": args.hf_router_compute_dtype,
        "sorted_topk": reference.moe.router.sorted_topk,
        "absorb_router_scores": (
            reference.moe.routed_experts.token_dispatcher.absorb_router_scores
        ),
        "swiglu_limit": reference.moe.routed_experts.inner_experts.swiglu_limit,
    }
    if args.cpu_check:
        report["status"] = "CPU_META_API_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    assert args.execute and not args.cpu_check
    assert torch.npu.is_available()
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    manifest = json.loads((args.artifacts / "initial.json").read_text())
    assert manifest["original_step"] == 0 and manifest["resume"] is False
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
    source["lm_head.weight"] = outer["lm_head.weight"]
    native = convert(adapter, source)
    restored = boundary(
        adapter.to_hf({"layers.0." + k: v for k, v in native.items()}),
        to_reference=False,
    )
    assert restored.keys() == source.keys() - {"lm_head.weight"}
    assert all(torch.equal(v, source[k]) for k, v in restored.items())
    assert all(v.dtype == torch.bfloat16 for v in native.values())
    hf_state = {
        k.removeprefix("model.layers.0."): v
        for k, v in source.items()
        if k != "lm_head.weight"
    }
    tokens = torch.tensor(args.tokens, dtype=torch.long)
    assert tokens.numel() > 0 and tokens.numel() % 64 == 0
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 248320
    x = outer["model.embed_tokens.weight"][tokens].unsqueeze(0).clone()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cotangent = torch.randn(x.shape, generator=generator).to(torch.bfloat16)
    report["input"] = descriptor(x)
    report["cotangent"] = descriptor(cotangent)
    report["source_manifest_sha256"] = file_hash(args.artifacts / "initial.json")
    observations, gradients, original = {}, {}, {}
    report["instrumented_replay_vs_original"] = {}
    for name, model, state, is_hf in (
        ("hf", hf, hf_state, True),
        ("reference", reference, native, False),
    ):
        model.to_empty(device=device)
        if not is_hf:
            model.moe._init_self_buffers(buffer_device=device)
        model.load_state_dict(state, strict=True)
        model.train()
        original[name], gradients[name] = capture_original(
            model, hf=is_hf, x=x.to(device), cotangent=cotangent.to(device)
        )
        torch.save(
            {"observations": original[name], "gradients": gradients[name]},
            args.output / f"{name}-original-graph.pt",
        )
        observations[name], replay_gradients = capture(
            model,
            hf=is_hf,
            x=x.to(device),
            cotangent=cotangent.to(device),
            folder=args.output,
        )
        report["instrumented_replay_vs_original"][name] = {
            "output": compare(
                observations[name]["layer_output"], original[name]["output"], ATOL, RTOL
            ),
            "dx": compare(
                observations[name]["input.vjp"], original[name]["dx"], ATOL, RTOL
            ),
            "parameters": {
                key: compare(value, gradients[name][key], ATOL, RTOL)
                for key, value in replay_gradients.items()
            },
            "exact": {
                "output": torch.equal(
                    observations[name]["layer_output"], original[name]["output"]
                ),
                "dx": torch.equal(
                    observations[name]["input.vjp"], original[name]["dx"]
                ),
            },
        }
        del replay_gradients
        model.to("cpu")
    reference_grads = boundary(
        adapter.to_hf({"layers.0." + k: v for k, v in gradients["reference"].items()}),
        to_reference=False,
    )
    hf_grads = {"model.layers.0." + k: v for k, v in gradients["hf"].items()}
    assert reference_grads.keys() == hf_grads.keys()
    report["parameter_vjp"] = {
        k: compare(v, reference_grads[k], ATOL, RTOL)
        | {"exact": torch.equal(v, reference_grads[k])}
        for k, v in hf_grads.items()
    }
    report["stages"] = {}
    for stage in STAGES:
        a, b = observations["hf"][stage], observations["reference"][stage]
        entry = {"hf": descriptor(a), "reference": descriptor(b)}
        if stage == "router_ids":
            entry["ordered_exact"] = torch.equal(a, b)
            entry["sets_exact"] = torch.equal(a.sort(-1).values, b.sort(-1).values)
        else:
            entry["comparison"] = compare(a, b, ATOL, RTOL)
        report["stages"][stage] = entry
    # Retained stage VJPs use the SAME final cotangent, not equal local
    # cotangents when downstream computations already differ.
    report["stage_vjp"] = {
        key: compare(value, observations["reference"][key], ATOL, RTOL)
        for key, value in observations["hf"].items()
        if key.endswith(".vjp") and key in observations["reference"]
    }
    a, b = observations["hf"], observations["reference"]
    ai, bi = a["router_ids"], b["router_ids"]
    report["routing"] = {
        "expert_set_mismatch_tokens": int(
            (ai.sort(-1).values != bi.sort(-1).values).any(-1).sum()
        ),
        "ordered_weights_float_comparison": compare(
            a["router_weights"].float(), b["router_weights"].float(), ATOL, RTOL
        ),
        "weights_aligned_by_expert_id": compare(
            a["router_weights"].gather(-1, ai.argsort(-1)).float(),
            b["router_weights"].gather(-1, bi.argsort(-1)).float(),
            ATOL,
            RTOL,
        ),
    }
    for side, values in observations.items():
        top = values["router_probs"].float().topk(9, dim=-1).values
        report["routing"][side + "_min_top8_top9_margin"] = float(
            (top[..., 7] - top[..., 8]).min()
        )
    report["first_failed_stage"] = next(
        (
            k
            for k in STAGES
            if k != "router_ids" and not report["stages"][k]["comparison"]["passed"]
        ),
        None,
    )
    report["instrumented_trace_all_passed"] = (
        report["first_failed_stage"] is None
        and report["routing"]["expert_set_mismatch_tokens"] == 0
        and all(v["passed"] for v in report["parameter_vjp"].values())
        and all(v["passed"] for v in report["stage_vjp"].values())
    )
    report["original_graph"] = {
        key: compare(original["hf"][key], original["reference"][key], ATOL, RTOL)
        | {"exact": torch.equal(original["hf"][key], original["reference"][key])}
        for key in ("output", "dx")
    }
    report["numerical_all_passed"] = all(
        value["passed"] for value in report["original_graph"].values()
    ) and all(value["passed"] for value in report["parameter_vjp"].values())
    report["original_graph_all_exact"] = all(
        value["exact"]
        for group in (report["original_graph"], report["parameter_vjp"])
        for value in group.values()
    )
    report["acceptance_basis"] = (
        "Original full-layer output/dx/parameter VJP without internal hooks or "
        "retain_grad. Stage traces are separate replays and may alter BF16 "
        "accumulation; their differences do not determine primary acceptance. "
        "This isolated result cannot explain or accept the uninstrumented full40 run."
    )
    report["status"] = "isolated_layer_diagnostic_complete_not_training_acceptance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--hf-gdn-deterministic-reduction",
        action="store_true",
        help="Candidate dk/db block reduction; reference unchanged.",
    )
    parser.add_argument(
        "--hf-gdn-numerical-policy",
        choices=("native", "split_qkv_fp32_norm"),
        default="native",
        help="Production GDN policy; excludes manual GDN interventions.",
    )
    parser.add_argument(
        "--hf-gdn-qkv-projections",
        choices=("native", "separate"),
        default="native",
        help="Candidate-only split Q/K/V GEMMs using original HF weight views.",
    )
    parser.add_argument(
        "--hf-gdn-norm-cast",
        choices=("native", "reference-order"),
        default="native",
        help="Candidate-only gated norm: FP32 weight multiplication before BF16 cast.",
    )
    parser.add_argument(
        "--hf-moe-graph",
        choices=("native", "reference-order"),
        default="native",
        help="Candidate-only MoE construction order; original-graph primary VJP.",
    )
    parser.add_argument(
        "--hf-experts-implementation",
        choices=("native", "pre_down"),
        default="native",
        help="Production HF experts option; reference unchanged.",
    )
    parser.add_argument(
        "--hf-router-compute-dtype",
        choices=("native", "float32", "production-float32"),
        default="native",
        help="Diagnostic intervention only; run native first in a separate directory.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(range(100, 228)))
    args = parser.parse_args()
    if args.execute:
        assert args.artifacts is not None
    assert not args.output.exists(), "Require a new output directory"
    args.output.mkdir(parents=True)
    report = {
        "scope": "Original full layer0, same-input/cotangent, no optimizer/FSDP",
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "entry_sha256": file_hash(Path(__file__)),
        "status": "started_not_accepted",
    }
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
