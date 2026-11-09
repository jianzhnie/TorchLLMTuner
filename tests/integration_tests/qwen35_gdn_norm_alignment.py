"""Isolate real GDN gated-norm VJPs, including the 93/128 objective.

No reference mutation or precision intervention. Capture uses original full GDN
geometry and the existing independent recurrence diagnostic, not Trainer.
--trace replays saved activations without rerunning GDN. NPU requires --execute.
93/128 means nonzero inputs and zero output cotangents after token 93, NOT a
real attention_mask. Passing this probe cannot enable production padding:
real-mask semantics, complete module and AC need separate acceptance.
"""

from __future__ import annotations

# ruff: noqa: I001
# Native HF must resolve its optional backends before reference compatibility.
import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNormGated,
)
from tests.integration_tests.qwen35_gdn_module_alignment import (
    ATOL,
    RTOL,
    SHAPES,
    GatedDeltaNet,
    OverrideConfig,
    Qwen35StateDictAdapter,
    apply_overrides,
    callable_source,
    compare,
    file_hash,
    gated_delta_rule,
    load_weights,
    model_registry,
    run_side,
    set_spmd_backend,
)
from torchtitan.models.qwen3_5.model import RMSNormGated


def verify_tensor(tensor, *, shape=None):
    assert tensor.numel() and torch.isfinite(tensor).all(), "empty/nonfinite"
    assert torch.count_nonzero(tensor), "entirely zero tensor"
    if shape is not None:
        assert tuple(tensor.shape) == shape, tensor.shape
    return tensor


def forward_reference_cast_order(module, hidden_states, gate):
    """Candidate-only cast-boundary intervention on the existing HF instance.

    Match actual RMSNormGated.forward, including its FP32 weight multiply;
    do not replace parameters, change their storage dtype, or patch reference.
    """
    input_dtype = hidden_states.dtype
    x = hidden_states.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + module.variance_epsilon)
    x = (module.weight.float() * x).to(input_dtype)
    x = x * torch.nn.functional.silu(gate.float())
    return x.to(input_dtype)


def norm_vjp(module, observation, cotangent, shape, device):
    module.to(device)
    x = observation["norm.input"].reshape(shape).to(device).detach().requires_grad_()
    z = observation["norm.gate"].reshape(shape).to(device).detach().requires_grad_()
    dy = cotangent.reshape(shape).to(device)
    output = module(x, z)
    dx, dz, dw = torch.autograd.grad(output, (x, z, module.weight), dy)
    values = {"output": output, "dx": dx, "dz": dz, "dw": dw}
    result = {
        k: verify_tensor(v.detach().cpu()).reshape(128)
        if k == "dw"
        else verify_tensor(v.detach().cpu()).reshape(4096, 128)
        for k, v in values.items()
    }
    module.to("cpu")
    return result


def channel_errors(a, b):
    """Record all channels, including low-magnitude cancellation failures."""
    assert a.shape == b.shape == (128,)
    error = (a.float() - b.float()).abs()
    limit = ATOL + RTOL * a.float().abs()
    return [
        {
            "channel": i,
            "hf": float(a[i]),
            "reference": float(b[i]),
            "absolute_error": float(error[i]),
            "limit": float(limit[i]),
            "passed": bool(error[i] <= limit[i]),
        }
        for i in range(128)
    ]


def capture(args, report, device):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert config.hidden_size == 2048 and config.num_hidden_layers == 40
    assert config.linear_num_value_heads == 32
    assert config.linear_value_head_dim == 128
    set_spmd_backend("partial_dtensor")
    spec = model_registry("35B-A3B", attn_backend="flex")
    assert apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.qwen3_5.gated_delta.npu"]),
        spec.model,
    )
    with torch.device("meta"):
        hf = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
        ref = spec.model.layers[0].delta_net.build()
    assert type(ref) is GatedDeltaNet
    assert {k: tuple(v.shape) for k, v in hf.state_dict().items()} == SHAPES
    adapter = Qwen35StateDictAdapter(spec.model, None)
    hf_state, ref_state = load_weights(args, adapter, report)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn((1, 128, 2048), generator=generator) * 0.7
        + torch.linspace(-0.3, 0.5, 2048)[None, None, :]
        + torch.linspace(-0.2, 0.1, 128)[None, :, None]
    ).to(torch.bfloat16)
    dy = torch.randn(x.shape, generator=generator).to(torch.bfloat16)
    dy[:, args.valid_tokens :] = 0
    captured = {}
    for side, module, state in (("hf", hf, hf_state), ("reference", ref, ref_state)):
        result = run_side(
            module,
            state,
            x,
            dy,
            device,
            shared_kernel=gated_delta_rule if side == "hf" else None,
        )
        captured[side] = {
            "observations": result["traces"],
            "module_dw": result["paramgrad"]["norm.weight"],
        }
    payload = {
        "sides": captured,
        "weight": hf_state["norm.weight"],
        "seed": args.seed,
        "valid_tokens": args.valid_tokens,
        "initialization": report["initialization"],
        "capture_script_sha256": file_hash(Path(__file__)),
        "capture_sources": {
            "run_side": callable_source(run_side),
            "hf": callable_source(Qwen3_5MoeGatedDeltaNet),
            "reference": callable_source(GatedDeltaNet),
            "recurrence": callable_source(gated_delta_rule),
        },
    }
    torch.save(payload, args.output / "norm-trace.pt")
    return payload


def run(args, report):
    assert args.execute, "NPU execution must be explicit"
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    if args.trace:
        report["trace_sha256"] = file_hash(args.trace)
        payload = torch.load(args.trace, map_location="cpu", weights_only=True)
    else:
        payload = capture(args, report, device)
        report["trace_sha256"] = file_hash(args.output / "norm-trace.pt")
    report["initialization"] = payload["initialization"]
    report["capture_sources"] = payload["capture_sources"]
    report["valid_tokens"] = payload["valid_tokens"]
    assert 0 < payload["valid_tokens"] <= 128
    weight = verify_tensor(payload["weight"], shape=(128,))
    assert weight.dtype == torch.bfloat16
    hf = Qwen3_5MoeRMSNormGated(128, eps=1e-6).to(torch.bfloat16)
    ref = RMSNormGated(RMSNormGated.Config(dim=128, eps=1e-6)).to(torch.bfloat16)
    modules = {"hf": hf, "reference": ref}
    if args.reference_cast_order:
        candidate = Qwen3_5MoeRMSNormGated(128, eps=1e-6).to(torch.bfloat16)
        original_weight = candidate.weight
        candidate.forward = MethodType(forward_reference_cast_order, candidate)
        assert candidate.weight is original_weight
        assert candidate.weight.dtype == torch.bfloat16
        assert set(candidate.state_dict()) == {"weight"}
        modules["reference-cast-order"] = candidate
    for module in modules.values():
        module.load_state_dict({"weight": weight}, strict=True)
    report["sources"] = {
        "hf_norm": callable_source(type(hf)),
        "reference_norm": callable_source(type(ref)),
        "script_sha256": file_hash(Path(__file__)),
    }
    sides = payload["sides"]
    report["captured_boundary_comparisons"] = {
        k: compare(
            sides["hf"]["observations"][k].reshape(-1, 128),
            sides["reference"]["observations"][k].reshape(-1, 128),
        )
        for k in ("norm.input", "norm.gate", "norm.output.grad")
    }
    results, metrics, channels, intervention_metrics = {}, {}, {}, {}
    # Each pair receives byte-identical x/z/dy. Replay both measured sources;
    # independent cotangents cannot hide upstream differences.
    for source, saved in sides.items():
        observation = saved["observations"]
        for k in ("norm.input", "norm.gate", "norm.output.grad"):
            verify_tensor(observation[k].reshape(4096, 128))
        for layout in ("native", "common-flat"):
            key = source + ":" + layout
            results[key] = {}
            for name, module in modules.items():
                shape = (
                    (1, 128, 32, 128)
                    if name == "reference" and layout == "native"
                    else (4096, 128)
                )
                results[key][name] = norm_vjp(
                    module, observation, observation["norm.output.grad"], shape, device
                )
            a, b = results[key]["hf"], results[key]["reference"]
            metrics[key] = {k: compare(a[k], b[k]) for k in a}
            channels[key] = {"native": channel_errors(a["dw"], b["dw"])}
            if args.reference_cast_order:
                candidate_values = results[key]["reference-cast-order"]
                intervention_metrics[key] = {
                    k: compare(candidate_values[k], b[k]) for k in b
                }
                channels[key]["reference-cast-order"] = channel_errors(
                    candidate_values["dw"], b["dw"]
                )
    # Verify each own-activation norm-only replay against full-module dweight.
    report["module_vs_norm_replay_dw"] = {
        side: compare(sides[side]["module_dw"], results[side + ":native"][side]["dw"])
        for side in sides
    }
    torch.save(results, args.output / "norm-vjps.pt")
    report["comparisons"], report["dw_channels"] = metrics, channels
    report["native_passed"] = all(
        v["passed"] for m in metrics.values() for v in m.values()
    )
    report["intervention_comparisons"] = intervention_metrics
    report["intervention_passed"] = (
        all(v["passed"] for m in intervention_metrics.values() for v in m.values())
        if args.reference_cast_order
        else None
    )
    replay_passed = all(
        value["passed"] for value in report["module_vs_norm_replay_dw"].values()
    )
    report["acceptance_basis"] = (
        "Candidate reference-cast-order vs unchanged reference; native retained"
        if args.reference_cast_order
        else "Native HF vs unchanged reference"
    )
    report["passed"] = replay_passed and (
        report["intervention_passed"]
        if args.reference_cast_order
        else report["native_passed"]
    )
    report["status"] = "PASS" if report["passed"] else "NUMERICAL_MISMATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--trace", type=Path, help="Previously captured norm-trace.pt")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument(
        "--reference-cast-order",
        action="store_true",
        help="Additional candidate-only FP32 weight multiply before BF16 cast; "
        "exit criterion selects this comparison, native evidence retained",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-tokens", type=int, default=93)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if not args.trace and (args.model is None or args.artifacts is None):
        parser.error("capture needs --model and --artifacts; replay needs --trace")
    if not 0 < args.valid_tokens <= 128:
        parser.error("valid-tokens must be in 1..128")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "ERROR",
        "reference_cast_order": args.reference_cast_order,
        "padding_scope": (
            "No attention_mask is passed. Inputs remain nonzero; valid_tokens "
            "only zeros trailing output cotangents. This is an objective-mask "
            "norm VJP probe, not real padding support. Production padding "
            "guard stays enabled pending real-mask/module/AC acceptance."
        ),
        "atol": ATOL,
        "rtol": RTOL,
        "scope": (
            "Real gated norm same-input/cotangent VJP; "
            "not padding support or training acceptance"
        ),
        "cast_boundary": (
            "HF: BF16 normalized x * BF16 weight; reference: "
            "FP32 weight * FP32 normalized x then BF16"
        ),
        "note": (
            "Common-flat tests layout/reduction sensitivity without changing "
            "reference arithmetic. Capture uses norm hooks; module replay "
            "is reported, not presumed exact."
        ),
    }
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
