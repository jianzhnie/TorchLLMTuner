"""Real HF/reference 35B router VJP diagnostic; not training acceptance.

Dedicated reference environment, one NPU, explicitly scheduled by the caller::

    python tests/integration_tests/qwen35_router_alignment.py \
      --model /original/HF --artifacts /tmp/tlt_qwen35_cross_framework_v1 \
      --output /tmp/router.json --execute

CPU --help works without the optional reference environment. No installation,
full model, tokenizer, distributed runtime, or production patch is performed.
"""

from __future__ import annotations

# ruff: noqa: I001
# Resolve native HF before reference optional-backend compatibility patches.
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter,
    )
    import torchtitan_npu  # noqa: F401 -- existing patches, no device calls here
    from torchtitan.distributed.utils import set_spmd_backend
    from torchtitan.models.common.moe import TokenChoiceTopKRouter
    from torchtitan_npu.models.qwen3_5 import model_registry
    from llmtuner.models.hf.expert_kernels import stabilize_moe_router
except ImportError as exc:
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None

ATOL, RTOL = 0.005, 0.02


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def info(tensor):
    tensor = tensor.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nonempty": tensor.numel() > 0,
        "finite": bool(tensor.isfinite().all()),
        "nonzero": bool(tensor.count_nonzero()),
    }


def compare(hf, ref):
    result = {"hf": info(hf), "reference": info(ref), "passed": False}
    valid = all(
        result[side][key]
        for side in ("hf", "reference")
        for key in ("nonempty", "finite", "nonzero")
    )
    if hf.shape != ref.shape or not valid:
        return result
    a, b = hf.float(), ref.float()
    delta = (a - b).abs()
    failures = delta > ATOL + RTOL * a.abs()
    index = int(delta.reshape(-1).argmax())
    result.update(
        max_abs=float(delta.max()),
        rms=float(delta.square().mean().sqrt()),
        failed_elements=int(failures.sum()),
        elements=a.numel(),
        exact_elements=int((a == b).sum()),
        max_error_flat_index=index,
        hf_at_max=float(a.reshape(-1)[index]),
        reference_at_max=float(b.reshape(-1)[index]),
        passed=not bool(failures.any()),
    )
    return result


def run_router(module, weight, inputs, cotangent, device, *, reference):
    module.to_empty(device=device)
    module.to(dtype=weight.dtype)
    module.load_state_dict(
        {"gate.weight" if reference else "weight": weight}, strict=True
    )
    parameter = module.gate.weight if reference else module.weight
    assert torch.equal(parameter.detach().cpu(), weight)
    module.train()
    x = inputs.to(device).detach().requires_grad_(True)
    captured = {}

    def gate_hook(_module, _args, output):
        assert not captured
        captured["logits"] = output

    handle = module.gate.register_forward_hook(gate_hook) if reference else None
    try:
        if reference:
            scores, indices, probabilities = module(x)
            logits = captured["logits"]
        else:
            logits, scores, indices = module(x)
            probabilities = torch.softmax(logits.float(), dim=-1)
        # Expert-addressed VJP: same upstream coefficient for the same expert,
        # independent of top-k ordering. Selection remains nondifferentiable.
        coefficients = cotangent.to(device).gather(-1, indices.reshape(1, 128, 8))
        objective = (scores.reshape(1, 128, 8).float() * coefficients).sum()
        objective.backward()
        assert x.grad is not None and parameter.grad is not None
        tensors = {
            "logits": logits.reshape(128, 256),
            "probabilities": probabilities.reshape(128, 256),
            "scores": scores.reshape(128, 8),
            "indices": indices.reshape(128, 8),
            "dx": x.grad,
            "dw": parameter.grad,
        }
        return {key: value.detach().cpu().clone() for key, value in tensors.items()}
    finally:
        if handle is not None:
            handle.remove()
        module.to("cpu")


def compare_routers(a, b):
    comparisons = {
        key: compare(a[key], b[key]) for key in ("logits", "probabilities", "dx", "dw")
    }
    ai, ao = a["indices"].sort(-1)
    bi, bo = b["indices"].sort(-1)
    same = (ai == bi).all(-1)
    intersection = (ai[:, :, None] == bi[:, None, :]).any(-1)
    # Dense expert-id representation: zero denotes an unselected expert.
    # This exposes mismatched selections instead of comparing unrelated slots.
    aligned = []
    for values in (a, b):
        dense = torch.zeros((128, 256), dtype=values["scores"].dtype)
        dense.scatter_(1, values["indices"], values["scores"])
        aligned.append(dense)
    comparisons["scores_by_expert_id"] = compare(*aligned)
    matched_rows = int(same.sum())
    matched_scores = None
    if matched_rows:
        matched_scores = compare(
            a["scores"].gather(1, ao)[same], b["scores"].gather(1, bo)[same]
        )
    return {
        "comparisons": comparisons,
        "expert_sets": {
            "passed": bool(same.all()),
            "different_tokens": int((~same).sum()),
            "matched_tokens": matched_rows,
            "missing_selected_experts": int((~intersection).sum()),
            "first_different_token_indices": (~same).nonzero().flatten()[:16].tolist(),
        },
        "scores_on_matching_sets": matched_scores,
        "passed": bool(same.all()) and all(v["passed"] for v in comparisons.values()),
    }


def diagnose(args, report):
    if DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Use the existing HF5.13/reference NPU environment"
        ) from DEPENDENCY_ERROR
    assert importlib.metadata.version("transformers").startswith("5.13.")
    manifest_path = args.artifacts / "initial.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["original_step"] == 0 and manifest["resume"] is False
    assert manifest["config_sha256"] == sha256(args.model / "config.json")
    layer_path = args.artifacts / "initial-0.pt"
    layer_hash = sha256(layer_path)
    assert layer_hash == manifest["files"][layer_path.name]
    state = torch.load(layer_path, map_location="cpu", weights_only=True, mmap=True)
    weight = state["model.layers.0.mlp.gate.weight"].clone()
    assert weight.shape == (256, 2048) and weight.dtype == torch.bfloat16
    assert all(info(weight)[key] for key in ("finite", "nonempty", "nonzero"))
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    for key, value in {
        "hidden_size": 2048,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
    }.items():
        assert getattr(config, key) == value, key
    set_spmd_backend("partial_dtensor")
    spec = model_registry("35B-A3B", attn_backend="flex")
    cfg = spec.model.layers[0].moe.router
    assert cfg.top_k == 8 and cfg.num_experts == 256
    assert cfg.score_func == "softmax" and cfg.route_norm and cfg.route_scale == 1
    assert cfg.num_expert_groups is None and not cfg._debug_force_load_balance
    with torch.device("meta"):
        hf = Qwen3_5MoeTopKRouter(config)
        ref = cfg.build()
        hf_fp32 = Qwen3_5MoeTopKRouter(config) if args.hf_fp32 else None
        production = Qwen3_5MoeTopKRouter(config) if args.production_fp32 else None
    if production is not None:
        holder = torch.nn.Module()
        holder.config = SimpleNamespace(
            model_type="qwen3_5_moe_text", router_compute_dtype="float32"
        )
        holder.add_module("gate", production)
        stabilize_moe_router(holder)
    assert isinstance(ref, TokenChoiceTopKRouter)
    report.update(
        initialization={
            "manifest": manifest,
            "manifest_sha256": sha256(manifest_path),
            "layer_sha256": layer_hash,
            "native_key": "model.layers.0.mlp.gate.weight",
            "weight": info(weight),
        },
        sources={
            "hf_router": source(Qwen3_5MoeTopKRouter),
            "reference_router": source(type(ref)),
            "reference_gate": source(type(ref.gate)),
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "transformers")
        },
    )
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn((1, 128, 2048), generator=generator) * 0.7
        + torch.linspace(-0.3, 0.5, 2048)[None, None, :]
        + torch.linspace(-0.2, 0.1, 128)[None, :, None]
    ).to(torch.bfloat16)
    cotangent = torch.randn((1, 128, 256), generator=generator)
    report["input"] = info(x)
    report["cotangent"] = info(cotangent)
    a = run_router(hf, weight, x, cotangent, device, reference=False)
    b = run_router(ref, weight, x, cotangent, device, reference=True)
    report["native_comparison"] = compare_routers(a, b)
    report["actual_dtypes"] = {
        side: {key: str(value.dtype) for key, value in tensors.items()}
        for side, tensors in (("hf", a), ("reference", b))
    }
    if args.hf_fp32:
        # Diagnostic only: run the ORIGINAL HF router with FP32 activations and
        # the same BF16 weight values promoted to FP32. No production mutation.
        c = run_router(
            hf_fp32, weight.float(), x.float(), cotangent, device, reference=False
        )
        report["hf_fp32_diagnostic"] = compare_routers(c, b)
        report["actual_dtypes"]["hf_fp32"] = {
            key: str(value.dtype) for key, value in c.items()
        }
    if production is not None:
        c = run_router(production, weight, x, cotangent, device, reference=False)
        report["production_fp32"] = compare_routers(c, b)
        report["actual_dtypes"]["production_fp32"] = {
            key: str(value.dtype) for key, value in c.items()
        }
    report["status"] = "passed" if report["native_comparison"]["passed"] else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, default=Path("/tmp/tlt_qwen35_cross_framework_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--production-fp32", action="store_true")
    parser.add_argument(
        "--hf-fp32",
        action="store_true",
        help="Extra original HF FP32 router diagnostic; not acceptance",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Explicit caller-authorized NPU execution",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "error",
        "scope": "Published layer0 router only; not training acceptance",
        "atol": ATOL,
        "rtol": RTOL,
        "criterion": "abs(reference-HF) <= atol + rtol*abs(HF); equal expert sets",
        "vjp": (
            "Sum selected scores times shared FP32 expert-addressed random cotangent"
        ),
        "seed": args.seed,
    }
    try:
        diagnose(args, report)
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception:
            if report["status"] != "error":
                raise
            traceback.print_exc()
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
