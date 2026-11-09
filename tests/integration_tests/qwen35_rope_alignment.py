"""Original HF/reference partial RoPE VJP, not full-attention acceptance.

Use --cpu-check for CPU geometry/API checks, or explicitly --execute for NPU.
Q/K are post-QK-norm activations: [1,128,16/2,256], first 64 dims rotated.
Text positions use the actual 2D reference path; --mrope additionally tests
asymmetric temporal/height/width positions, without constructing a new model.
"""

from __future__ import annotations

# ruff: noqa: I001
import argparse
import copy
import hashlib
import inspect
import json
import sys
import traceback
from pathlib import Path

try:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    import torchtitan_npu.models.qwen3_5  # noqa: F401 -- existing import compatibility
    from torchtitan.models.qwen3_5.rope import MRoPE
    from llmtuner.models.hf.qkv_projection import (
        apply_qwen35_fp32_rotary,
        forward_qwen35_fp32_cache,
    )
except ImportError as exc:
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None

ATOL, RTOL = 0.005, 0.02

if "--execute" in sys.argv:
    import torch_npu  # noqa: F401 -- explicit NPU execution only


def source(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    assert a.shape == b.shape and a.numel()
    valid = bool(a.isfinite().all() and b.isfinite().all())
    nonzero = bool(a.count_nonzero() and b.count_nonzero())
    if not valid:
        return {"finite": False, "passed": False}
    delta = (a - b).abs()
    bad = delta > ATOL + RTOL * a.abs()
    return {
        "finite": valid,
        "nonzero": nonzero,
        "max_abs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
        "failed_elements": int(bad.sum()),
        "exact_elements": int((a == b).sum()),
        "elements": a.numel(),
        "passed": nonzero and not bool(bad.any()),
    }


def vjp(q, k, uq, uk, operation):
    q, k = q.detach().requires_grad_(), k.detach().requires_grad_()
    a, b = operation(q, k)
    dq, dk = torch.autograd.grad((a, b), (q, k), (uq, uk))
    return {"q": a, "k": b, "dq": dq, "dk": dk}


def run(args, report):
    if DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Requires existing HF5.13 and upstream reference environment"
        ) from DEPENDENCY_ERROR
    config_path = args.model / "config.json"
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert config.model_type == "qwen3_5_moe_text"
    assert config.hidden_size == 2048 and config.head_dim == 256
    assert config.num_attention_heads == 16 and config.num_key_value_heads == 2
    rp = config.rope_parameters
    assert rp["rope_type"] == "default" and rp["rope_theta"] == 10_000_000
    assert rp["partial_rotary_factor"] == 0.25
    assert rp.get("mrope_section", [11, 11, 10]) == [11, 11, 10]
    device = torch.device("cpu" if args.cpu_check else f"npu:{args.device}")
    if not args.cpu_check:
        # Register the real vendor backend only in the explicit NPU mode.
        # Dependency import remains at module scope through CLI entry below.
        torch.npu.set_device(device)
    hf = Qwen3_5MoeTextRotaryEmbedding(config, device=device)
    production = None
    if args.production_adapter:
        production_config = copy.deepcopy(config)
        production_config.rotary_compute_dtype = "float32"
        production_config.rotary_cache_dtype = "float32"
        production = Qwen3_5MoeTextRotaryEmbedding(production_config, device=device)
    # Runtime reference init_states rebuilds this cache in FP32 after to_empty.
    # The cache length is bounded to this probe's position domain, not a model
    # geometry change; frequencies and application are the original methods.
    with torch.device(device):
        ref = MRoPE.Config(
            dim=64, max_seq_len=512, theta=10_000_000, mrope_section=[11, 11, 10]
        ).build()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def sample(heads):
        return (
            (
                torch.randn((1, 128, heads, 256), generator=generator)
                + torch.linspace(-0.3, 0.5, 256)
            )
            .bfloat16()
            .to(device)
        )

    q, k, uq, uk = sample(16), sample(2), sample(16), sample(2)
    positions = torch.arange(128, device=device)[None, :]
    report.update(
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        sources={
            "hf": source(Qwen3_5MoeTextRotaryEmbedding),
            "hf_apply": source(apply_rotary_pos_emb),
            "reference": source(MRoPE),
            "reference_apply": source(MRoPE.apply_rotary_emb),
        },
        geometry={
            "q": list(q.shape),
            "k": list(k.shape),
            "rotary_dim": 64,
            "unrotated_dim": 192,
            "mrope_section": [11, 11, 10],
        },
        cache={
            "hf_inv_freq_dtype": str(hf.inv_freq.dtype),
            "reference_cache_dtype": str(ref.cache.dtype),
        },
        cases={},
    )
    if production is not None:
        report["sources"].update(
            production_cache=source(forward_qwen35_fp32_cache),
            production_apply=source(apply_qwen35_fp32_rotary),
        )
        report["production_options"] = {
            "rotary_compute_dtype": "float32",
            "rotary_cache_dtype": "float32",
            "scope": (
                "Actual production cache/application helpers; not installer acceptance"
            ),
        }
    cases = {"text": positions}
    if args.mrope:
        cases["mrope"] = torch.stack(
            (positions, positions * 2 + 1, positions * 3 + 2), dim=-1
        )
    for name, pos in cases.items():
        hf_pos = pos if pos.ndim == 2 else pos.permute(2, 0, 1)
        cos, sin = hf(q, hf_pos)
        cache = ref._reshape_cache(q[..., :64], pos)
        rc, rs = cache[..., :64].squeeze(2), cache[..., 64:].squeeze(2)
        assert cos.shape == sin.shape == (1, 128, 64)

        def original_hf(a, b, cos=cos, sin=sin):
            return apply_rotary_pos_emb(a, b, cos, sin, unsqueeze_dim=2)

        def original_ref(a, b, pos=pos):
            ar, br = ref(a[..., :64], b[..., :64], pos)
            return torch.cat((ar, a[..., 64:]), -1), torch.cat((br, b[..., 64:]), -1)

        # Diagnostic only: original HF function with FP32 inputs and FP32
        # reference cache, followed by one BF16 cast. No production changes.
        def fp32_application(a, b, rc=rc, rs=rs):
            ar, br = apply_rotary_pos_emb(a.float(), b.float(), rc, rs, unsqueeze_dim=2)
            return ar.to(a.dtype), br.to(b.dtype)

        a = vjp(q, k, uq, uk, original_hf)
        b = vjp(q, k, uq, uk, original_ref)
        c = vjp(q, k, uq, uk, fp32_application)
        results = [a, b, c]
        production_case = None
        if production is not None:
            pc, ps = forward_qwen35_fp32_cache(production, q, hf_pos)

            def production_application(a, b, pc=pc, ps=ps):
                return apply_qwen35_fp32_rotary(a, b, pc, ps, unsqueeze_dim=2)

            d = vjp(q, k, uq, uk, production_application)
            results.append(d)
            production_case = {
                "cos_dtype": str(pc.dtype),
                "sin_dtype": str(ps.dtype),
                "cache_cos": compare(pc, rc),
                "cache_sin": compare(ps, rs),
                **{key: compare(d[key][..., :64], b[key][..., :64]) for key in d},
            }
            assert pc.dtype == ps.dtype == torch.float32
        for result in results:
            for key, expected in (("q", q), ("k", k), ("dq", uq), ("dk", uk)):
                assert torch.equal(result[key][..., 64:], expected[..., 64:]), key
        report["cases"][name] = {
            "positions_shape": list(pos.shape),
            "hf_cos_dtype": str(cos.dtype),
            "reference_cos_dtype": str(rc.dtype),
            "cache_cos": compare(cos, rc),
            "cache_sin": compare(sin, rs),
            "native": {key: compare(a[key][..., :64], b[key][..., :64]) for key in a},
            "fp32_shared_cache_diagnostic": {
                key: compare(c[key][..., :64], b[key][..., :64]) for key in c
            },
            "unrotated_forward_and_vjp_exact": True,
        }
        if production_case is not None:
            report["cases"][name]["production_adapter"] = production_case
    report["native_numerically_passed"] = all(
        metric["passed"]
        for case in report["cases"].values()
        for metric in case["native"].values()
    )
    report["exit_criterion"] = (
        "production_adapter: cache cos/sin and q/k/dq/dk all pass; unrotated exact"
        if args.production_adapter
        else "native q/k/dq/dk all pass; unrotated exact"
    )
    report["numerically_passed"] = all(
        metric["passed"]
        for case in report["cases"].values()
        for metric in case[
            "production_adapter" if args.production_adapter else "native"
        ].values()
        if isinstance(metric, dict)
    )
    report["status"] = (
        "cpu_geometry_and_vjp_only"
        if args.cpu_check
        else ("passed" if report["numerically_passed"] else "failed")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mrope", action="store_true")
    parser.add_argument(
        "--production-adapter",
        action="store_true",
        help="Evaluate actual FP32 production helpers; retain native evidence, "
        "but require all production cache/output/VJP checks to pass",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "error",
        "atol": ATOL,
        "rtol": RTOL,
        "scope": "Original partial RoPE only; no model/training acceptance",
        "seed": args.seed,
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if report["status"] == "failed" or (
        args.production_adapter and not report.get("numerically_passed", False)
    ):
        raise SystemExit(1)


# Opt-in native backend registration; CPU geometry checks never import it.
if __name__ == "__main__":
    main()
