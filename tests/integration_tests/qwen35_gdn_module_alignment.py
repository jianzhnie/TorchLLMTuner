"""Full 35B GDN module VJP diagnostic, not full-model/100-step acceptance.

In the existing reference environment (HF must import before NPU compatibility
patches)::

    python tests/integration_tests/qwen35_gdn_module_alignment.py \
      --model /original/HF --output /scratch/gdn.json --execute

--cpu-check validates meta geometry and the real adapter roundtrip on CPU only.
No tokenizer, optimizer, distributed runtime, or reduced geometry is involved.
"""

from __future__ import annotations

# ruff: noqa: I001
# Deliberate top-level order: native HF resolves its optional backends before
# the reference package installs its existing FLA import compatibility helpers.
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)
import torch_npu  # noqa: F401 -- register backend; no device operation at import
import torchtitan_npu  # noqa: F401 -- existing reference patches
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.qwen3_5.model import GatedDeltaNet
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan_npu.models.qwen3_5 import model_registry
from torchtitan_npu.override.qwen3_5.gated_delta import (
    TritonGatedDeltaKernel,
    _npu_causal_conv,
)
from llmtuner.accelerator.gated_delta import gated_delta_rule
from llmtuner.models.qwen.qwen3_5.linear_attention import apply_linear_attention_backend

ATOL = 0.005
RTOL = 0.02
NATIVE = "model.layers.0.linear_attn."
ADAPTER = "model.language_model.layers.0.linear_attn."
REFERENCE = "layers.0.attn."
SHAPES = {
    "A_log": (32,),
    "dt_bias": (32,),
    "conv1d.weight": (8192, 1, 4),
    "norm.weight": (128,),
    "out_proj.weight": (2048, 4096),
    "in_proj_qkv.weight": (8192, 2048),
    "in_proj_z.weight": (4096, 2048),
    "in_proj_a.weight": (32, 2048),
    "in_proj_b.weight": (32, 2048),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def callable_source(value) -> dict:
    path = Path(inspect.getfile(value)).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def validate(value: torch.Tensor) -> dict:
    value = value.detach().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nonempty": value.numel() > 0,
        "finite": bool(torch.isfinite(value).all()),
        "nonzero": bool(torch.count_nonzero(value)),
    }


def compare(hf: torch.Tensor, reference: torch.Tensor) -> dict:
    info = {"hf": validate(hf), "reference": validate(reference)}
    valid = all(
        side[key] for side in info.values() for key in ("nonempty", "finite", "nonzero")
    )
    info["passed"] = False
    if hf.shape != reference.shape or not valid:
        return info
    a, b = hf.detach().float().cpu(), reference.detach().float().cpu()
    error = (a - b).abs()
    failed = error > ATOL + RTOL * a.abs()
    index = int(error.reshape(-1).argmax())
    info.update(
        max_absolute_error=float(error.max()),
        rms_error=float(error.square().mean().sqrt()),
        failed_elements=int(failed.sum()),
        elements=a.numel(),
        max_error_flat_index=index,
        hf_at_max_error=float(a.reshape(-1)[index]),
        reference_at_max_error=float(b.reshape(-1)[index]),
        passed=not bool(failed.any()),
    )
    return info


def load_weights(args, adapter, report) -> tuple[dict, dict]:
    manifest_path = args.artifacts / "initial.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config_sha256"] == file_hash(args.model / "config.json")
    assert manifest["original_step"] == 0 and manifest["resume"] is False
    hashes = {}
    for name in ("initial-0.pt", "initial-outer.pt"):
        hashes[name] = file_hash(args.artifacts / name)
        assert hashes[name] == manifest["files"][name], f"Hash mismatch: {name}"
    layer = torch.load(
        args.artifacts / "initial-0.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    hf = {
        key[len(NATIVE) :]: value
        for key, value in layer.items()
        if key.startswith(NATIVE)
    }
    assert set(hf) == set(SHAPES), sorted(hf)
    for key, value in hf.items():
        assert tuple(value.shape) == SHAPES[key], key
        assert value.dtype == torch.bfloat16, key
        assert bool(torch.isfinite(value).all()), f"Nonfinite initial {key}"
    # The original adapter insists on lm_head even for partial layer groups.
    # Supply the REAL mmap-backed source head, then discard its mapping.
    outer = torch.load(
        args.artifacts / "initial-outer.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    group = {ADAPTER + key: value for key, value in hf.items()}
    group["lm_head.weight"] = outer["lm_head.weight"]
    converted = adapter.from_hf(group)
    del converted["lm_head.weight"]
    assert all(key.startswith(REFERENCE) for key in converted)
    reference = {key[len(REFERENCE) :]: value for key, value in converted.items()}
    roundtrip = adapter.to_hf(converted)
    assert set(roundtrip) == {ADAPTER + key for key in hf}
    for key, value in hf.items():
        assert torch.equal(value, roundtrip[ADAPTER + key]), key
    report["initialization"] = {
        "manifest": manifest,
        "manifest_sha256": file_hash(manifest_path),
        "verified_files": hashes,
        "adapter_roundtrip_exact": True,
        "head_guard": "real initial-outer.pt mmap head; discarded before module load",
    }
    return hf, reference


def run_side(
    module, weights, inputs, upstream, device, *, fp32_qk=False, shared_kernel=None
) -> dict:
    module.to_empty(device=device)
    module.to(dtype=torch.bfloat16)
    module.load_state_dict(weights, strict=True)
    module.train()
    for name, value in module.state_dict().items():
        assert torch.equal(value.detach().cpu(), weights[name]), name
    x = inputs.to(device).detach().requires_grad_(True)
    traces = {}

    def capture(name, tensor):
        assert name not in traces, f"Repeated observation: {name}"
        traces[name] = tensor.detach().cpu().clone()
        if tensor.requires_grad:

            def backward(gradient):
                traces[name + ".grad"] = gradient.detach().cpu().clone()

            tensor.register_hook(backward)

    def norm_before(_module, args):
        capture("norm.input", args[0])
        capture("norm.gate", args[1])

    def norm_after(_module, _args, output):
        capture("norm.output", output)

    handles = [
        module.norm.register_forward_pre_hook(norm_before),
        module.norm.register_forward_hook(norm_after),
    ]
    original_chunk = None
    if isinstance(module, Qwen3_5MoeGatedDeltaNet):
        original_chunk = module.chunk_gated_delta_rule

        def observed_chunk(*args, **kwargs):
            if shared_kernel is not None:
                assert kwargs["use_qk_l2norm_in_kernel"]
                assert kwargs.get("initial_state") is None
                assert not kwargs.get("output_final_state", False)
                output = shared_kernel(*args[:3], kwargs["g"], kwargs["beta"])
                capture("kernel.output", output)
                return output, None
            if fp32_qk:
                # Diagnostic intervention only: preserve the reference's
                # FP32 normalize -> BF16 boundary, using ordinary autograd.
                # Its hand-written VJP may still differ from this cast graph.
                assert kwargs["use_qk_l2norm_in_kernel"]
                normalized = []
                for tensor in args[:2]:
                    value = tensor.float()
                    inverse = torch.rsqrt((value * value).sum(-1, True) + 1e-6)
                    normalized.append((value * inverse).to(tensor.dtype))
                args = (*normalized, *args[2:])
                kwargs["use_qk_l2norm_in_kernel"] = False
            output = original_chunk(*args, **kwargs)
            capture("kernel.output", output[0])
            return output

        module.chunk_gated_delta_rule = observed_chunk
    else:

        def kernel_after(_module, _args, output):
            capture("kernel.output", output)

        handles.append(module.kernel.register_forward_hook(kernel_after))
    try:
        output = module(x)
        assert isinstance(output, torch.Tensor) and output.shape == inputs.shape
        output.backward(upstream.to(device))
    finally:
        for handle in handles:
            handle.remove()
        if original_chunk is not None:
            module.chunk_gated_delta_rule = original_chunk
    assert x.grad is not None
    grads = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        grads[name] = parameter.grad.detach().cpu().clone()
    result = {
        "output": output.detach().cpu(),
        "dx": x.grad.detach().cpu(),
        "paramgrad": grads,
        "traces": traces,
    }
    module.to("cpu")
    return result


def norm_cross_injection(hf, reference, observations, device, seed) -> dict:
    """Call original norm modules on identical observed activations and VJPs."""
    results = {}
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    upstream = torch.randn((4096, 128), generator=generator).to(torch.bfloat16)
    for source, observation in observations.items():
        compared = []
        for module, shape in (
            (hf.norm, (-1, 128)),
            (reference.norm, (1, 128, 32, 128)),
        ):
            module.to(device)
            module.zero_grad(set_to_none=True)
            x = observation["norm.input"].reshape(shape).to(device)
            z = observation["norm.gate"].reshape(shape).to(device)
            x = x.detach().requires_grad_(True)
            z = z.detach().requires_grad_(True)
            output = module(x, z)
            output.backward(upstream.reshape(shape).to(device))
            assert x.grad is not None and z.grad is not None
            assert module.weight.grad is not None
            compared.append(
                {
                    "output": output.detach().reshape(-1, 128).cpu(),
                    "dx": x.grad.detach().reshape(-1, 128).cpu(),
                    "dz": z.grad.detach().reshape(-1, 128).cpu(),
                    "weightgrad": module.weight.grad.detach().cpu().clone(),
                }
            )
            module.to("cpu")
        results[source] = {
            key: compare(compared[0][key], compared[1][key]) for key in compared[0]
        }
    return results


def diagnose(args, report) -> None:
    assert importlib.metadata.version("transformers") == "5.13.0"
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    expected = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    for key, value in expected.items():
        assert getattr(config, key) == value, key
    set_spmd_backend("partial_dtensor")
    spec = model_registry("35B-A3B", attn_backend="flex")
    assert len(spec.model.layers) == 40
    applied = apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.qwen3_5.gated_delta.npu"]),
        spec.model,
    )
    assert applied, "Original NPU override did not apply"
    with torch.device("meta"):
        hf = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
        reference = spec.model.layers[0].delta_net.build()
    if args.production_adapter:
        # Route through the same instance adaptation and guards as the Trainer.
        # This remains an isolated module test, not distributed acceptance.
        config.linear_attention_backend = "npu_triton"
        holder = torch.nn.Module()
        holder.config = config
        holder.add_module("linear_attn", hf)
        apply_linear_attention_backend(holder)
    assert type(reference) is GatedDeltaNet
    assert isinstance(reference.kernel, TritonGatedDeltaKernel)
    assert GatedDeltaNet._causal_conv is _npu_causal_conv
    assert {key: tuple(value.shape) for key, value in hf.state_dict().items()} == SHAPES
    adapter = Qwen35StateDictAdapter(spec.model, None)
    hf_weights, reference_weights = load_weights(args, adapter, report)
    assert set(reference.state_dict()) == set(reference_weights)
    for key, value in reference.state_dict().items():
        assert value.shape == reference_weights[key].shape, key
    report["geometry"] = expected
    report["parameter_count"] = sum(value.numel() for value in hf_weights.values())
    report["sources"] = {
        "hf": callable_source(Qwen3_5MoeGatedDeltaNet),
        "reference": callable_source(GatedDeltaNet),
        "adapter": callable_source(Qwen35StateDictAdapter),
        "npu_kernel": callable_source(TritonGatedDeltaKernel),
        "npu_conv": callable_source(_npu_causal_conv),
    }
    report["hf_backends"] = {
        "chunk": str(hf.chunk_gated_delta_rule),
        "causal_conv": str(hf.causal_conv1d_fn),
        "norm": type(hf.norm).__module__ + "." + type(hf.norm).__name__,
    }
    if args.cpu_check:
        report["status"] = "cpu_layout_only"
        return
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (1, 128, 2048)
    inputs = (
        torch.randn(shape, generator=generator) * 0.7
        + torch.linspace(-0.3, 0.5, 2048)[None, None, :]
        + torch.linspace(-0.2, 0.1, 128)[None, :, None]
    ).to(torch.bfloat16)
    upstream = torch.randn(shape, generator=generator).to(torch.bfloat16)
    if args.valid_tokens is not None:
        # Emulate right-padding excluded from a causal training objective.
        # Keep padded inputs nonzero: masking all inputs would hide leakage.
        upstream[:, args.valid_tokens :] = 0
        report["valid_tokens"] = args.valid_tokens
        report["padding_scope"] = (
            "Nonzero right-padded activations; padded output cotangents zero"
        )
    report["input"] = validate(inputs)
    report["cotangent"] = validate(upstream)
    a = run_side(
        hf,
        hf_weights,
        inputs,
        upstream,
        device,
        fp32_qk=args.fp32_qk,
        shared_kernel=(
            gated_delta_rule
            if args.native_npu_kernel
            else reference.kernel
            if args.shared_kernel
            else None
        ),
    )
    b = run_side(reference, reference_weights, inputs, upstream, device)
    native_grads = adapter.to_hf(
        {REFERENCE + key: value for key, value in b["paramgrad"].items()}
    )
    assert set(native_grads) == {ADAPTER + key for key in a["paramgrad"]}
    comparisons = {key: compare(a[key], b[key]) for key in ("output", "dx")}
    comparisons.update(
        {
            NATIVE + key: compare(value, native_grads[ADAPTER + key])
            for key, value in a["paramgrad"].items()
        }
    )
    report["comparisons"] = comparisons
    assert set(a["traces"]) == set(b["traces"])
    report["stage_comparisons"] = {
        key: compare(
            a["traces"][key].reshape(-1, 128), b["traces"][key].reshape(-1, 128)
        )
        for key in a["traces"]
    }
    report["norm_cross_injection"] = norm_cross_injection(
        hf,
        reference,
        {"hf_activation": a["traces"], "reference_activation": b["traces"]},
        device,
        args.seed,
    )
    report["failures"] = [
        key for key, value in comparisons.items() if not value["passed"]
    ]
    report["status"] = "failed" if report["failures"] else "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Original HF directory whose config hash is in initial.json",
    )
    parser.add_argument(
        "--artifacts", type=Path, default=Path("/tmp/tlt_qwen35_cross_framework_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-tokens", type=int)
    interventions = parser.add_mutually_exclusive_group()
    interventions.add_argument(
        "--production-adapter",
        action="store_true",
        help="Apply the production HF instance adapter and its input guards",
    )
    interventions.add_argument(
        "--native-npu-kernel",
        action="store_true",
        help="Use the independent llmtuner NPU recurrence inside the native HF module",
    )
    interventions.add_argument(
        "--fp32-qk",
        action="store_true",
        help="Diagnostic intervention: FP32 HF QK normalization, not production",
    )
    interventions.add_argument(
        "--shared-kernel",
        action="store_true",
        help="Diagnostic intervention: use original reference recurrence for HF",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-check", action="store_true")
    modes.add_argument("--execute", action="store_true", help="Explicit real NPU run")
    args = parser.parse_args()
    if args.valid_tokens is not None and not 0 < args.valid_tokens < 128:
        raise ValueError("valid-tokens must be strictly between 0 and 128")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "error",
        "scope": "Full published GDN module VJP only",
        "not_acceptance": (
            "Not full decoder/model, LIMO, or 100-step training acceptance"
        ),
        "atol": ATOL,
        "rtol": RTOL,
        "comparison_rule": "abs(reference-HF) <= atol + rtol*abs(HF)",
        "seed": args.seed,
        "intervention": (
            "HF uses production instance adapter"
            if args.production_adapter
            else "HF uses independent llmtuner NPU recurrence"
            if args.native_npu_kernel
            else "HF uses reference recurrence"
            if args.shared_kernel
            else "HF FP32 QK normalize"
            if args.fp32_qk
            else "none"
        ),
        "stage_trace": "Original kernel output and norm input/gate/output VJPs",
    }
    try:
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "transformers")
        }
        diagnose(args, report)
    except BaseException as exc:
        report.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        # Reporting must not replace an original execution exception.
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
