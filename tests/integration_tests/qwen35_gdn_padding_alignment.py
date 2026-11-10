"""Research-only B1 right-padding GDN eager/full-AC VJP diagnostic.

HF receives its real 2D mask (HF5.13 B1 does not zero hidden states).
Reference receives None: one document, no reset. Default research mode does not
install guards. --production-adapters invokes real production installation and
retains owned validation hooks. Neither mode establishes FSDP/Trainer acceptance.
"""

from __future__ import annotations

# ruff: noqa: I001
# Preserve native-HF-before-reference compatibility import order.
import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    apply_mask_to_padding_states,
)
from tests.integration_tests.qwen35_gdn_module_alignment import (
    ADAPTER,
    REFERENCE,
    SHAPES,
    OverrideConfig,
    Qwen35StateDictAdapter,
    apply_overrides,
    callable_source,
    compare,
    file_hash,
    gated_delta_rule,
    load_weights,
    model_registry,
    set_spmd_backend,
)
from llmtuner.models.qwen3_5.gdn_numerics import (
    _validate_forward,
    configure_gdn_numerics,
    forward_fp32_weight_norm,
    forward_split_qkv,
)

from llmtuner.models.qwen3_5.linear_attention import (
    apply_linear_attention_backend,
    chunk_gated_delta_rule,
    validate_gdn_inputs,
)
from llmtuner.models.hf.padding_metadata import validate_gdn_padding_metadata


def verify_hooks(module, production):
    assert not any(m._forward_hooks for m in module.modules())
    for child in module.modules():
        if child is not module:
            assert not child._forward_pre_hooks
    expected = {_validate_forward, validate_gdn_inputs} if production else set()
    hooks = tuple(module._forward_pre_hooks.values())
    assert len(hooks) == len(expected) and set(hooks) == expected


def research_chunk(q, k, v, *, g, beta, **kwargs):
    assert kwargs.pop("use_qk_l2norm_in_kernel", False)
    assert kwargs.pop("initial_state", None) is None
    assert not kwargs.pop("output_final_state", False)
    assert kwargs.pop("cu_seqlens", None) is None
    assert not kwargs, kwargs
    return gated_delta_rule(q, k, v, g, beta), None


def finite(value):
    assert value.numel() and torch.isfinite(value).all(), "empty/nonfinite evidence"
    return value.detach().cpu().clone()


def evaluate(
    module, state, x_cpu, dy_cpu, mask_cpu, device, *, hf, ac, production=False
):
    module.to_empty(device=device)
    module.to(dtype=torch.bfloat16)
    module.load_state_dict(state, strict=True)
    module.train()
    for key, value in module.state_dict().items():
        assert torch.equal(value.detach().cpu(), state[key]), key
    assert all(p.dtype == torch.bfloat16 for p in module.parameters())
    verify_hooks(module, production=production)
    x = x_cpu.to(device).detach().requires_grad_(True)
    mask = mask_cpu.to(device)

    def forward(value):
        return module(value, attention_mask=mask) if hf else module(value, None)

    output = checkpoint(forward, x, use_reentrant=False) if ac else forward(x)
    output.backward(dy_cpu.to(device))
    assert x.grad is not None
    gradients = {}
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        gradients[name] = finite(parameter.grad)
    result = {"output": finite(output), "dx": finite(x.grad), "parameters": gradients}
    module.zero_grad(set_to_none=True)
    module.to("cpu")
    return result


def comparison(a, b, valid):
    assert a["parameters"].keys() == b["parameters"].keys()
    return {
        "valid_output": compare(a["output"][:, :valid], b["output"][:, :valid]),
        "dx": compare(a["dx"][:, :valid], b["dx"][:, :valid]),
        "parameters": {
            k: compare(v, b["parameters"][k]) for k, v in a["parameters"].items()
        },
        "parameter_exact": {
            k: torch.equal(v, b["parameters"][k]) for k, v in a["parameters"].items()
        },
        "full_output_exact": torch.equal(a["output"], b["output"]),
        "valid_output_exact": torch.equal(
            a["output"][:, :valid], b["output"][:, :valid]
        ),
        "dx_exact": torch.equal(a["dx"], b["dx"]),
    }


def all_passed(metrics):
    return (
        metrics["valid_output"]["passed"]
        and metrics["dx"]["passed"]
        and all(value["passed"] for value in metrics["parameters"].values())
    )


def run(args, report):
    assert args.batch_size == 1, (
        "B2 needs separately validated varlen row-reset semantics"
    )
    assert 0 < args.valid_tokens < 128
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert config.hidden_size == 2048 and config.num_hidden_layers == 40
    set_spmd_backend("partial_dtensor")
    spec = model_registry("35B-A3B", attn_backend="flex")
    assert apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.qwen3_5.gated_delta.npu"]),
        spec.model,
    )
    with torch.device("meta"):
        hf = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(torch.bfloat16)
        ref = spec.model.layers[0].delta_net.build().to(torch.bfloat16)
    assert {k: tuple(v.shape) for k, v in hf.state_dict().items()} == SHAPES
    before = dict(hf.named_parameters())
    keys_before = set(hf.state_dict())
    if args.production_adapters:
        config.linear_attention_backend = "npu_triton"
        config.gdn_numerical_policy = "split_qkv_fp32_norm"
        config.gdn_allow_right_padding = True
        config.qkv_projection = "separate"
        config.norm_backend = "native"
        holder = torch.nn.Module()
        holder.config = config
        holder.add_module("linear_attn", hf)
        configure_gdn_numerics(holder)
        apply_linear_attention_backend(holder)
        assert hf.chunk_gated_delta_rule is chunk_gated_delta_rule
        assert hf._llmtuner_gdn_allow_right_padding is True
    else:
        # Preserve the original explicit research math/recurrence assembly.
        hf.in_proj_qkv.forward = MethodType(forward_split_qkv, hf.in_proj_qkv)
        hf.norm.forward = MethodType(forward_fp32_weight_norm, hf.norm)
        hf.chunk_gated_delta_rule = research_chunk
    after = dict(hf.named_parameters())
    assert after.keys() == before.keys()
    assert all(after[k] is p and p.dtype == torch.bfloat16 for k, p in before.items())
    assert set(hf.state_dict()) == keys_before
    verify_hooks(hf, args.production_adapters)
    verify_hooks(ref, False)
    report["installation"] = {
        "mode": "production-adapters" if args.production_adapters else "research",
        "parameter_identity_preserved": True,
        "state_keys_preserved": True,
        "parameters_bfloat16": True,
        "owned_validation_hooks": [
            f.__qualname__ for f in hf._forward_pre_hooks.values()
        ],
        "no_observation_hooks": True,
    }
    adapter = Qwen35StateDictAdapter(spec.model, None)
    states = load_weights(args, adapter, report)
    report["sources"] = {
        "script_sha256": file_hash(Path(__file__)),
        "hf": callable_source(type(hf)),
        "reference": callable_source(type(ref)),
        "norm_math": callable_source(forward_fp32_weight_norm),
        "qkv_math": callable_source(forward_split_qkv),
        "recurrence": callable_source(gated_delta_rule),
        "hf_mask": callable_source(apply_mask_to_padding_states),
        "checkpoint": callable_source(checkpoint),
        "configure_gdn_numerics": callable_source(configure_gdn_numerics),
        "apply_linear_attention_backend": callable_source(
            apply_linear_attention_backend
        ),
        "padding_metadata": callable_source(validate_gdn_padding_metadata),
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn((1, 128, 2048), generator=generator) * 0.7
        + torch.linspace(-0.3, 0.5, 2048)[None, None, :]
    ).bfloat16()
    dy = torch.randn(x.shape, generator=generator).bfloat16()
    dy[:, args.valid_tokens :] = 0
    mask = torch.arange(128)[None, :] < args.valid_tokens
    assert torch.equal(apply_mask_to_padding_states(x, mask), x)
    changed = x.clone()
    changed[:, args.valid_tokens :] = (
        changed[:, args.valid_tokens :].float() * -2 + 0.3
    ).bfloat16()
    torch.save(
        {
            "x": x,
            "changed_padding_x": changed,
            "dy": dy,
            "attention_mask": mask,
            "positions": torch.arange(128)[None, :],
        },
        args.output / "inputs.pt",
    )
    if args.cpu_check:
        report["status"] = "CPU_META_ONLY_NOT_NUMERICAL_ACCEPTANCE"
        return
    assert args.execute
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    results = {}
    for side, module, state in (("hf", hf, states[0]), ("reference", ref, states[1])):
        results[side] = {}
        for mode, ac, inputs in (
            ("eager", False, x),
            ("full_ac", True, x),
            ("changed_padding", False, changed),
        ):
            result = evaluate(
                module,
                state,
                inputs,
                dy,
                mask,
                device,
                hf=side == "hf",
                ac=ac,
                production=side == "hf" and args.production_adapters,
            )
            if side == "reference":
                converted = adapter.to_hf(
                    {REFERENCE + k: v for k, v in result["parameters"].items()}
                )
                result["parameters"] = {
                    k.removeprefix(ADAPTER): v for k, v in converted.items()
                }
            results[side][mode] = result
            torch.save(result, args.output / f"{side}-{mode}.pt")
    metrics = {
        "cross_eager": comparison(
            results["hf"]["eager"], results["reference"]["eager"], args.valid_tokens
        ),
        "cross_full_ac": comparison(
            results["hf"]["full_ac"], results["reference"]["full_ac"], args.valid_tokens
        ),
    }
    for side in results:
        for mode in ("full_ac", "changed_padding"):
            metrics[side + ":" + mode] = comparison(
                results[side][mode], results[side]["eager"], args.valid_tokens
            )
    leakage = {
        side + ":" + mode: {
            "padded_dx_nonzero": int(r["dx"][:, args.valid_tokens :].count_nonzero()),
            "dx_nonzero": bool(r["dx"].count_nonzero()),
            "all_parameter_gradients_nonzero": all(
                bool(v.count_nonzero()) for v in r["parameters"].values()
            ),
        }
        for side, modes in results.items()
        for mode, r in modes.items()
    }
    report.update(comparisons=metrics, leakage=leakage)
    report["passed"] = all(all_passed(m) for m in metrics.values()) and all(
        v["padded_dx_nonzero"] == 0
        and v["dx_nonzero"]
        and v["all_parameter_gradients_nonzero"]
        for v in leakage.values()
    )
    report["status"] = "PASS" if report["passed"] else "NUMERICAL_MISMATCH"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--valid-tokens", type=int, default=93)
    parser.add_argument("--batch-size", type=int, choices=[1], default=1)
    parser.add_argument(
        "--production-adapters",
        action="store_true",
        help="Install real npu_triton + split_qkv_fp32_norm with right-padding "
        "opt-in and owned guard hooks; default keeps research assembly",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-check", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "ERROR",
        "production_adapters": args.production_adapters,
        "atol": 0.005,
        "rtol": 0.02,
        "scope": (
            "Research B1 full GDN right-padding mask/eager/full AC; "
            "not production support/FSDP/Trainer"
        ),
        "mask_contract": (
            "HF real 2D mask, B1 leaves input nonzero; "
            "reference None single document no reset; positions0..127"
        ),
        "guard": (
            "Explicit production adapters retain their owned validators; "
            "default research mode does not install guards. Single-module "
            "results cannot establish masked FSDP/Trainer acceptance."
        ),
    }
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
