"""Audit HF 5.13 text layouts against TorchTitan v0.3.0 without allocating weights.

Run in the isolated reference environment with TorchTitan-NPU on PYTHONPATH.
This validates only state keys and shapes, never numerical equivalence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchtitan_npu.models.qwen3_5 as reference
from transformers import AutoConfig, AutoModelForCausalLM


def audit(model_path: str) -> dict:
    spec = reference.model_registry("35B-A3B", attn_backend="flex")
    config = spec.model
    for layer in config.layers:
        # Native HF routing has no auxiliary-free expert bias updates.
        layer.moe.load_balance_coeff = None
    with torch.device("meta"):
        model = config.build()
        hf = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_path).text_config
        )
    adapter = spec.state_dict_adapter(config, None)
    # Non-square, non-symmetric values expose transpose or gate/up swaps that
    # shape-only meta checks cannot catch. This is still not a forward test.
    witness = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.arange(
            60, dtype=torch.float32
        ).reshape(2, 6, 5),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.arange(
            30, dtype=torch.float32
        ).reshape(2, 5, 3),
        "lm_head.weight": torch.ones(1, 1),
    }
    legacy = {
        key: value.transpose(-1, -2) if ".experts." in key else value
        for key, value in witness.items()
    }
    restored = adapter.to_hf(adapter.from_hf(legacy))
    assert restored.keys() == legacy.keys(), "expert roundtrip keys differ"
    for key, value in legacy.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    state = {}
    for key, value in hf.state_dict().items():
        name = "model.language_model." + key[6:] if key.startswith("model.") else key
        # v0.3.0's adapter expects [E,D,2F]/[E,F,D], while HF 5.13
        # stores [E,2F,D]/[E,D,F]. Convert at this reference-only boundary.
        if key.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            value = value.transpose(-1, -2)
        state[name] = value
    converted = adapter.from_hf(state)
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("vision_encoder.")
    }
    return {
        "standard": "meta keys and shapes only; no numerical acceptance",
        "missing": sorted(expected.keys() - converted.keys()),
        "extra": sorted(converted.keys() - expected.keys()),
        "shape_mismatches": [
            (key, list(expected[key].shape), list(converted[key].shape))
            for key in sorted(expected.keys() & converted.keys())
            if expected[key].shape != converted[key].shape
        ],
        "reference_text_parameters": sum(
            parameter.numel()
            for key, parameter in model.named_parameters()
            if not key.startswith("vision_encoder.")
        ),
        "hf_parameters": sum(parameter.numel() for parameter in hf.parameters()),
        "tensor_keys": len(expected),
        "expert_adapter_roundtrip": "finite exact values on asymmetric matrices",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if (
        result["missing"]
        or result["extra"]
        or result["shape_mismatches"]
        or result["reference_text_parameters"] != result["hf_parameters"]
        or not result["tensor_keys"]
    ):
        raise SystemExit("Qwen3.5 reference text layout differs")


if __name__ == "__main__":
    main()
