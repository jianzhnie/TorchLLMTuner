"""Compare FP32 RoPE backward with the active NPU rotary kernel on real Q/K."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from llmtuner.models.hf.qkv_projection import apply_configured_rotary
from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    import torch_npu

    torch.npu.set_device(0)
    set_deterministic_algorithms()
    torch.manual_seed(37)
    trace = torch.load(args.trace, weights_only=True, mmap=True)
    module = SimpleNamespace(
        config=SimpleNamespace(
            rotary_compute_dtype="float32",
            rotary_cache_dtype="bfloat16",
            rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
        )
    )
    report = []
    for name in ("layer0.q_norm", "layer0.k_norm"):
        inputs = trace[name].to("npu")
        reference_input = inputs.detach().clone().requires_grad_()
        actual_input = inputs.detach().clone().requires_grad_()
        positions = torch.arange(inputs.shape[1], device="npu")[None]
        frequencies = 1.0 / (
            1000000.0
            ** (
                torch.arange(0, inputs.shape[-1], 2, device="npu").float()
                / inputs.shape[-1]
            )
        )
        angles = positions.float().unsqueeze(-1) * frequencies
        angles = torch.cat((angles, angles), -1).unsqueeze(2)
        cosine, sine = angles.cos().bfloat16().float(), angles.sin().bfloat16().float()
        reference = torch_npu.npu_rotary_mul(reference_input.float(), cosine, sine).to(
            inputs.dtype
        )
        actual, _ = apply_configured_rotary(
            module,
            actual_input.transpose(1, 2),
            actual_input.transpose(1, 2),
            None,
            None,
            positions,
        )
        actual = actual.transpose(1, 2)
        upstream = torch.randn_like(reference)
        expected_gradient = torch.autograd.grad(reference, reference_input, upstream)[0]
        actual_gradient = torch.autograd.grad(actual, actual_input, upstream)[0]
        result = {"name": name}
        for key, observed, expected in (
            ("output", actual, reference),
            ("input_gradient", actual_gradient, expected_gradient),
        ):
            delta = (observed.float() - expected.float()).abs()
            result[key] = {
                "exact": torch.equal(observed, expected),
                "max_ae": delta.max().item(),
                "mean_ae": delta.mean().item(),
            }
        report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert all(
        row[key]["exact"] for row in report for key in ("output", "input_gradient")
    )


if __name__ == "__main__":
    main()
