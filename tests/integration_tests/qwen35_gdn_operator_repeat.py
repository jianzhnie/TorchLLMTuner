"""Fixed-input actual reference GDN repeats, independent of outer autograd merges.

No production/reference kernel edits. Saves original inputs and each output plus
dq/dk/dv/dg/dbeta. Exit zero means collection, not numerical acceptance.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

import torch

from tests.integration_tests.qwen35_decoder_layer_trace import descriptor, file_hash
from tests.integration_tests.qwen35_moe_module_trace import compare_branch


def fixed_inputs(tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    key_shape, value_shape = (1, tokens, 16, 128), (1, tokens, 32, 128)
    q, k = [torch.randn(key_shape, generator=generator).bfloat16() for _ in range(2)]
    v = torch.randn(value_shape, generator=generator).bfloat16()
    g = -torch.rand(value_shape[:3], generator=generator)
    beta = torch.rand(value_shape[:3], generator=generator).bfloat16()
    dy = torch.randn(value_shape, generator=generator).bfloat16()
    return dict(q=q, k=k, v=v, g=g, beta=beta, dy=dy)


def evaluate(kernel, inputs):
    # External leaf clones only; no projections, norms, branch clones or hooks.
    names = ("q", "k", "v", "g", "beta")
    leaves = [inputs[name].detach().clone().requires_grad_() for name in names]
    output = kernel(*leaves)
    grads = torch.autograd.grad(output, leaves, inputs["dy"])
    return {"output": output.detach().cpu()} | {
        "d" + name: grad.detach().cpu() for name, grad in zip(names, grads, strict=True)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--candidate-deterministic", action="store_true")
    parser.add_argument(
        "--inputs", type=Path, help="Optional saved q/k/v/g/beta/dy dictionary"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.tokens > 0 and args.tokens % 64 == 0 and args.repeats >= 3
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    inputs = (
        fixed_inputs(args.tokens, args.seed)
        if args.inputs is None
        else torch.load(args.inputs, map_location="cpu", weights_only=True)
    )
    assert set(inputs) == {"q", "k", "v", "g", "beta", "dy"}
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value)
        for value in inputs.values()
    )
    expected = fixed_inputs(args.tokens, args.seed)
    assert all(
        inputs[name].shape == value.shape and inputs[name].dtype == value.dtype
        for name, value in expected.items()
    )
    module = importlib.import_module("torchtitan_npu.ops.triton.gdn.gated_delta")
    kernel = module.gated_delta_rule
    source = Path(inspect.getsourcefile(kernel))
    report = {
        "entry_sha256": file_hash(Path(__file__)),
        "status": "started",
        "kernel": {"path": str(source), "sha256": file_hash(source)},
        "inputs": {key: descriptor(value) for key, value in inputs.items()},
        "tolerances": {"atol": 0.005, "rtol": 0.02},
        "scope": "reference operator repeat only; not candidate acceptance",
        "autograd_multithreading": torch.autograd.is_multithreading_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "candidate_deterministic": args.candidate_deterministic,
    }
    try:
        torch.save(inputs, args.output / "inputs.pt")
        if args.cpu_check:
            report["status"] = "CPU_IMPORT_INPUT_ONLY_NOT_KERNEL_ACCEPTANCE"
            return
        assert torch.npu.is_available()
        torch.npu.set_device(args.device)
        device_inputs = {
            key: value.to(torch.device("npu", args.device))
            for key, value in inputs.items()
        }
        first, previous = None, None
        report["repeats"] = {}
        for repeat in range(args.repeats):
            result = evaluate(kernel, device_inputs)
            torch.save(result, args.output / f"reference-repeat{repeat}.pt")
            entry = {
                "tensors": {key: descriptor(value) for key, value in result.items()},
                "finite": all(
                    torch.isfinite(value).all().item() for value in result.values()
                ),
                "nonzero": {
                    key: bool(torch.count_nonzero(value))
                    for key, value in result.items()
                },
            }
            for label, baseline in (("vs_repeat0", first), ("vs_previous", previous)):
                if baseline is not None:
                    entry[label] = {
                        key: compare_branch(value, baseline[key])
                        for key, value in result.items()
                    }
            report["repeats"][str(repeat)] = entry
            first = first if first is not None else result
            previous = result
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        if args.candidate_deterministic:
            candidate_module = importlib.import_module(
                "llmtuner.accelerator.gated_delta"
            )
            candidate_source = Path(candidate_module.__file__)
            report["candidate_source"] = {
                "path": str(candidate_source),
                "sha256": file_hash(candidate_source),
            }

            def candidate(*values):
                return candidate_module.gated_delta_rule(
                    *values, deterministic_reduction=True
                )

            candidate_first = None
            report["candidate_repeats"] = {}
            for repeat in range(args.repeats):
                result = evaluate(candidate, device_inputs)
                torch.save(result, args.output / f"candidate-repeat{repeat}.pt")
                assert all(result[key].dtype == first[key].dtype for key in result)
                entry = {
                    "tensors": {
                        key: descriptor(value) for key, value in result.items()
                    },
                    "finite": all(
                        bool(torch.isfinite(value).all()) for value in result.values()
                    ),
                    "nonzero": {
                        key: bool(torch.count_nonzero(value))
                        for key, value in result.items()
                    },
                    "vs_reference_repeat0": {
                        key: compare_branch(value, first[key])
                        for key, value in result.items()
                    },
                }
                if candidate_first is not None:
                    entry["vs_candidate_repeat0"] = {
                        key: compare_branch(value, candidate_first[key])
                        for key, value in result.items()
                    }
                candidate_first = result if candidate_first is None else candidate_first
                report["candidate_repeats"][str(repeat)] = entry
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        report["status"] = "operator_repeat_complete_not_candidate_acceptance"
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
