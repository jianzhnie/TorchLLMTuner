"""Compare actual baseline router VJPs using captured 30B inputs and weights.

Run explicitly on one idle NPU with the baseline source on PYTHONPATH.
This isolates routing; it does not establish full training equivalence.
"""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from llmtuner.models.hf.model_setup import forward_fp32_topk_router
from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    torch.npu.set_device(0)
    set_deterministic_algorithms()
    torch.manual_seed(37)
    trace = torch.load(args.trace, weights_only=True, mmap=True)
    state = torch.load(args.capture / "initial-hf.pt", weights_only=True, mmap=True)
    weight = state["model.layers.0.mlp.gate.weight"].to("npu")
    inputs = trace["layer0.post_norm"].reshape(-1, weight.shape[1]).to("npu")
    report = []
    for hf32 in (False, True):
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = hf32
        baseline = (
            TokenChoiceTopKRouter.Config(
                num_experts=weight.shape[0],
                gate=Linear.Config(
                    in_features=weight.shape[1], out_features=weight.shape[0]
                ),
                top_k=8,
                score_func="softmax",
                route_norm=True,
            )
            .build()
            .to(device="npu", dtype=weight.dtype)
        )
        with torch.no_grad():
            baseline.gate.weight.copy_(weight)
        candidate_weight = weight.detach().clone().requires_grad_()
        candidate = SimpleNamespace(
            weight=candidate_weight,
            hidden_dim=weight.shape[1],
            top_k=8,
            norm_topk_prob=True,
        )
        baseline_input = inputs.detach().clone().requires_grad_()
        candidate_input = inputs.detach().clone().requires_grad_()
        expected, expected_indices, _ = baseline(baseline_input)
        _, actual, actual_indices = forward_fp32_topk_router(candidate, candidate_input)
        upstream = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(
            expected, (baseline_input, baseline.gate.weight), upstream
        )
        actual_grads = torch.autograd.grad(
            actual, (candidate_input, candidate_weight), upstream
        )
        comparisons = [("scores", actual, expected)]
        comparisons.extend(
            zip(
                ("input_gradient", "weight_gradient"),
                actual_grads,
                expected_grads,
                strict=True,
            )
        )
        result = {
            "hf32": hf32,
            "indices_equal": torch.equal(actual_indices, expected_indices),
        }
        for name, observed, reference in comparisons:
            delta = (observed.float() - reference.float()).abs()
            result[name] = {
                "exact": torch.equal(observed, reference),
                "max_ae": delta.max().item(),
                "mean_ae": delta.mean().item(),
            }
        report.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert all(
        row["indices_equal"]
        and all(
            row[key]["exact"] for key in ("scores", "input_gradient", "weight_gradient")
        )
        for row in report
    )


if __name__ == "__main__":
    main()
