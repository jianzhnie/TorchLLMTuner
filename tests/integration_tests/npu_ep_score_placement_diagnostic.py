"""Measure BF16 expert score placement against the active baseline kernel.

Uses identical sorted tokens, counts and weights. Reports discrepancies without
turning numerical differences into an acceptance pass. Run on one idle NPU.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch

from llmtuner.models.common.moe.experts import GroupedExperts
from tests.caps import require_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_env("torch_npu")
    importlib.import_module("torchtitan_npu")
    from torchtitan_npu.converters.kernels.gmm import _run_experts_grouped_mm

    torch.npu.set_device(0)
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = False
    torch.manual_seed(37)
    experts, width, hidden = 4, 128, 128
    block = GroupedExperts(width, hidden, experts, use_grouped_mm=True).to(
        device="npu", dtype=torch.bfloat16
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(std=0.1)
    inputs = torch.randn(32, width, device="npu", dtype=torch.bfloat16)
    counts = torch.full((experts,), 8, device="npu", dtype=torch.int64)
    scores = torch.rand(32, 1, device="npu")
    probe = torch.randn_like(inputs)
    report = []
    for placement in ("after_down", "before_down", "packed_fused_before_down"):
        x = inputs.detach().clone().requires_grad_()
        score = scores.detach().clone().requires_grad_()
        gate = block.w1_EFD.detach().clone().requires_grad_()
        up = block.w3_EFD.detach().clone().requires_grad_()
        down = block.w2_EDF.detach().clone().requires_grad_()
        baseline = _run_experts_grouped_mm(
            torch.cat((gate, up), dim=1),
            down,
            None,
            x,
            counts,
            routed_scores=score,
        )
        offsets = counts.cumsum(0, dtype=torch.int32)
        if placement == "packed_fused_before_down":
            import torch_npu

            projected = block._grouped_mm(x, torch.cat((gate, up), dim=1), offsets)
            activation = torch_npu.npu_swiglu(projected, dim=-1)
        else:
            gate_output = block._grouped_mm(x, gate, offsets)
            up_output = block._grouped_mm(x, up, offsets)
            activation = block.activation_fn(gate_output, up_output)
        if placement in ("before_down", "packed_fused_before_down"):
            activation = activation * score.to(activation.dtype)
        actual = block._grouped_mm(activation, down, offsets)
        if placement == "after_down":
            actual = (actual.float() * score).to(actual.dtype)
        values = (x, score, gate, up, down)
        baseline_grads = torch.autograd.grad(baseline, values, probe)
        actual_grads = torch.autograd.grad(actual, values, probe)
        comparisons = [("output", actual, baseline)]
        comparisons.extend(
            zip(
                ("input_grad", "score_grad", "gate_grad", "up_grad", "down_grad"),
                actual_grads,
                baseline_grads,
                strict=True,
            )
        )
        metrics = {}
        for name, observed, expected in comparisons:
            assert torch.isfinite(observed).all() and torch.isfinite(expected).all()
            delta = (observed.float() - expected.float()).abs()
            metrics[name] = {
                "max_ae": delta.max().item(),
                "mean_ae": delta.mean().item(),
                "exact": torch.equal(observed, expected),
            }
        report.append({"placement": placement, "metrics": metrics})
    # Parameter-precision clipping rounds each BF16 parameter norm before
    # FP32 accumulation. Splitting one fused parameter changes that contract.
    norm_cases = []
    for seed in (37, 91, 137):
        torch.manual_seed(seed)
        first = torch.randn(4, 128, 128, device="npu", dtype=torch.bfloat16)
        second = torch.randn_like(first)
        fused_norm = torch.linalg.vector_norm(torch.cat((first, second), dim=1)).float()
        split_norm = (
            torch.linalg.vector_norm(first).float().square()
            + torch.linalg.vector_norm(second).float().square()
        ).sqrt()
        norm_cases.append(
            {
                "seed": seed,
                "fused": fused_norm.item(),
                "split": split_norm.item(),
                "difference": (fused_norm - split_norm).item(),
            }
        )
    report.append({"parameter_norm_grouping": norm_cases})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
