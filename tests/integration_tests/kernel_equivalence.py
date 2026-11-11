"""Single-idle-NPU kernel forward/backward checks against torch references.

Run from the repository root:
    python -m tests.integration_tests.kernel_equivalence --device-index 7

This exercises real kernels, not mocks. It does not measure performance or
validate collective behavior. FP32 RMSNorm/RoPE tolerances are 3e-5; BF16 packed
attention uses 2e-2; token permutation is exact.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

from llmtuner.accelerator.device import is_npu_available, set_device
from llmtuner.kernel.ops.attention import fusion_attention
from llmtuner.kernel.ops.moe import moe_token_permute
from llmtuner.kernel.ops.rms_norm import rms_norm
from llmtuner.kernel.ops.rotary import rotary_mul
from tests.caps import require_env


def compare(
    name: str,
    operation: Callable[..., torch.Tensor],
    inputs: list[torch.Tensor],
    device: torch.device,
    *,
    tolerance: float,
    **kwargs: object,
) -> dict[str, str | float]:
    """Compare output and every floating input gradient with asymmetric upstream."""
    reference = [
        tensor.detach().float().requires_grad_()
        if tensor.is_floating_point()
        else tensor.clone()
        for tensor in inputs
    ]
    actual = [
        tensor.detach().to(device).requires_grad_(tensor.is_floating_point())
        for tensor in inputs
    ]
    expected_output = operation(*reference, use_eager=True, **kwargs)
    actual_output = operation(*actual, **kwargs)
    torch.testing.assert_close(
        actual_output.cpu().float(),
        expected_output,
        atol=tolerance,
        rtol=tolerance,
    )
    gradient = torch.randn_like(expected_output)
    expected_output.backward(gradient)
    actual_output.backward(gradient.to(device=device, dtype=actual_output.dtype))
    gradient_error = 0.0
    for left, right in zip(actual, reference, strict=True):
        if not left.is_floating_point():
            continue
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(
            left.grad.cpu().float(), right.grad, atol=tolerance, rtol=tolerance
        )
        gradient_error = max(
            gradient_error, (left.grad.cpu().float() - right.grad).abs().max().item()
        )
    torch.npu.synchronize()
    return {
        "kernel": name,
        "output_max_abs": (actual_output.cpu().float() - expected_output)
        .abs()
        .max()
        .item(),
        "gradient_max_abs": gradient_error,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    require_env("torch_npu")
    if not is_npu_available():
        raise RuntimeError("This acceptance script requires a real, idle NPU")
    device = torch.device("npu", args.device_index)
    set_device(device)
    torch.manual_seed(37)
    report = [
        compare(
            "rms_norm",
            rms_norm,
            [torch.randn(4, 64), torch.randn(64)],
            device,
            tolerance=3e-5,
            epsilon=1e-6,
        ),
    ]
    for mode in ("half", "interleave"):
        angle = torch.randn(1, 3, 1, 64)
        report.append(
            compare(
                f"rotary_{mode}",
                rotary_mul,
                [torch.randn(2, 3, 2, 64), angle.cos(), angle.sin()],
                device,
                tolerance=3e-5,
                rotary_mode=mode,
            )
        )
    report.append(
        compare(
            "moe_token_permute",
            moe_token_permute,
            [
                torch.randn(6, 64),
                torch.tensor(
                    [[2, 0], [1, 2], [0, 1], [2, 0], [1, 1], [0, 2]], dtype=torch.int32
                ),
            ],
            device,
            tolerance=0.0,
        )
    )
    report.append(
        compare(
            "fusion_attention_bf16",
            fusion_attention,
            [torch.randn(tokens, 2, 64, dtype=torch.bfloat16) for tokens in (7, 9, 9)],
            device,
            tolerance=2e-2,
            head_num=2,
            scale=0.125,
            q_cu_seqlens=[0, 3, 7],
            k_cu_seqlens=[0, 4, 9],
        )
    )
    print(
        json.dumps(
            {"torch": torch.__version__, "device": str(device), "results": report},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
