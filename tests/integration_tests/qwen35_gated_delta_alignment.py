"""Compare native HF and TorchTitan-NPU GDN forward/backward on real NPUs.

Requires the isolated reference environment on PYTHONPATH. This operator test
does not replace full model or distributed training acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - register the real NPU backend
from torch.utils.checkpoint import checkpoint
from torchtitan_npu.ops.triton.gdn import gated_delta_rule
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
)

from llmtuner.accelerator.gated_delta import (
    gated_delta_rule as native_npu_gated_delta_rule,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--reset-token", type=int)
    parser.add_argument("--strided-upstream", action="store_true")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--candidate", choices=("hf", "native-npu"), default="hf")
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("Real NPU execution is required")
    if args.tokens <= 0 or args.tokens % 64:
        raise ValueError("tokens must be a positive multiple of 64")
    if args.reset_token is not None and not 0 < args.reset_token < args.tokens:
        raise ValueError("reset-token must fall strictly inside the sequence")
    if args.reset_token is not None and args.candidate != "native-npu":
        raise ValueError("reset comparison requires --candidate native-npu")
    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    # Published 35B-A3B GDN head geometry; this isolates a kernel, not a model.
    shape = (1, args.tokens, 32, 128)
    key_shape = (1, args.tokens, 16, 128)
    values = [torch.randn(key_shape).to(torch.bfloat16) for _ in range(2)]
    values.append(torch.randn(shape).to(torch.bfloat16))
    values += [-torch.rand(shape[:3]), torch.rand(shape[:3]).to(torch.bfloat16)]
    candidate = [value.npu().requires_grad_() for value in values]
    reference = [value.npu().requires_grad_() for value in values]
    upstream = torch.randn(shape).to(device="npu", dtype=torch.bfloat16)
    if args.strided_upstream:
        storage = torch.empty(
            (*shape[:-1], shape[-1] * 2), device="npu", dtype=torch.bfloat16
        )
        storage[..., ::2].copy_(upstream)
        upstream = storage[..., ::2]
        assert not upstream.is_contiguous()
    reset = None
    if args.reset_token is not None:
        reset = torch.zeros(shape[:2], device="npu", dtype=torch.bool)
        reset[:, args.reset_token] = True
    if args.candidate == "native-npu":
        if args.activation_checkpoint:
            hf_output = checkpoint(
                native_npu_gated_delta_rule,
                *candidate,
                reset=reset,
                use_reentrant=False,
            )
        else:
            hf_output = native_npu_gated_delta_rule(*candidate, reset=reset)
    else:
        if args.activation_checkpoint:
            raise ValueError("checkpoint comparison requires --candidate native-npu")
        hf_output, _ = torch_chunk_gated_delta_rule(
            candidate[0].repeat_interleave(2, dim=2),
            candidate[1].repeat_interleave(2, dim=2),
            *candidate[2:],
            use_qk_l2norm_in_kernel=True,
        )
    npu_output = gated_delta_rule(*reference, reset=reset)
    hf_grads = torch.autograd.grad(hf_output, candidate, upstream)
    npu_grads = torch.autograd.grad(npu_output, reference, upstream)
    torch.npu.synchronize()
    result = {
        "scope": "GDN operator only, BF16 qkv/beta, FP32 decay",
        "candidate": args.candidate,
        "reset_token": args.reset_token,
        "activation_checkpoint": args.activation_checkpoint,
        "upstream_stride": list(upstream.stride()),
        "shape": list(shape),
        "qk_shape": list(key_shape),
        "atol": 5e-3,
        "rtol": 2e-2,
        "comparisons": {},
    }
    failures = []
    names = ("output", "dq", "dk", "dv", "dg", "dbeta")
    for name, actual, expected in zip(
        names, (hf_output, *hf_grads), (npu_output, *npu_grads), strict=True
    ):
        actual, expected = (
            actual.detach().float().cpu(),
            expected.detach().float().cpu(),
        )
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        nonzero = bool(torch.count_nonzero(actual) and torch.count_nonzero(expected))
        close = torch.allclose(actual, expected, atol=5e-3, rtol=2e-2)
        result["comparisons"][name] = {
            "finite": finite,
            "nonzero": nonzero,
            "max_absolute_error": (actual - expected).abs().max().item(),
            "exact": torch.equal(actual, expected),
            "passed": finite and nonzero and close,
        }
        if not (finite and nonzero and close):
            failures.append(name)
    result["status"] = "failed" if failures else "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(f"GDN numerical comparison failed: {failures}")


if __name__ == "__main__":
    main()
