"""Compare NPU fused ops with their native BF16 references."""

import time

import torch
import torch_npu
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


def measure(name, native, fused, inputs, tolerance=5e-2):
    expected = native()
    actual = fused()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    upstream = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    assert actual.count_nonzero().item() > 0
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert actual_gradient.count_nonzero().item() > 0
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance
        )
    print(f"PASS {name}: forward and all input gradients; rtol=atol={tolerance}")
    for label, function in (("native", native), ("fused", fused)):
        for _ in range(5):
            function()
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            function()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - started) / 20
        print(f"{name} {label} forward: {elapsed * 1000:.3f} ms", flush=True)


def main() -> None:
    torch.npu.set_device(0)
    torch.manual_seed(20261104)
    weight = torch.randn(2048, dtype=torch.bfloat16, device="npu", requires_grad=True)
    hidden = torch.randn(
        4096, 2048, dtype=torch.bfloat16, device="npu", requires_grad=True
    )
    measure(
        "rms_norm",
        lambda: torch.nn.functional.rms_norm(hidden, (2048,), weight, 1e-6),
        lambda: torch_npu.npu_rms_norm(hidden, weight, 1e-6)[0],
        (hidden, weight),
    )
    query = torch.randn(
        1, 4096, 8, 128, dtype=torch.bfloat16, device="npu", requires_grad=True
    )
    key = query.clone()
    cos = torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="npu")
    sin = torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="npu")
    measure(
        "rotary_q",
        lambda: apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)[0],
        lambda: torch_npu.npu_rotary_mul(query, cos.unsqueeze(2), sin.unsqueeze(2)),
        (query,),
    )
    projected = torch.randn(
        4096, 2048, dtype=torch.bfloat16, device="npu", requires_grad=True
    )
    measure(
        "swiglu",
        lambda: torch.nn.functional.silu(projected[..., :1024]) * projected[..., 1024:],
        lambda: torch_npu.npu_swiglu(projected, dim=-1),
        (projected,),
    )


if __name__ == "__main__":
    main()
