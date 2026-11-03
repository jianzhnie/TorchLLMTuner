"""Compare NPU fused ops with their native BF16 references."""

import time

import torch
import torch_npu
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb


def measure(name, native, fused, tolerance=5e-2):
    expected = native()
    actual = fused()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for function in (native, fused):
        for _ in range(5):
            function()
        torch.npu.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            function()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - started) / 20
        print(f"{name}: {elapsed * 1000:.3f} ms", flush=True)


def main() -> None:
    torch.npu.set_device(0)
    weight = torch.randn(2048, dtype=torch.bfloat16, device="npu")
    hidden = torch.randn(4096, 2048, dtype=torch.bfloat16, device="npu")
    measure(
        "rms_norm",
        lambda: torch.nn.functional.rms_norm(hidden, (2048,), weight, 1e-6),
        lambda: torch_npu.npu_rms_norm(hidden, weight, 1e-6)[0],
    )
    query = torch.randn(1, 4096, 8, 128, dtype=torch.bfloat16, device="npu")
    key = query.clone()
    cos = torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="npu")
    sin = torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="npu")
    measure(
        "rotary_q",
        lambda: apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)[0],
        lambda: torch_npu.npu_rotary_mul(query, cos.unsqueeze(2), sin.unsqueeze(2)),
    )
    projected = torch.randn(4096, 2048, dtype=torch.bfloat16, device="npu")
    measure(
        "swiglu",
        lambda: torch.nn.functional.silu(projected[..., :1024]) * projected[..., 1024:],
        lambda: torch_npu.npu_swiglu(projected, dim=-1),
    )


if __name__ == "__main__":
    main()
