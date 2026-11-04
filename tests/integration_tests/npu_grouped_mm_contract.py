"""Real NPU grouped GEMM: alignment, empty experts, autograd and op contracts.

Run explicitly: PYTHONPATH=. python tests/integration_tests/npu_grouped_mm_contract.py
"""

from tests.caps import require_env

require_env("torch_npu")

import torch
import torch_npu  # noqa: F401 -- register the NPU backend before creating tensors

from llmtuner.accelerator.grouped_mm import aligned_npu_grouped_mm


def main() -> None:
    torch.npu.set_device(0)
    torch.manual_seed(20261104)
    for counts in ([3, 0, 5], [0, 0, 0]):
        tokens = sum(counts)
        # Both operands intentionally start at an unaligned BF16 address.
        left = torch.randn(tokens * 16 + 1, device="npu", dtype=torch.bfloat16)
        left = left[1:].view(tokens, 16).requires_grad_()
        right = torch.randn(3 * 16 * 24 + 1, device="npu", dtype=torch.bfloat16)
        right = right[1:].view(3, 16, 24).requires_grad_()
        offsets = torch.tensor(counts, device="npu", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        before = (left.detach().clone(), right.detach().clone(), offsets.clone())
        reference_left = left.detach().clone().requires_grad_()
        reference_right = right.detach().clone().requires_grad_()
        segments = reference_left.split(counts)
        expected = torch.cat(
            [segment @ reference_right[index] for index, segment in enumerate(segments)]
        )
        actual = aligned_npu_grouped_mm(left, right, offsets)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for operand, reference in (
            (left, reference_left),
            (right, reference_right),
        ):
            torch.testing.assert_close(
                operand.grad, reference.grad, rtol=5e-2, atol=5e-2
            )
        assert torch.count_nonzero(right.grad[1]).item() == 0
        if tokens:
            assert left.grad.count_nonzero().item() > 0
            assert right.grad.count_nonzero().item() > 0
        for operand, original in zip((left, right, offsets), before, strict=True):
            torch.testing.assert_close(operand, original, rtol=0, atol=0)
        # Upstream autograd_registration_check supports CPU/CUDA/XPU only.
        # Gradients are checked numerically above on the actual NPU instead.
        checks = torch.library.opcheck(
            aligned_npu_grouped_mm,
            (left, right, offsets),
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        )
        assert all(value == "SUCCESS" for value in checks.values()), checks
        print(f"PASS grouped GEMM counts={counts}: {checks}", flush=True)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
