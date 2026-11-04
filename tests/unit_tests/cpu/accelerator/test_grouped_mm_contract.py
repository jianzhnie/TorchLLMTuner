"""Empty expert segments must preserve gradients and reject invalid metadata."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from llmtuner.accelerator.grouped_mm import aligned_npu_grouped_mm


@pytest.mark.parametrize(
    ("left_shape", "right_shape", "output_shape"),
    [
        ((0, 8), (3, 8, 16), (0, 16)),
        ((8, 0), (0, 16), (3, 8, 16)),
        ((3, 8, 16), (16, 0), (8, 0)),
    ],
)
def test_empty_grouped_mm_preserves_both_gradient_paths(
    left_shape, right_shape, output_shape
):
    left = torch.randn(left_shape, requires_grad=True)
    right = torch.randn(right_shape, requires_grad=True)
    offsets = torch.zeros(3, dtype=torch.int32)
    output = aligned_npu_grouped_mm(left, right, offsets)
    assert output.shape == output_shape
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    for operand in (left, right):
        assert operand.grad is not None
        assert operand.grad.shape == operand.shape
        assert torch.count_nonzero(operand.grad) == 0


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    ("left_shape", "right_shape", "offset_shape", "message"),
    [
        ((0, 8), (3, 7, 16), (3,), "contracting dimensions"),
        ((0, 8), (3, 8, 16), (2,), "one offset per expert"),
        ((0, 8), (3, 8, 16), (3, 1), "one-dimensional"),
        ((0,), (3, 8, 16), (3,), "matrix dimensions"),
    ],
)
def test_invalid_empty_metadata_is_rejected(
    fake, left_shape, right_shape, offset_shape, message
):
    def run():
        return aligned_npu_grouped_mm(
            torch.empty(left_shape),
            torch.empty(right_shape),
            torch.zeros(offset_shape, dtype=torch.int32),
        )

    with pytest.raises((ValueError, RuntimeError), match=message):
        if fake:
            with FakeTensorMode():
                run()
        else:
            run()


@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [((0, 8), (3, 8, 16)), ((8, 0), (0, 16)), ((3, 8, 16), (16, 0))],
)
def test_empty_fake_matches_runtime_metadata(left_shape, right_shape):
    left = torch.empty(left_shape, dtype=torch.bfloat16)
    right = torch.empty(right_shape, dtype=torch.bfloat16)
    offsets = torch.zeros(3, dtype=torch.int32)
    actual = aligned_npu_grouped_mm(left, right, offsets)
    with FakeTensorMode() as mode:
        fake = aligned_npu_grouped_mm(
            mode.from_tensor(left), mode.from_tensor(right), mode.from_tensor(offsets)
        )
    assert fake.shape == actual.shape
    assert fake.stride() == actual.stride()
    assert fake.dtype == actual.dtype
