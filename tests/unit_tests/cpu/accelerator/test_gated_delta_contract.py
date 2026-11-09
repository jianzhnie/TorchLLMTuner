"""CPU/Fake contracts only: never import torch_npu or launch an NPU kernel."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from llmtuner.accelerator.gated_delta import (
    _require_triton,
    gated_delta_autograd_backward,
    gated_delta_backward_op,
    gated_delta_op,
    gated_delta_rule,
    gated_delta_setup_context,
)


def test_schema_preserves_functional_forward_and_backward_contract():
    forward = torch.ops.llmtuner.gated_delta_rule.default._schema
    backward = torch.ops.llmtuner.gated_delta_rule_backward.default._schema
    assert [arg.name for arg in forward.arguments] == [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "reset",
        "scale",
        "deterministic_reduction",
    ]
    assert [arg.name for arg in backward.arguments] == [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "reset",
        "dy",
        "scale",
        "deterministic_reduction",
    ]
    for schema in (forward, backward):
        assert schema.arguments[-1].kwarg_only
        assert schema.arguments[-1].default_value is False
        assert all(arg.alias_info is None for arg in schema.arguments)
        assert all(result.alias_info is None for result in schema.returns)
    assert len(forward.returns) == 1
    assert len(backward.returns) == 5
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "llmtuner::gated_delta_rule", "Autograd"
    )
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "llmtuner::gated_delta_rule", "CPU"
    )


def test_public_wrapper_and_raw_op_have_no_cpu_fallback():
    q = torch.empty((1, 64, 2, 128), dtype=torch.bfloat16)
    g = torch.empty((1, 64, 2), dtype=torch.float32)
    reset = torch.zeros((1, 64), dtype=torch.bool)
    with pytest.raises(ValueError, match="native NPU"):
        gated_delta_rule(q, q, q, g, g)
    with pytest.raises(NotImplementedError, match="CPU"):
        gated_delta_op(q, q, q, g, g, reset, scale=128**-0.5)


def test_fake_metadata_and_backward_in_isolated_cpu_process():
    # PrivateUse1 naming is process-global. Isolate the metadata-only backend
    # so no other test's device discovery is affected. Block optional Triton,
    # and disable PyTorch's automatic loading of installed vendor runtimes.
    code = (
        "import sys; sys.modules['triton'] = None; "
        "from tests.unit_tests.cpu.accelerator.test_gated_delta_contract "
        "import run_fake_contract; run_fake_contract()"
    )
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_fake_contract():
    """Exercise actual dispatcher Fake implementations with meta storage only."""
    assert "torch_npu" not in sys.modules
    assert not any(name.startswith("torchtitan") for name in sys.modules)
    with pytest.raises(RuntimeError, match="Ascend-compatible Triton"):
        _require_triton()
    torch.utils.rename_privateuse1_backend("npu")
    # Only FakeTensor's device metadata queries are provided. There are no
    # allocators, native kernels, device initialization, or numerical mocks.
    torch._register_device_module(
        "npu", SimpleNamespace(is_initialized=lambda: False, current_device=lambda: 0)
    )
    with FakeTensorMode() as mode:

        def fake(shape, dtype=torch.bfloat16, device="npu:0"):
            return FakeTensor(
                mode,
                torch.empty(shape, device="meta", dtype=dtype),
                torch.device(device),
            )

        def strided_fake(heads):
            # Slice meta storage before assigning Fake device metadata:
            # CPU-only PyTorch cannot create an NPU device guard for slicing.
            storage = torch.empty(
                (1, 128, heads, 256), device="meta", dtype=torch.bfloat16
            )[..., ::2]
            return FakeTensor(mode, storage, torch.device("npu:0"))

        q = fake((1, 128, 16, 128))
        v = fake((1, 128, 32, 128))
        g = fake((1, 128, 32), torch.float32)
        beta = fake((1, 128, 32))
        reset = fake((1, 128), torch.bool)
        # Public grouped-head path must work without Triton, including the
        # implicit reset allocation and default scale.
        output = gated_delta_rule(q, q, v, g, beta)
        assert (output.shape, output.dtype, output.device) == (
            v.shape,
            v.dtype,
            v.device,
        )
        assert output.is_contiguous()
        assert output is not v
        qh = fake(v.shape)
        inputs = (qh, qh, v, g, beta, reset)
        grads = gated_delta_backward_op(*inputs, output, scale=128**-0.5)
        assert len(grads) == 5
        for grad, original in zip(grads, inputs[:5], strict=True):
            assert (grad.shape, grad.dtype, grad.device) == (
                original.shape,
                original.dtype,
                original.device,
            )
            assert grad is not original

        invalid = [
            (0, fake((1, 128, 32)), ValueError, "rank-4"),
            (1, fake((1, 128, 16, 128)), ValueError, "match q"),
            (2, fake((1, 128, 16, 128)), ValueError, "match q"),
            (3, fake((1, 128, 16), torch.float32), ValueError, "dimensions"),
            (4, fake(g.shape, torch.float64), TypeError, "float32"),
            (5, fake((1, 128), torch.int32), ValueError, "boolean"),
            (3, fake(g.shape, torch.float32, "npu:1"), ValueError, "one device"),
            (0, strided_fake(32), ValueError, "contiguous"),
        ]
        for index, bad, error, message in invalid:
            operands = list(inputs)
            operands[index] = bad
            with pytest.raises(error, match=message):
                gated_delta_op(*operands, scale=128**-0.5)
        for scale in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                gated_delta_op(*inputs, scale=scale)
        for dy, message in (
            (fake(v.shape, torch.float32), "match v"),
            (strided_fake(32), "contiguous"),
            (fake(v.shape, device="npu:1"), "match v"),
        ):
            with pytest.raises(ValueError, match=message):
                gated_delta_backward_op(*inputs, dy, scale=128**-0.5)
        for shape in ((1, 63, 32, 128), (1, 128, 32, 64), (0, 128, 32, 128)):
            bad = fake(shape)
            with pytest.raises(ValueError):
                gated_delta_rule(bad, bad, bad, g, beta)
        with pytest.raises(ValueError, match="multiple"):
            gated_delta_rule(
                fake((1, 128, 17, 128)), fake((1, 128, 17, 128)), v, g, beta
            )
        # Noncontiguous public inputs are intentionally canonicalized; raw
        # custom-op inputs must already be contiguous for pointer arithmetic.
        noncontiguous = strided_fake(16)
        assert (
            gated_delta_rule(noncontiguous, noncontiguous, v, g, beta).shape == v.shape
        )

        saved = []
        ctx = SimpleNamespace(save_for_backward=lambda *values: saved.extend(values))
        gated_delta_setup_context(ctx, inputs, {"scale": 128**-0.5}, output)
        assert all(a is b for a, b in zip(saved, inputs, strict=True))
        ctx.saved_tensors = tuple(saved)
        # Exercise the actual autograd callback with a contiguous cotangent;
        # CPU-only PyTorch cannot guard the NPU copy needed for a strided one.
        # No numeric gradient claim is made by Fake execution.
        dy = fake(v.shape)
        assert dy.is_contiguous()
        callback_grads = gated_delta_autograd_backward(ctx, dy)
        assert len(callback_grads) == 6 and callback_grads[-1] is None
        assert [grad.shape for grad in callback_grads[:-1]] == [
            value.shape for value in inputs[:5]
        ]
        # Explicit block reduction follows the same fake/schema/autograd contract.
        deterministic_output = gated_delta_rule(
            q, q, v, g, beta, deterministic_reduction=True
        )
        assert deterministic_output.shape == output.shape
        gated_delta_setup_context(
            ctx, inputs, {"scale": 128**-0.5, "deterministic_reduction": True}, output
        )
        assert ctx.deterministic_reduction is True
        deterministic_grads = gated_delta_autograd_backward(ctx, dy)
        assert [grad.shape for grad in deterministic_grads[:-1]] == [
            value.shape for value in inputs[:5]
        ]
        with pytest.raises(TypeError, match="must be bool"):
            gated_delta_rule(q, q, v, g, beta, deterministic_reduction="yes")
    assert "torch_npu" not in sys.modules
