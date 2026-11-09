"""CPU hook/IO contracts only; synthetic tensors are never NPU acceptance."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.integration_tests.qwen35_conv_alignment import (
    capture_hf_conv,
    capture_reference_conv,
    compare_capture_boundaries,
    conv_capture_path,
    raw_vjp,
)


def test_capture_only_single_hf_phase_and_rank_specific_paths(tmp_path):
    assert conv_capture_path(None, phase="reference", steps=5, rank=0) is None
    for phase, steps in (("reference", 2), ("prepare", 1), ("hf", 2)):
        with pytest.raises(ValueError, match="single-step HF/reference"):
            conv_capture_path(tmp_path, phase=phase, steps=steps, rank=0)
    assert (
        conv_capture_path(tmp_path, phase="hf", steps=1, rank=3).name == "rank-00003.pt"
    )


def test_capture_actual_cotangent_preserves_original_vjp_and_cleans_hooks(tmp_path):
    conv = torch.nn.Conv1d(4, 4, 4, padding=3, groups=4, bias=False)
    gdn = SimpleNamespace(conv1d=conv, causal_conv1d_fn=None)
    x = torch.arange(32).reshape(1, 4, 8).float().requires_grad_()
    dy = torch.linspace(-1, 1, 32).reshape(1, 4, 8)
    original = torch.nn.functional.silu(conv(x)[:, :, :8])
    original.backward(dy)
    dx, dw = x.grad.clone(), conv.weight.grad.clone()
    x.grad = None
    conv.zero_grad(set_to_none=True)
    parameter = conv.weight
    forward = conv.forward.__func__
    calls = []

    @contextmanager
    def errors():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    path = tmp_path / "captures" / "rank-00000.pt"
    with capture_hf_conv(
        gdn, path, provenance={"scope": "CPU_CONTRACT_ONLY"}, local_errors=errors
    ):
        raw = conv(x)
        raw.retain_grad()
        torch.nn.functional.silu(raw[:, :, :8]).backward(dy)
    payload = torch.load(path, weights_only=True)
    assert torch.equal(payload["input"], x.detach())
    assert torch.equal(payload["weight"], parameter.detach())
    assert torch.equal(payload["cotangent"], raw.grad)
    assert not bool(payload["cotangent"][:, :, 8:].any())
    assert torch.equal(x.grad, dx) and torch.equal(parameter.grad, dw)
    assert conv.weight is parameter and conv.forward.__func__ is forward
    assert not conv._forward_hooks and not raw._backward_hooks
    assert calls == ["enter", "exit", "enter", "exit"]
    with pytest.raises(AssertionError):
        with capture_hf_conv(gdn, path, provenance={"scope": "CPU"}):
            pass


def test_capture_failure_cleanup_and_coordinated_io(tmp_path):
    conv = torch.nn.Conv1d(2, 2, 4, padding=3, groups=2, bias=False)
    gdn = SimpleNamespace(conv1d=conv, causal_conv1d_fn=None)
    path = tmp_path / "capture.pt"
    with pytest.raises(RuntimeError, match="body"):
        with capture_hf_conv(gdn, path, provenance={"scope": "CPU"}):
            conv(torch.ones(1, 2, 8))
            raise RuntimeError("body")
    assert not path.exists() and not conv._forward_hooks
    # An external writer wins after preflight: exclusive final IO must raise
    # through the supplied coordinated-error context, never overwrite.
    errors = []

    @contextmanager
    def local_errors():
        try:
            yield
        except Exception as error:
            errors.append(type(error))
            raise

    with pytest.raises(FileExistsError):
        with capture_hf_conv(
            gdn, path, provenance={"scope": "CPU"}, local_errors=local_errors
        ):
            conv(torch.ones(1, 2, 8))[:, :, :8].sum().backward()
            path.write_bytes(b"existing")
    assert errors == [FileExistsError]
    assert path.read_bytes() == b"existing" and not conv._forward_hooks


def test_crossframework_capture_contains_original_forward_loss_backward_only():
    path = (
        Path(__file__).resolve().parents[3]
        / "integration_tests/qwen35_cross_framework.py"
    )
    tree = ast.parse(path.read_text())
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    block = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "conv_observer"
            for item in node.items
        )
    )
    text = ast.get_source_segment(path.read_text(), block)
    assert "else model(input_ids=tokens, use_cache=False).logits" in text
    assert "functional.cross_entropy" in text and "backward_loss.backward()" in text
    assert "optimizer.step" not in text
    assert "capture_first_conv" not in text  # no branching on math inside capture


@pytest.mark.parametrize("failed_copy", [1, 4])
def test_observation_copy_error_deferred_until_original_backward_finishes(
    tmp_path, monkeypatch, failed_copy
):
    conv = torch.nn.Conv1d(2, 2, 4, padding=3, groups=2, bias=False)
    gdn = SimpleNamespace(conv1d=conv, causal_conv1d_fn=None)
    x = torch.ones(1, 2, 8, requires_grad=True)
    original_cpu = torch.Tensor.cpu
    calls = 0
    finished = False
    coordinated = []

    def failing_cpu(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_copy:
            raise OSError("observer copy failed")
        return original_cpu(tensor, *args, **kwargs)

    @contextmanager
    def errors():
        try:
            yield
        except OSError:
            coordinated.append(finished)
            raise

    monkeypatch.setattr(torch.Tensor, "cpu", failing_cpu)
    path = tmp_path / "capture.pt"
    with pytest.raises(OSError, match="observer copy"):
        with capture_hf_conv(
            gdn, path, provenance={"scope": "CPU"}, local_errors=errors
        ):
            conv(x)[:, :, :8].sum().backward()
            finished = True
            assert x.grad is not None and conv.weight.grad is not None
    assert coordinated == [True]
    assert not path.exists() and not conv._forward_hooks


def test_remote_setup_failure_removes_already_registered_hook(tmp_path):
    conv = torch.nn.Conv1d(2, 2, 4, padding=3, groups=2, bias=False)
    gdn = SimpleNamespace(conv1d=conv, causal_conv1d_fn=None)

    @contextmanager
    def remote_error():
        yield
        assert conv._forward_hooks  # local setup succeeded before agreement
        raise RuntimeError("another rank failed setup")

    with pytest.raises(RuntimeError, match="another rank"):
        with capture_hf_conv(
            gdn,
            tmp_path / "capture.pt",
            provenance={"scope": "CPU"},
            local_errors=remote_error,
        ):
            pytest.fail("must not start model after failed setup agreement")
    assert not conv._forward_hooks


def test_packed_left_pad_original_parameter_vjp_and_repeat_contract():
    torch.manual_seed(7)
    conv = torch.nn.Conv1d(4, 4, 4, padding=3, groups=4, bias=False).double()
    parameter, forward = conv.weight, conv.forward.__func__
    x = torch.randn(1, 4, 9, dtype=torch.float64)
    dy = torch.nn.functional.pad(torch.randn_like(x), (0, 3))
    original = raw_vjp(conv, x, dy, packed=True)
    left = raw_vjp(conv, x, dy, packed=True, left_pad=True)
    repeat = raw_vjp(conv, x, dy, packed=True, left_pad=True)
    for key in ("output", "dx", "dw"):
        torch.testing.assert_close(left[key], original[key], atol=1e-12, rtol=1e-12)
        assert torch.equal(left[key], repeat[key])
    assert conv.weight is parameter and conv.forward.__func__ is forward
    assert conv.padding == (3,) and set(conv.state_dict()) == {"weight"}


class FunctionalReferenceConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_q = torch.nn.Conv1d(2, 2, 4, groups=2, bias=False)
        self.conv_k = torch.nn.Conv1d(2, 2, 4, groups=2, bias=False)
        self.conv_v = torch.nn.Conv1d(4, 4, 4, groups=4, bias=False)

    def forward(self, x):
        outputs = []
        for name, part in zip(("q", "k", "v"), x.split((2, 2, 4), dim=1), strict=True):
            module = getattr(self, "conv_" + name)
            raw = torch.nn.functional.conv1d(
                torch.nn.functional.pad(part, (3, 0)),
                module.weight,
                groups=part.shape[1],
            )
            outputs.append(torch.nn.functional.silu(raw))
        return torch.cat(outputs, dim=1)


def test_reference_actual_functional_boundary_scale_and_original_vjp(tmp_path):
    reference = FunctionalReferenceConv()
    x = torch.randn(1, 8, 9, requires_grad=True)
    dy = torch.randn_like(x)
    reference(x).backward(dy / 8)
    dx = x.grad.clone()
    grads = [p.grad.clone() for p in reference.parameters()]
    reference.zero_grad(set_to_none=True)
    x.grad = None
    path = tmp_path / "reference.pt"
    with capture_reference_conv(
        reference, path, provenance={"scope": "CPU"}, loss_scale=1 / 8
    ):
        reference(x).backward(dy / 8)
    payload = torch.load(path, weights_only=True)
    assert payload["loss_scale"] == 1 / 8 and payload["normalization_multiplier"] == 8
    assert torch.equal(payload["input"], x.detach())
    assert payload["output"].shape == x.shape
    assert payload["cotangent"].shape == x.shape
    assert torch.equal(
        payload["normalized_cotangent"], payload["cotangent"].float() * 8
    )
    assert torch.equal(dx, x.grad)
    assert all(
        torch.equal(p.grad, grad)
        for p, grad in zip(reference.parameters(), grads, strict=True)
    )
    assert not reference._forward_pre_hooks
    assert all(not m._forward_hooks for m in reference.modules())


def test_reference_copy_failure_deferred_until_backward_complete(tmp_path, monkeypatch):
    reference = FunctionalReferenceConv()
    original = torch.Tensor.cpu
    count = 0
    finished = False
    coordinated = []

    def fail(tensor, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise OSError("reference observer copy")
        return original(tensor, *args, **kwargs)

    @contextmanager
    def errors():
        try:
            yield
        except OSError:
            coordinated.append(finished)
            raise

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    with pytest.raises(OSError, match="reference observer"):
        with capture_reference_conv(
            reference,
            tmp_path / "failed.pt",
            provenance={"scope": "CPU"},
            loss_scale=1 / 8,
            local_errors=errors,
        ):
            reference(torch.ones(1, 8, 9)).sum().backward()
            finished = True
    assert coordinated == [True]
    assert not reference._forward_pre_hooks


def test_causal_boundary_comparison_retains_original_scaled_dy():
    provenance = {
        "initial_manifest_sha256": "a",
        "initial_layer0_sha256": "b",
        "rank": 0,
        "step": 0,
        "tokens": [1, 2],
    }
    x = torch.ones(1, 8, 9, dtype=torch.bfloat16)
    weight = torch.ones(8, 1, 4, dtype=torch.bfloat16)
    hf = {
        "loss_scale": 1.0,
        "provenance": provenance,
        "input": x,
        "weight": weight,
        "output": torch.nn.functional.pad(x, (0, 3)),
        "cotangent": torch.nn.functional.pad(x, (0, 3)),
    }
    reference = {
        "loss_scale": 1 / 8,
        "provenance": provenance,
        "input": x,
        "weight": weight,
        "output": x,
        "cotangent": x / 8,
    }
    raw = reference["cotangent"].clone()
    assert all(
        item["exact"] for item in compare_capture_boundaries(hf, reference).values()
    )
    assert torch.equal(raw, reference["cotangent"])
    reference["loss_scale"] = 1.0
    with pytest.raises(AssertionError):
        compare_capture_boundaries(hf, reference)
