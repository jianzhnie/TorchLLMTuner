"""Pure CPU original-graph capture and deferred observation error contracts."""

from contextlib import contextmanager

import pytest
import torch

from tests.integration_tests.qwen35_backward_capture import (
    backward_capture_path,
    capture_layer_cotangents,
    compare_layer_cotangents,
    ordered_layers,
)


def test_actual_40layer_original_vjp_and_loss_scale_comparison(tmp_path):
    torch.manual_seed(1)
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(40)])
    # Near identity avoids an artificial vanishing-gradient test.
    with torch.no_grad():
        for layer in layers:
            layer.weight.copy_(torch.eye(4) + layer.weight / 100)
    x = torch.randn(1, 3, 4, requires_grad=True)
    dy = torch.randn_like(x)

    def execute():
        value = x
        for layer in layers:
            value = layer(value)
        return value

    execute().backward(dy)
    dx, grads = x.grad.clone(), [p.grad.clone() for p in layers.parameters()]
    provenance = {
        "initial_manifest_sha256": "a",
        "initial_layer0_sha256": "b",
        "rank": 0,
        "step": 0,
        "tokens": [1, 2, 3],
    }
    for name, scale in (("hf", 1), ("reference", 1 / 8)):
        layers.zero_grad(set_to_none=True)
        x.grad = None
        with capture_layer_cotangents(
            layers, tmp_path / (name + ".pt"), provenance=provenance, loss_scale=scale
        ):
            output = execute()
            output.backward(dy * scale)
        assert torch.equal(x.grad / scale, dx)
        assert all(
            torch.equal(p.grad / scale, grad)
            for p, grad in zip(layers.parameters(), grads, strict=True)
        )
        assert all(not layer._forward_hooks for layer in layers)
        assert not output._backward_hooks
    hf = torch.load(tmp_path / "hf.pt", weights_only=True)
    reference = torch.load(tmp_path / "reference.pt", weights_only=True)
    report = compare_layer_cotangents(hf, reference)
    assert (
        report["numerical_all_passed"] and report["first_nonexact_from_output"] is None
    )
    reference["raw_cotangents"][20].add_(1)
    report = compare_layer_cotangents(hf, reference)
    assert report["first_failed_from_output"] == 20
    assert report["first_nonexact_from_output"] == 20
    assert torch.equal(hf["raw_cotangents"][39], dy)


def test_tensor_copy_failure_deferred_and_coordinated(tmp_path, monkeypatch):
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(40)])
    original = torch.Tensor.cpu
    calls, finished, coordinated = 0, False, []

    def fail(value, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("observe copy failed")
        return original(value, *args, **kwargs)

    @contextmanager
    def errors():
        try:
            yield
        except OSError:
            coordinated.append(finished)
            raise

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    path = tmp_path / "failed.pt"
    with pytest.raises(OSError, match="observe copy"):
        with capture_layer_cotangents(
            layers, path, provenance={"scope": "CPU"}, loss_scale=1, local_errors=errors
        ):
            value = torch.ones(1, 2)
            for layer in layers:
                value = layer(value)
            value.sum().backward()
            finished = True
    assert coordinated == [True] and not path.exists()
    assert all(not layer._forward_hooks for layer in layers)


def test_remote_setup_failure_cleans_all_hooks(tmp_path):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(40)])

    @contextmanager
    def remote_error():
        yield
        raise RuntimeError("another rank failed")

    with pytest.raises(RuntimeError, match="another rank"):
        with capture_layer_cotangents(
            layers,
            tmp_path / "failed.pt",
            provenance={"scope": "CPU"},
            loss_scale=1,
            local_errors=remote_error,
        ):
            pytest.fail("model must not execute after failed setup")
    assert all(not layer._forward_hooks for layer in layers)


def test_real_module_container_order_and_path_contract(tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict({str(i): torch.nn.Identity() for i in range(40)})
    assert ordered_layers(model, reference=True)[0] is model.layers["0"]
    assert (
        backward_capture_path(tmp_path, phase="reference", steps=1, rank=7).name
        == "rank-00007.pt"
    )
    with pytest.raises(ValueError):
        backward_capture_path(tmp_path, phase="reference", steps=2, rank=7)
