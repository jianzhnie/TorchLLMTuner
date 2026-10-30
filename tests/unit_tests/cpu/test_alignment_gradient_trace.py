from types import SimpleNamespace

import torch

from tests.integration_tests.alignment_gradient_trace import (
    local_gradient_record,
    observe_gradient_clipping,
)


def test_local_samples_are_bounded_independent_copies():
    gradient = torch.arange(64.0).reshape(2, 4, 8)
    record = local_gradient_record(gradient, sample_size=3)
    assert record["sample"].numel() == 3
    assert all(value.numel() == 3 for value in record["half_samples"])
    torch.testing.assert_close(record["half_samples"][1], gradient[0, 2, :3])
    gradient.zero_()
    torch.testing.assert_close(record["sample"], torch.arange(3.0))


def test_observer_preserves_real_clip_and_preclip_evidence(tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    parameter = model.layers[0].weight
    parameter.grad = torch.full_like(parameter, 3.0)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    norm = observed(model.parameters(), 1.0)
    torch.testing.assert_close(norm, torch.tensor(6.0))
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    name = "part0.layers.0.weight"
    torch.testing.assert_close(record["before"][name]["sample"], torch.full((4,), 3.0))
    torch.testing.assert_close(
        record["after"][name]["sample"], parameter.grad.flatten()
    )
    torch.testing.assert_close(record["returned_norm"], norm)


def test_gradient_trace_selects_exact_layer_indices(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(11)]
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_GRADIENT_LAYERS", "1,10")
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert set(record["before"]) == {"part0.layers.1.weight", "part0.layers.10.weight"}
