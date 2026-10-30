"""Full-model diagnostics must capture only explicitly selected layers."""

import pytest
import torch

from tests.integration_tests.alignment_module_trace import selected_trace_modules


@pytest.fixture
def modules(monkeypatch):
    from tests.integration_tests import npu_model_forward_trace

    modules = {
        name: object()
        for name in (
            "embedding",
            "lm_head",
            "final_norm",
            "rotary_cache",
            "layer0",
            "layer0.attention",
            "layer1",
            "layer1.moe",
        )
    }
    monkeypatch.setattr(npu_model_forward_trace, "trace_modules", lambda *_: modules)
    return modules


def test_selection_excludes_logits_embedding_and_other_layers(modules):
    selected = selected_trace_modules(None, baseline=True, selection="1")
    assert set(selected) == {"final_norm", "layer1", "layer1.moe"}
    assert selected["layer1"] is modules["layer1"]


def test_boundary_selection_excludes_inner_modules(modules, monkeypatch):
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_TRACE_BOUNDARIES_ONLY", "1")
    selected = selected_trace_modules(None, baseline=True, selection="0,1")
    assert set(selected) == {"final_norm", "layer0", "layer1"}


def test_final_selection_avoids_traversing_decoder():
    from types import SimpleNamespace

    norm = object()
    trainer = SimpleNamespace(model_parts=[SimpleNamespace(norm=norm)])
    assert selected_trace_modules(trainer, baseline=True, selection="final") == {
        "final_norm": norm
    }


@pytest.mark.parametrize("selection", ["", "-1", "1,", "all", "1 2", "2"])
def test_invalid_selection_rejected(modules, selection):
    with pytest.raises(ValueError):
        selected_trace_modules(None, baseline=False, selection=selection)


def test_module_observer_preserves_gradients_and_captures_only_first_step(tmp_path):
    from types import SimpleNamespace

    from tests.integration_tests.npu_baseline_training_capture import (
        observe_first_step_modules,
    )

    trainer = SimpleNamespace(step=1)
    module = torch.nn.Linear(4, 3)
    inputs = torch.randn(2, 4, requires_grad=True)
    probe = torch.randn(2, 3)
    expected = torch.autograd.grad(module(inputs), inputs, probe)[0]
    partial_path = tmp_path / "trace.partial.pt"
    captured, handles = observe_first_step_modules(
        trainer, {"final_norm": module}, partial_path=partial_path
    )
    observed = torch.autograd.grad(module(inputs), inputs, probe)[0]
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    torch.testing.assert_close(captured["final_norm.input_gradient"], expected)
    torch.testing.assert_close(captured["final_norm.gradient"], probe)
    persisted = torch.load(partial_path, weights_only=True)
    torch.testing.assert_close(persisted["final_norm.input_gradient"], expected)
    saved = captured["final_norm.output"].clone()
    trainer.step = 2
    module(inputs + 1)
    torch.testing.assert_close(captured["final_norm.output"], saved)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("precision", ["", "bf16", None, True])
def test_training_rejects_unknown_norm_precision(precision):
    from llmtuner.config.training_config import TrainingConfig
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="grad_norm_precision"):
        TrainingConfig(grad_norm_precision=precision)


def test_training_norm_precision_is_explicit_and_visible():
    from llmtuner.config import LLMTunerConfig
    from llmtuner.config.training_config import TrainingConfig

    assert TrainingConfig().grad_norm_precision == "float32"
    config = LLMTunerConfig(training=TrainingConfig(grad_norm_precision="parameter"))
    assert config.grad_norm_precision == "parameter"


def test_candidate_trace_resolves_native_ep_router():
    from types import SimpleNamespace

    from tests.integration_tests.npu_model_forward_trace import trace_modules

    marker = object()
    gate = torch.nn.Linear(4, 2)
    projection = torch.nn.Linear(4, 4)
    attention = SimpleNamespace(
        q_norm=marker, k_norm=marker, o_proj=marker, q_proj=projection
    )
    layer = SimpleNamespace(
        self_attn=attention,
        mlp=SimpleNamespace(router=SimpleNamespace(gate=gate)),
        input_layernorm=marker,
        post_attention_layernorm=marker,
    )
    model = SimpleNamespace(
        tok_embeddings=marker,
        norm=marker,
        lm_head=marker,
        layers=[layer],
        model=SimpleNamespace(model=SimpleNamespace(rotary_emb=marker)),
    )
    modules = trace_modules(SimpleNamespace(model_parts=[model]), baseline=False)
    assert modules["layer0.router"] is layer.mlp.router
    assert modules["layer0.router_gate"] is gate
    assert modules["layer0.wq"] is projection
    assert "layer0.wk" not in modules
