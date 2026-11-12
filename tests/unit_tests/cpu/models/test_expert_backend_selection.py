"""Explicit expert requests select the invoked backend before model allocation."""

from copy import deepcopy

import pytest
import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.models.configuration import (
    build_model_config_for,
    configure_reference_experts,
    resolve_model_class,
)
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import (
    small_config,
    text_config,
)
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("family", ["qwen3", "qwen35", "deepseek"])
@pytest.mark.parametrize("backend", ["eager", "grouped_mm", "batched_mm"])
def test_explicit_expert_backend_executes_and_preserves_training(
    family, backend, monkeypatch
):
    if family == "qwen3":
        config = model_config()
    elif family == "qwen35":
        config = text_config(Qwen3_5MoeTextConfig)
    else:
        config = small_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    constructor = resolve_model_class(config)
    torch.manual_seed(93)
    reference = constructor(deepcopy(config))
    rng = torch.get_rng_state().clone()
    config.experts_implementation = backend
    torch.manual_seed(93)
    actual = constructor(config)
    assert config._experts_implementation == backend
    assert torch.equal(torch.get_rng_state(), rng)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 2, 9]])
    expected = reference(tokens).logits
    calls = []
    get_interface = ALL_EXPERTS_FUNCTIONS.get_interface

    def observed_interface(requested, default):
        assert requested == backend
        implementation = get_interface(requested, default)

        def observed(module, *args, **kwargs):
            calls.append(module)
            return implementation(module, *args, **kwargs)

        return observed

    # Eager selects the model's original forward rather than a registry entry.
    # Observe resolution and execution without changing the implementation.
    monkeypatch.setattr(ALL_EXPERTS_FUNCTIONS, "get_interface", observed_interface)
    result = actual(tokens).logits
    targets = {
        id(layer.mlp.experts)
        for layer in actual.layers
        if layer.moe_block_name is not None
    }
    assert calls and len(calls) == len(targets)
    assert {id(module) for module in calls} == targets
    assert result.isfinite().all() and result.count_nonzero() > 0
    # Different GEMM decompositions may change FP32 summation order. Keep the
    # predefined 1e-5 relative / 1e-6 absolute bound for logits and gradients.
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    result.square().sum().backward()
    gradients = 0
    for name, parameter in actual.named_parameters():
        baseline = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (baseline.grad is None), name
        if parameter.grad is not None:
            gradients += int(bool(parameter.grad.count_nonzero()))
            torch.testing.assert_close(
                parameter.grad, baseline.grad, rtol=1e-5, atol=1e-6
            )
    assert gradients > 0
    for model in (reference, actual):
        torch.optim.SGD(model.parameters(), lr=0.01, foreach=False).step()
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter, dict(reference.named_parameters())[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("backend", ["torchtitan", "pre_down", "invalid", False, []])
def test_unimplemented_expert_strategy_rejected_before_allocation(backend):
    config = model_config()
    config.experts_implementation = backend
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model-owned expert implementation"):
        resolve_model_class(config)(config)
    assert torch.equal(torch.get_rng_state(), rng)


def test_native_preserves_existing_reference_backend():
    config = model_config()
    config._experts_implementation = "eager"
    config.experts_implementation = "native"
    configure_reference_experts(config)
    assert config._experts_implementation == "eager"


@pytest.mark.parametrize(
    "backend", ["grouped_mm", "batched_mm", "torchtitan", "pre_down"]
)
def test_kimi_rejects_unimplemented_expert_backends_before_allocation(backend):
    config = KimiLinearConfig()
    config.experts_implementation = backend
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Kimi local experts support"):
        KimiLinearForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)


def test_training_configuration_reaches_explicit_expert_constructor():
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            experts_implementation="eager",
            arch_overrides=model_config().to_dict(),
        ),
    )
    config = build_model_config_for(cfg)
    assert config.experts_implementation == "eager"
    model = resolve_model_class(config)(config)
    assert model.config._experts_implementation == "eager"
    assert model.layers[0].mlp.experts.config._experts_implementation == "eager"
