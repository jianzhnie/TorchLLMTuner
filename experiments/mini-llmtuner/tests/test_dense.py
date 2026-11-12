"""Independent HF dense oracle and explicit dense capability checks."""

import importlib

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

PACKAGE = "experiments.mini-llmtuner"
configuration = importlib.import_module(f"{PACKAGE}.config")
models = importlib.import_module(f"{PACKAGE}.model")
Trainer = importlib.import_module(f"{PACKAGE}.trainer").ExperimentTrainer


def test_dense_matches_hf_forward_and_gradients():
    config = configuration.load_model_config(None, "qwen3")
    config._attn_implementation = "sdpa"
    model = models.Qwen3Model(config)
    reference = Qwen3ForCausalLM(config)
    reference.load_state_dict(model.model.state_dict(), strict=True)
    tokens = torch.tensor([[1, 3, 7, 9], [2, 5, 8, 4]])
    positions = torch.arange(4).expand(2, -1)
    actual = model(tokens, position_ids=positions)
    expected = reference(tokens, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(
            model.model.get_parameter(name).grad,
            parameter.grad,
            atol=1e-5,
            rtol=1e-4,
        )
    with pytest.raises(ValueError, match="EP requires"):
        model.validate_parallel(tp_degree=1, ep_degree=2)


def test_dense_train_and_resume(tmp_path):
    config = configuration.ExperimentConfig(architecture="qwen3", steps=2)
    full = Trainer(config, torch.device("cpu"))
    full.train()
    split = Trainer(
        configuration.ExperimentConfig(
            architecture="qwen3",
            steps=1,
            checkpoint=tmp_path,
        ),
        torch.device("cpu"),
    )
    split.train()
    restored = Trainer(
        configuration.ExperimentConfig(
            architecture="qwen3",
            steps=2,
            resume=tmp_path,
        ),
        torch.device("cpu"),
    )
    restored.train()
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[name], atol=0, rtol=0
        )
    assert restored.assembly.parameter_mappings
    assert restored.model.contract.architecture == "qwen3"
