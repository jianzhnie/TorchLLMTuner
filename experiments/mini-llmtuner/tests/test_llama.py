"""Llama native HF numerical oracle, train/update and exact local recovery."""

import importlib

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

PACKAGE = "experiments.mini-llmtuner"
configuration = importlib.import_module(f"{PACKAGE}.config")
LlamaModel = importlib.import_module(f"{PACKAGE}.llama").LlamaModel
Trainer = importlib.import_module(f"{PACKAGE}.trainer").ExperimentTrainer


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("tied", [False, True])
def test_llama_matches_hf_training(backend, tied):
    config = configuration.load_model_config(None, "llama")
    config._attn_implementation = backend
    config.tie_word_embeddings = tied
    model = LlamaModel(config)
    reference = LlamaForCausalLM(config)
    reference.load_state_dict(model.model.state_dict(), strict=True)
    assert model.enable_weight_tying == tied
    tokens = torch.tensor([[1, 3, 7, 9], [2, 5, 8, 4]])
    positions = torch.arange(4).expand(2, -1)
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=False)
        for m in (model, reference)
    ]
    for _ in range(3):
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        actual = model(tokens, position_ids=positions)
        expected = reference(tokens, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        targets = tokens.roll(-1, dims=1)
        for logits in (actual, expected):
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
            ).backward()
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                model.model.get_parameter(name).grad,
                parameter.grad,
                atol=1e-6,
                rtol=1e-5,
            )
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                model.model.get_parameter(name),
                parameter,
                atol=1e-6,
                rtol=1e-5,
            )
    with pytest.raises(ValueError, match="EP requires"):
        model.validate_parallel(tp_degree=1, ep_degree=2)


def test_llama_resume(tmp_path):
    def trainer(steps, **kwargs):
        return Trainer(
            configuration.ExperimentConfig(
                architecture="llama",
                steps=steps,
                **kwargs,
            ),
            torch.device("cpu"),
        )

    full = trainer(3)
    full.train()
    split = trainer(1, checkpoint=tmp_path)
    split.train()
    restored = trainer(3, resume=tmp_path)
    restored.train()
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[name], atol=0, rtol=0
        )
    assert not restored.model.contract.tp_gradient_sums
    assert restored.assembly.parameter_mappings


def test_llama_rejects_unsupported_mlp_bias():
    config = configuration.load_model_config(None, "llama")
    config.mlp_bias = True
    with pytest.raises(ValueError, match="MLP bias"):
        LlamaModel(config)
