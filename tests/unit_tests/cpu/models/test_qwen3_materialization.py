"""Model-owned buffer reconstruction and allocated-parameter initialization."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from llmtuner.config import LLMTunerConfig, ModelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.trainer.runtime.assembly import _build_model
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("method", ["initialize_parameters", "materialization_buffers"])
def test_meta_construction_requires_model_lifecycle_before_allocation(
    monkeypatch, method
):
    cfg = LLMTunerConfig(model=ModelConfig(initialization_device="meta"))

    def reject(*args, **kwargs):
        raise AssertionError(
            "Missing lifecycle must fail before constructing the model"
        )

    monkeypatch.setattr(Qwen3MoeForCausalLM, method, None)
    monkeypatch.setattr(Qwen3MoeForCausalLM, "__init__", reject)
    with pytest.raises(UnsupportedCombinationError, match=method):
        _build_model(SimpleNamespace(cfg=cfg, parallel_dims=None))


def test_model_owned_meta_lifecycle_preserves_keys_aliases_and_initializer_rng():
    config = model_config()
    reference = Qwen3MoeForCausalLM(deepcopy(config))
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(deepcopy(config))
    parameter = model.model.embed_tokens.weight
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    buffers = model.materialization_buffers(torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    assert model.model.rotary_emb.inv_freq.is_meta
    materialize_meta_model(model, torch.device("cpu"), runtime_buffers=buffers)
    assert model.model.embed_tokens.weight is model.lm_head.weight is parameter
    assert tuple(model.state_dict()) == keys
    torch.testing.assert_close(
        model.model.rotary_emb.inv_freq,
        reference.model.rotary_emb.inv_freq,
        rtol=0,
        atol=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(193)
        reference.apply(reference._init_weights)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(193)
        model.initialize_parameters()
        actual_rng = torch.get_rng_state().clone()
    assert torch.equal(expected_rng, actual_rng)
    expected = reference.state_dict()
    assert len(expected) > 10
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert torch.count_nonzero(model.model.embed_tokens.weight) > 0


def test_initializer_rejects_unmaterialized_parameters_without_consuming_rng():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(model_config())
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="materialized before initialization"):
        model.initialize_parameters()
    assert torch.equal(rng, torch.get_rng_state())


def test_initializer_rejects_converted_experts_before_consuming_rng():
    model = Qwen3MoeForCausalLM(model_config())
    model.model.layers[0].mlp = torch.nn.Identity()
    rng = torch.get_rng_state().clone()
    with pytest.raises(TypeError, match="own initialization contract"):
        model.initialize_parameters()
    assert torch.equal(rng, torch.get_rng_state())


def test_buffer_reconstruction_does_not_ignore_rotary_policy():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(model_config())
    model.config.rotary_init_device = "target"
    with pytest.raises(ValueError, match="constructor RoPE"):
        model.materialization_buffers(torch.device("cpu"))
    assert model.model.rotary_emb.inv_freq.is_meta
