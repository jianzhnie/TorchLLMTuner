"""The production training path requires registered model-owned structure."""

import pytest
import torch
from transformers import AutoConfig, Qwen3MoeConfig

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.models.configuration import (
    build_model_config,
    resolve_model_class,
    unwrap_text_config,
)
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.registry import get_model_spec


def test_unknown_model_has_no_transformers_fallback():
    config = Qwen3MoeConfig()
    config.model_type = "unregistered_model"
    with pytest.raises(ValueError, match="fallback is disabled"):
        resolve_model_class(config)


def test_registered_runtime_trains_without_hf_adapter():
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = Qwen3MoeForCausalLM(config)
    assert type(model).__module__ == "llmtuner.models.qwen.qwen3.model"
    assert not hasattr(model, "_bind_legacy_decoder")
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.01, foreach=False)
    ids = torch.tensor([1, 3, 7, 9])
    positions = torch.arange(4)
    before = model.tok_embeddings.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(ids, positions=positions).logits
        assert logits.numel() == 128
        assert torch.isfinite(logits).all()
        logits.square().mean().backward()
        optimizer.step()
    assert not torch.equal(before, model.tok_embeddings.weight)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


def test_runtime_requires_model_owned_decoder_structure(monkeypatch):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    monkeypatch.setattr(Qwen3MoeForCausalLM, "decoder_structure", None)
    model = Qwen3MoeForCausalLM(config)
    assert model.decoder_structure is None


def test_default_architecture_has_local_registration():
    specification = get_model_spec(ModelConfig().model_name_or_path)
    assert specification is not None
    assert specification.resolve_model_class().__module__.startswith("llmtuner.models.")


def test_runtime_ignores_private_tp_plan_aliases(monkeypatch):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = Qwen3MoeForCausalLM(config)
    monkeypatch.setattr(Qwen3MoeForCausalLM, "tp_config", property(lambda self: {}))
    model.model._tp_config = {"wrong_target": "colwise"}
    model.model._tp_plan = {"another_wrong_target": "rowwise"}
    assert model.tp_config == {}


def test_config_roundtrip_uses_local_registration(tmp_path, monkeypatch):
    def reject_global_registration(*args, **kwargs):
        raise AssertionError("Loading must not mutate AutoConfig's global registry")

    monkeypatch.setattr(AutoConfig, "register", reject_global_registration)
    original = build_model_config(
        "qwen3_moe",
        seq_len=32,
        arch_overrides={"hidden_size": 16, "num_hidden_layers": 1},
    )
    original.save_pretrained(tmp_path)
    loaded = build_model_config(str(tmp_path), seq_len=64)
    assert type(loaded) is type(original)
    assert loaded.hidden_size == 16
    assert loaded.num_hidden_layers == 1
    assert loaded.max_position_embeddings >= 64
    assert resolve_model_class(loaded) is Qwen3MoeForCausalLM


def test_unregistered_offline_config_is_rejected():
    with pytest.raises(ValueError, match="No local configuration registered"):
        build_model_config("llama", seq_len=32, arch_overrides={})


def test_unwrap_does_not_invent_external_architecture_names():
    composite = Qwen3MoeConfig()
    composite.text_config = Qwen3MoeConfig(architectures=None)
    text = unwrap_text_config(composite)
    assert text.architectures is None
    assert resolve_model_class(text) is Qwen3MoeForCausalLM


def test_unregistered_saved_config_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    with pytest.raises(ValueError, match="No local configuration registered"):
        build_model_config(str(tmp_path), seq_len=32)


def test_cli_geometry_fill_uses_registered_configuration(tmp_path, monkeypatch):
    config = Qwen3MoeConfig(
        vocab_size=48,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config.save_pretrained(tmp_path)

    def reject_automatic_loading(*args, **kwargs):
        raise AssertionError("CLI must use the registered configuration class")

    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_automatic_loading)
    cfg = LLMTunerConfig(model=ModelConfig(model_name_or_path=str(tmp_path)))
    cfg.auto_fill_model()
    assert cfg.model.hidden_size == 32
    assert cfg.model.vocab_size == 48
    assert cfg.model.num_hidden_layers == 1
