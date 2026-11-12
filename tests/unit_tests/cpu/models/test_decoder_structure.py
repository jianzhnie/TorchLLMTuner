"""Explicit structural binding must preserve storage and initialization."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM


def small_config():
    return Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=True,
    )


def test_binding_preserves_rng_parameters_and_state():
    model = Qwen3MoeForCausalLM(small_config())
    state = tuple(model.state_dict())
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    bound = model.decoder_structure.bind(model)
    assert bound.decoder is model.model
    assert bound.blocks is model.model.layers
    assert bound.embedding.weight is bound.head.weight
    assert tuple(model.state_dict()) == state
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_declaration_rejects_missing_or_misowned_targets():
    model = Qwen3MoeForCausalLM(small_config())
    with pytest.raises(AttributeError):
        replace(model.decoder_structure, norm="model.missing").bind(model)
    with pytest.raises(ValueError, match="direct child"):
        replace(model.decoder_structure, embedding="lm_head").bind(model)
    with pytest.raises(TypeError, match="ModuleList"):
        replace(model.decoder_structure, blocks="model.norm").bind(model)


def test_explicit_structure_rebinds_after_module_replacement():
    model = Qwen3MoeForCausalLM(small_config())
    structure = model.decoder_structure
    before = structure.bind(model)
    replacement = nn.LayerNorm(16)
    model.model.norm = replacement
    after = structure.bind(model)
    assert after.norm is replacement
    assert before.norm is not after.norm
    assert after.embedding is before.embedding
    assert not any(key.startswith("_decoder.") for key in model.state_dict())


def test_meta_binding_does_not_materialize_parameters():
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(small_config())
    bound = model.decoder_structure.bind(model)
    assert bound.embedding.weight.is_meta
    assert bound.embedding.weight is bound.head.weight


@pytest.mark.parametrize(
    ("role", "path"),
    [
        (role, path)
        for role in ("decoder", "embedding", "blocks", "norm", "head")
        for path in (None, "", ".model", "model.", "model..layers")
        if (role, path) != ("decoder", "")
    ],
)
def test_declaration_rejects_malformed_paths_before_binding(role, path):
    values = dict(
        decoder="model",
        embedding="model.embed_tokens",
        blocks="model.layers",
        norm="model.norm",
        head="lm_head",
    )
    values[role] = path
    with pytest.raises(ValueError, match="module path|direct child"):
        DecoderStructure(**values)


@pytest.mark.parametrize("name", ["", "self.attn", 1])
def test_attention_declaration_requires_a_direct_child(name):
    with pytest.raises(ValueError, match="attention.*direct child"):
        DecoderStructure(
            decoder="model",
            embedding="model.embed_tokens",
            blocks="model.layers",
            norm="model.norm",
            head="lm_head",
            attention=name,
        )


def test_root_decoder_can_declare_direct_children():
    model = nn.Module()
    model.embedding = nn.Embedding(4, 2)
    model.layers = nn.ModuleList([nn.Linear(2, 2)])
    model.norm = nn.LayerNorm(2)
    model.head = nn.Linear(2, 4)
    structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="layers",
        norm="norm",
        head="head",
    )
    bound = structure.bind(model)
    assert bound.decoder is model
    assert bound.blocks is model.layers


@pytest.mark.parametrize("role", ["decoder", "embedding", "blocks", "norm", "head"])
def test_binding_rejects_unregistered_module_attributes(role):
    model = Qwen3MoeForCausalLM(small_config())
    structure = model.decoder_structure
    path = getattr(structure, role)
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path)
    module = parent._modules.pop(name)
    object.__setattr__(parent, name, module)
    assert model.get_submodule(path) is module
    with pytest.raises(AttributeError, match="not registered"):
        structure.bind(model)


def test_attention_binding_rejects_unregistered_block_collection():
    model = Qwen3MoeForCausalLM(small_config())
    blocks = model.model._modules.pop("layers")
    object.__setattr__(model.model, "layers", blocks)
    with pytest.raises(AttributeError, match="not registered"):
        model.decoder_structure.bind_attentions(model)
