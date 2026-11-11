"""Explicit family construction preserves initialization and training APIs."""

from copy import deepcopy

import pytest
import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM as ReferenceDeepseek,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceQwen,
)

from llmtuner.models.deepseek.deepseek_v4.modeling import DeepseekV4ForCausalLM
from llmtuner.models.explicit import build_explicit_model
from llmtuner.models.hf.factory import resolve_model_class
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_deepseek_v4 import small_config
from tests.unit_tests.cpu.models.test_qwen35_rotary_initialization import text_config


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
def test_owned_family_wrapper_binds_without_candidate_inspection(family, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Declared decoder must not search candidate attributes")

    monkeypatch.setattr("llmtuner.models.hf.model.resolve_submodule_name", reject)
    config = (
        small_config()
        if family == "deepseek"
        else text_config(ReferenceQwen.config_class)
    )
    wrapper = HFTransformerModel(config)
    bound = wrapper.model.decoder_structure.bind(wrapper.model)
    assert wrapper.tok_embeddings is bound.embedding
    assert wrapper.layers is bound.blocks
    assert wrapper.norm is bound.norm
    assert wrapper.lm_head is bound.head
    assert not any(key.startswith("_decoder.") for key in wrapper.state_dict())


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
@pytest.mark.parametrize("expert_backend", [None, "eager"])
def test_explicit_family_preserves_initialization_forward_and_checkpoint(
    family, expert_backend
):
    if family == "deepseek":
        config = small_config()
        reference_type, owned_type = ReferenceDeepseek, DeepseekV4ForCausalLM
    else:
        config = text_config(ReferenceQwen.config_class)
        config.num_hidden_layers = 2
        config.layer_types = ["linear_attention", "full_attention"]
        reference_type, owned_type = ReferenceQwen, Qwen3_5MoeForCausalLM
    config._attn_implementation = "eager"
    config._experts_implementation = expert_backend
    assert resolve_model_class(config) is owned_type
    torch.manual_seed(91)
    reference = reference_type(deepcopy(config))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    constructor = build_explicit_model if family == "deepseek" else owned_type
    owned = constructor(deepcopy(config))
    assert type(owned) is owned_type
    assert (
        owned.config._experts_implementation == reference.config._experts_implementation
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert type(owned.model) is not type(reference.model)
    assert type(owned.model.layers[0]) is not type(reference.model.layers[0])
    assert owned.model.layers[0].mlp.router_logits_in_output is False
    assert reference.state_dict().keys() == owned.state_dict().keys()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, owned.state_dict()[name]), name
    tokens = torch.tensor([[1, 3, 2, 7, 4, 6, 8, 5]])
    outputs = []
    for model in (reference, owned):
        result = model(input_ids=tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(result.loss) and result.loss.item() > 0
        result.loss.backward()
        outputs.append(result)
    assert torch.equal(outputs[0].logits, outputs[1].logits)
    assert torch.equal(outputs[0].loss, outputs[1].loss)
    gradients = 0
    for (name, expected), (_, actual) in zip(
        reference.named_parameters(), owned.named_parameters(), strict=True
    ):
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            gradients += int(bool(expected.grad.count_nonzero()))
            assert torch.equal(expected.grad, actual.grad), name
    assert gradients > 0
    restored = constructor(deepcopy(config))
    restored.load_state_dict(owned.state_dict(), strict=True)
    assert torch.equal(
        restored(input_ids=tokens, use_cache=False).logits, outputs[1].logits
    )


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
@pytest.mark.parametrize("tied", [False, True])
def test_owned_family_declares_moe_slots_and_actual_weight_identity(family, tied):
    config = (
        small_config()
        if family == "deepseek"
        else text_config(ReferenceQwen.config_class)
    )
    config.tie_word_embeddings = tied
    constructor = (
        DeepseekV4ForCausalLM if family == "deepseek" else Qwen3_5MoeForCausalLM
    )
    model = constructor(config)
    assert model.enable_weight_tying is tied
    bound = model.decoder_structure.bind(model)
    for layer in bound.blocks:
        assert layer.moe_block_name == "mlp"
        assert layer._modules[layer.moe_block_name] is layer.mlp
    # The contract follows live Parameter identity, including manual untie/retie.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert model.enable_weight_tying is False
    model.lm_head.weight = model.model.embed_tokens.weight
    assert model.enable_weight_tying is True
