"""Model-owned assembly preserves reference initialization and training."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeModel as ReferenceDecoder,
)

from llmtuner.models.hf.factory import resolve_model_class
from llmtuner.models.qwen.qwen3.model import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("experts", [None, "eager"])
def test_owned_qwen3_matches_reference_initialization_and_update(
    tied, backend, experts
):
    config = Qwen3MoeConfig(
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
        mlp_only_layers=[1],
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = backend
    config._experts_implementation = experts
    assert resolve_model_class(config) is Qwen3MoeForCausalLM
    torch.manual_seed(37)
    expected = ReferenceModel(deepcopy(config))
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    actual = Qwen3MoeForCausalLM(deepcopy(config))
    assert (
        actual.config._experts_implementation == expected.config._experts_implementation
    )
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert type(actual.model.layers[0]) is Qwen3MoeDecoderLayer
    assert (actual.model.embed_tokens.weight is actual.lm_head.weight) == tied
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], value), name

    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 2, 9]])
    outputs = []
    for model in (expected, actual):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        output = model(input_ids=tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(output.loss) and output.loss.item() > 0
        output.loss.backward()
        outputs.append(output)
        optimizer.step()
    assert torch.equal(outputs[0].logits, outputs[1].logits)
    assert torch.equal(outputs[0].loss, outputs[1].loss)
    for (name, reference), (_, owned) in zip(
        expected.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert (reference.grad is None) == (owned.grad is None), name
        if reference.grad is not None:
            assert torch.equal(reference.grad, owned.grad), name
        assert torch.equal(reference, owned), name


@pytest.mark.parametrize(
    ("owned", "reference"),
    [(Qwen3MoeModel, ReferenceDecoder), (Qwen3MoeForCausalLM, ReferenceModel)],
)
def test_owned_experts_capability_delegates_to_reference(owned, reference):
    assert owned._can_set_experts_implementation() == (
        reference._can_set_experts_implementation()
    )
