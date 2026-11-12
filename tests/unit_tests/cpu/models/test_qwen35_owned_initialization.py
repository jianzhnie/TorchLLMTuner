"""Qwen3.5 text projections use model-owned guarded initialization."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel as ReferencePretrained,
)

from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    initialize_model_weights,
)
from llmtuner.models.qwen.qwen3_5_moe.rotary import Qwen3_5MoeTextRotaryEmbedding
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("std", [0.0, 0.02])
@pytest.mark.parametrize("tied", [False, True])
def test_projection_initialization_matches_whole_model_reference(
    monkeypatch, std, tied
):
    config = text_config(ReferenceModel.config_class)
    config.num_hidden_layers = 2
    config.layer_types = ["linear_attention", "full_attention"]
    config.initializer_range = std
    config.tie_word_embeddings = tied
    config.pad_token_id = 0
    torch.manual_seed(127)
    expected = ReferenceModel(deepcopy(config))
    expected_rng = torch.get_rng_state().clone()
    original = ReferencePretrained._init_weights

    def guarded(self, module):
        if isinstance(
            module, (nn.Linear, nn.Embedding, nn.Conv1d, Qwen3_5MoeTextRotaryEmbedding)
        ):
            pytest.fail("Explicit text initialization delegated to reference")
        return original(self, module)

    monkeypatch.setattr(ReferencePretrained, "_init_weights", guarded)
    torch.manual_seed(127)
    actual = Qwen3_5MoeForCausalLM(deepcopy(config))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.enable_weight_tying == tied
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["linear", "embedding", "conv"])
@pytest.mark.parametrize("loaded", [False, True])
def test_projection_initialization_respects_loaded_tensor_markers(kind, loaded):
    modules = {
        "linear": lambda: nn.Linear(4, 8),
        "embedding": lambda: nn.Embedding(8, 4, padding_idx=0),
        "conv": lambda: nn.Conv1d(4, 4, 3, groups=4),
    }
    module = modules[kind]()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.25)
            parameter._is_hf_initialized = loaded
    identities = tuple(id(parameter) for parameter in module.parameters())
    rng = torch.get_rng_state().clone()
    assert initialize_model_weights(module, 0.02)
    assert identities == tuple(id(parameter) for parameter in module.parameters())
    if loaded:
        assert torch.equal(torch.get_rng_state(), rng)
        for parameter in module.parameters():
            torch.testing.assert_close(
                parameter, torch.full_like(parameter, 0.25), rtol=0, atol=0
            )
    else:
        assert not torch.equal(torch.get_rng_state(), rng)
        assert module.weight.count_nonzero() > 0
        if kind == "embedding":
            assert module.weight[0].count_nonzero() == 0
        else:
            assert module.bias.count_nonzero() == 0
