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
from llmtuner.models.hf.factory import resolve_model_class
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_deepseek_v4 import small_config
from tests.unit_tests.cpu.models.test_qwen35_rotary_initialization import text_config


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
    owned = owned_type(deepcopy(config))
    assert (
        owned.config._experts_implementation == reference.config._experts_implementation
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert type(owned.model) is not type(reference.model)
    assert type(owned.model.layers[0]) is not type(reference.model.layers[0])
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
    restored = owned_type(deepcopy(config))
    restored.load_state_dict(owned.state_dict(), strict=True)
    assert torch.equal(
        restored(input_ids=tokens, use_cache=False).logits, outputs[1].logits
    )
