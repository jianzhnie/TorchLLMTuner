"""Compare experiment-owned math against a separate Transformers reference."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as Reference,
)

from experiments.qwen3_moe_framework.config import small_qwen3_moe_config
from experiments.qwen3_moe_framework.model import Qwen3MoeForCausalLM


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_custom_model_matches_reference(tied, backend):
    config = small_qwen3_moe_config()
    config.tie_word_embeddings = tied
    config.mlp_only_layers = [1]
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(37)
    expected = Reference(deepcopy(config))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    actual = Qwen3MoeForCausalLM(deepcopy(config))
    assert torch.equal(rng, torch.get_rng_state())
    assert expected.state_dict().keys() == actual.state_dict().keys()
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name
    assert (actual.lm_head.weight is actual.model.embed_tokens.weight) == tied
    ids = torch.tensor([[1, 3, 5, 2], [4, 6, 2, 1]])
    positions = torch.arange(4).expand(2, -1)
    outputs = []
    for model in (expected, actual):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
        logits.square().sum().backward()
        optimizer.step()
        outputs.append(logits)
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-7, rtol=1e-5)
    for (name, reference), (_, custom) in zip(
        expected.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert (reference.grad is None) == (custom.grad is None), name
        if reference.grad is not None:
            torch.testing.assert_close(
                reference.grad, custom.grad, atol=1e-6, rtol=1e-5, msg=name
            )
        torch.testing.assert_close(reference, custom, atol=1e-7, rtol=1e-5, msg=name)


def test_model_constructs_without_llmtuner_or_transformers_modeling():
    import subprocess
    import sys

    source = """
import importlib.abc
import sys
class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "llmtuner" or fullname.startswith("llmtuner."):
            raise ImportError("Model must not depend on llmtuner")
        if ".modeling_qwen3_moe" in fullname:
            raise ImportError("Model must own its execution")
sys.meta_path.insert(0, BlockFramework())
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from experiments.qwen3_moe_framework.model import Qwen3MoeModel
config = Qwen3MoeConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
    moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
    num_key_value_heads=1, head_dim=4, num_experts=2, num_experts_per_tok=1,
    router_aux_loss_coef=0.0)
model = Qwen3MoeModel(config)
logits = model(torch.tensor([[1,2,3]]), position_ids=torch.arange(3)[None])
logits.square().sum().backward()
assert model.lm_head.weight.grad.count_nonzero() > 0
assert not any(
    name == "llmtuner" or name.startswith("llmtuner.") for name in sys.modules
)
"""
    subprocess.run([sys.executable, "-c", source], check=True, timeout=60)
