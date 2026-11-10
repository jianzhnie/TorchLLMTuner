"""Local DeepSeek MoE accepts strided activations without changing gradients."""

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
)

from llmtuner.models.deepseek.deepseek_v4.modeling import DeepseekV4SparseMoeBlock
from tests.unit_tests.cpu.models.test_deepseek_v4 import small_config


def test_moe_strided_input_matches_contiguous_forward_and_backward():
    config = small_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(61)
    reference = DeepseekV4ForCausalLM(config).model.layers[1].mlp
    actual = DeepseekV4SparseMoeBlock(config, 1)
    actual.load_state_dict(reference.state_dict(), strict=True)
    source = torch.randn(2, 6, config.hidden_size)
    strided = source[:, ::2].detach().requires_grad_()
    dense = strided.detach().contiguous().requires_grad_()
    assert not strided.is_contiguous()
    expected = reference(dense)
    output = actual(strided)
    assert output.count_nonzero() > 0
    assert torch.equal(output, expected)
    expected.square().sum().backward()
    output.square().sum().backward()
    assert torch.equal(strided.grad, dense.grad)
    for (name, parameter), (_, reference_parameter) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (parameter.grad is None) == (reference_parameter.grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, reference_parameter.grad), name
