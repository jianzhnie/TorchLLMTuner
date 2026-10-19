"""Dynamic expert counts must compile without data-dependent Python branches."""

import copy

import pytest
import torch
from torch.nn import functional as F

from llmtuner.models.common.moe.experts import GroupedExperts


@pytest.mark.parametrize("counts", [(2, 0, 3), (0, 5, 0), (0, 0, 0)])
def test_compiled_experts_match_independent_segments(counts):
    torch.manual_seed(77)
    reference = GroupedExperts(4, 8, 3, use_grouped_mm=False).double()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    actual = copy.deepcopy(reference)
    inputs = torch.randn(sum(counts), 4, dtype=torch.float64, requires_grad=True)
    compiled_inputs = inputs.detach().clone().requires_grad_()
    outputs = []
    for expert, segment in enumerate(inputs.split(counts)):
        gate = F.linear(segment, reference.w1_EFD[expert])
        up = F.linear(segment, reference.w3_EFD[expert])
        outputs.append(F.linear(F.silu(gate) * up, reference.w2_EDF[expert]))
    expected = torch.cat(outputs)
    compiled = torch.compile(actual, backend="aot_eager", fullgraph=True)
    result = compiled(compiled_inputs, torch.tensor(counts))
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    result.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        compiled_inputs.grad, inputs.grad, rtol=1e-12, atol=1e-12
    )
    for parameter, target in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, target.grad, rtol=1e-12, atol=1e-12)
