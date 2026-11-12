"""Native Qwen3.5 routing math is owned locally, with reference parity."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter as ReferenceRouter,
)

from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from llmtuner.models.qwen.routing import Qwen3_5MoeTopKRouter
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_native_router_outputs_and_gradients_match_reference(dtype, top_k, monkeypatch):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.num_experts = 4
    config.num_experts_per_tok = top_k
    torch.manual_seed(101)
    model = Qwen3_5MoeForCausalLM(config).to(dtype)
    router = model.layers[0].mlp.gate
    assert type(router) is Qwen3_5MoeTopKRouter
    reference = ReferenceRouter(deepcopy(config)).to(dtype)
    reference.load_state_dict(router.state_dict(), strict=True)
    hidden = torch.randn(2, 3, config.hidden_size, dtype=dtype).transpose(0, 1)
    hidden.requires_grad_()
    other = hidden.detach().clone().requires_grad_()
    expected_logits, expected_scores, expected_ids = reference(other)

    def forbidden(*args, **kwargs):
        raise AssertionError("Owned native router delegated reference forward")

    monkeypatch.setattr(ReferenceRouter, "forward", forbidden)
    logits, scores, indices = router(hidden)
    assert logits.dtype == scores.dtype == dtype
    assert torch.equal(indices, expected_ids)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    assert logits.isfinite().all() and logits.count_nonzero() > 0
    assert scores.isfinite().all() and scores.count_nonzero() > 0
    logit_probe, score_probe = torch.randn_like(logits), torch.randn_like(scores)
    torch.autograd.backward((logits, scores), (logit_probe, score_probe))
    torch.autograd.backward(
        (expected_logits, expected_scores), (logit_probe, score_probe)
    )
    for actual, expected in (
        (hidden.grad, other.grad),
        (router.weight.grad, reference.weight.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.isfinite().all() and actual.count_nonzero() > 0
