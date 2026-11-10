"""Policy ownership must be checked before mutating any HF instance."""

from types import MethodType, SimpleNamespace

import pytest
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.hf.expert_kernels import (
    stabilize_moe_experts,
    stabilize_moe_router,
)


@pytest.mark.parametrize("policy", ["router", "experts"])
def test_policy_rejects_late_custom_forward_without_partial_installation(policy):
    config = AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    kind = Qwen3MoeTopKRouter if policy == "router" else Qwen3MoeSparseMoeBlock
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3_moe",
        router_compute_dtype="float32",
        experts_implementation="torchtitan",
        hidden_act="silu",
    )
    model.targets = nn.ModuleList([kind(config), kind(config)])

    def external_forward(self, inputs):
        return inputs

    model.targets[1].forward = MethodType(external_forward, model.targets[1])
    first_forward = model.targets[0].forward.__func__
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    install = stabilize_moe_router if policy == "router" else stabilize_moe_experts
    with pytest.raises(UnsupportedCombinationError, match="external custom"):
        install(model)
    assert model.targets[0].forward.__func__ is first_forward
    assert model.targets[1].forward.__func__ is external_forward
    assert "_llmtuner_local_dispatcher" not in vars(model.targets[0])
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert tuple(model.state_dict()) == keys

    del model.targets[1].forward
    install(model)
    installed = [target.forward.__func__ for target in model.targets]
    install(model)
    assert [target.forward.__func__ for target in model.targets] == installed
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert tuple(model.state_dict()) == keys
