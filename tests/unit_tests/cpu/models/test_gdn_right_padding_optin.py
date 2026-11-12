"""Legacy custom padding selectors must not silently enable native policies.

Native token-mask numerical coverage lives in test_gdn_padding_metadata and
collator/AC coverage in test_qwen35_collator_padding. Removed custom-backend
B1-only restrictions do not apply to native GDN.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.models.configuration import apply_runtime_config, unwrap_text_config
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextModel,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


def test_config_requires_explicit_backend_and_policy():
    assert ModelConfig().gdn_allow_right_padding is False
    for backend, policy in (
        ("native", "native"),
        ("npu_triton", "native"),
        ("native", "split_qkv_fp32_norm"),
    ):
        with pytest.raises(ConfigError, match="gdn_allow_right_padding"):
            ModelConfig(
                gdn_allow_right_padding=True,
                linear_attention_backend=backend,
                gdn_numerical_policy=policy,
            )
    with pytest.raises(ConfigError, match="boolean"):
        ModelConfig(gdn_allow_right_padding=1)


def test_factory_nested_text_propagation():
    cfg = SimpleNamespace(
        model=ModelConfig(
            gdn_allow_right_padding=True,
            linear_attention_backend="npu_triton",
            gdn_numerical_policy="split_qkv_fp32_norm",
        ),
        dataloader=SimpleNamespace(attention_mask_type=lambda: "causal"),
        norm_backend="native",
        qkv_projection="separate",
        rotary_init_device="constructor",
        attention_backend="auto",
    )
    outer = PretrainedConfig()
    outer.text_config = PretrainedConfig()
    apply_runtime_config(outer, cfg)
    assert unwrap_text_config(outer).gdn_allow_right_padding is True


@pytest.mark.parametrize("model_type", [Qwen3_5MoeForCausalLM, Qwen3_5MoeTextModel])
@pytest.mark.parametrize(
    "backend,policy",
    [
        ("native", "native"),
        ("npu_triton", "native"),
        ("native", "split_qkv_fp32_norm"),
        ("npu_triton", "split_qkv_fp32_norm"),
    ],
)
def test_custom_padding_request_fails_before_allocation(
    model_type, backend, policy, monkeypatch
):
    config = text_config(Qwen3_5MoeTextConfig)
    config.linear_attention_backend = backend
    config.gdn_numerical_policy = policy
    config.gdn_allow_right_padding = True
    before = config.to_dict()
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported padding policy reached allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    with pytest.raises(ValueError, match="explicit"):
        model_type(config)
    assert config.to_dict() == before
    assert torch.equal(torch.get_rng_state(), rng)
