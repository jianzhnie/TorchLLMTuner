"""Experimental opt-in guards/ownership only; not NPU/FSDP acceptance."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    apply_mask_to_padding_states,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.factory import apply_runtime_config, unwrap_text_config
from llmtuner.models.hf.gdn_numerics import (
    configure_gdn_numerics,
    validate_gdn_numerical_batch,
    validate_gdn_numerical_config,
    validate_gdn_numerical_sequence,
)
from llmtuner.models.hf.linear_attention import (
    validate_linear_attention_batch,
    validate_linear_attention_config,
    validate_linear_attention_sequence,
)


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


def config():
    c = AutoConfig.for_model("qwen3_5_moe_text")
    c.gdn_numerical_policy = "split_qkv_fp32_norm"
    c.linear_attention_backend = "npu_triton"
    c.gdn_allow_right_padding = True
    return c


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


@pytest.mark.parametrize(
    "validator", [validate_gdn_numerical_config, validate_linear_attention_config]
)
def test_raw_hf_config_dependency_cannot_bypass_model_config(validator):
    c = config()
    validator(c)
    c.linear_attention_backend = "native"
    with pytest.raises(UnsupportedCombinationError, match="gdn_allow_right_padding"):
        validator(c)


@pytest.mark.parametrize(
    "validator", [validate_gdn_numerical_sequence, validate_linear_attention_sequence]
)
def test_sequence_optin_polarity_and_original_mask_unchanged(validator):
    ids = torch.arange(128)
    padding = ids >= 93
    keep = (~padding)[None, :]
    validator(ids, ids, keep, padding, allow_right_padding=True)
    assert torch.equal(keep, (ids < 93)[None, :])
    with pytest.raises(UnsupportedCombinationError):
        validator(ids, ids, None, padding)
    for bad in (torch.ones(128, dtype=torch.bool), ids < 35, ids == 50):
        with pytest.raises(UnsupportedCombinationError):
            validator(ids, ids, None, bad, allow_right_padding=True)
    positions = ids.clone()
    positions[93:] = torch.arange(35)
    validator(ids, positions, None, padding, allow_right_padding=True)
    with pytest.raises(UnsupportedCombinationError):
        validator(ids, positions, None, None, allow_right_padding=True)
    positions[50] = 0
    with pytest.raises(UnsupportedCombinationError):
        validator(ids, positions, None, padding, allow_right_padding=True)
    with pytest.raises(UnsupportedCombinationError):
        validator(
            ids, ids, torch.ones(1, 1, 128, 128), padding, allow_right_padding=True
        )


@pytest.mark.parametrize(
    "validator", [validate_gdn_numerical_batch, validate_linear_attention_batch]
)
def test_batch_optin_still_rejects_b2_and_empty_prefix(validator):
    ids = torch.arange(128)[None, :]
    validator({"input": ids, "padding_mask": ids >= 93}, allow_right_padding=True)
    with pytest.raises(UnsupportedCombinationError):
        validator({"input": ids, "padding_mask": ids >= 93})
    for batch in (
        {"input": ids.expand(2, -1)},
        {"input": ids, "padding_mask": torch.ones_like(ids, dtype=torch.bool)},
    ):
        with pytest.raises(UnsupportedCombinationError):
            validator(batch, allow_right_padding=True)


def test_module_stamp_identity_and_original_hf_b1_mask_semantics():
    holder = nn.Module()
    holder.config = config()
    with torch.device("meta"):
        holder.gdn = Qwen3_5MoeGatedDeltaNet(holder.config, 0).bfloat16()
    before = dict(holder.named_parameters())
    configure_gdn_numerics(holder)
    assert holder.gdn._llmtuner_gdn_allow_right_padding is True
    assert all(dict(holder.named_parameters())[k] is p for k, p in before.items())
    holder.config.gdn_allow_right_padding = False
    configure_gdn_numerics(holder)
    assert holder.gdn._llmtuner_gdn_allow_right_padding is False
    x = torch.randn(1, 128, 2048).bfloat16()
    keep = (torch.arange(128) < 93)[None, :]
    masked = apply_mask_to_padding_states(x, keep)
    assert torch.equal(masked[:, :93], x[:, :93])
    # Old HF ignored B1 padding; newer HF also zeros the invalid suffix.
    assert torch.equal(masked[:, 93:], x[:, 93:]) or not masked[:, 93:].any()
    # CPU remains rejected even with opt-in; no backend/numerical mock.
    holder.config.gdn_allow_right_padding = True
    configure_gdn_numerics(holder)
    with pytest.raises(UnsupportedCombinationError, match="NPU"):
        holder.gdn(x, attention_mask=keep)
