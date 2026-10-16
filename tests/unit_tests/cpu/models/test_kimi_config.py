"""Persistent Kimi architecture fields and caller ownership contracts."""

import pytest

from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig


def test_kimi_config_roundtrip_preserves_hybrid_architecture(tmp_path):
    config = KimiLinearConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "head_dim": 8,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
        rope_scaling={"rope_type": "default", "rope_theta": 20000.0},
        skip_kda_recompute=True,
    )
    config.save_pretrained(tmp_path)
    restored = KimiLinearConfig.from_pretrained(tmp_path)
    # HF attaches the load path; architecture and unknown kernel fields persist.
    expected = config.to_dict()
    actual = restored.to_dict()
    expected.pop("_name_or_path", None)
    actual.pop("_name_or_path", None)
    assert actual == expected
    assert restored.is_mla and restored.is_moe and restored.is_linear_attn
    assert [restored.is_kda_layer(i) for i in range(3)] == [True, False, True]
    assert restored.skip_kda_recompute is True


def test_kimi_config_owns_nested_input_dictionaries():
    rope = {"rope_type": "default"}
    attention = {"kda_layers": [1], "full_attn_layers": []}
    config = KimiLinearConfig(rope_parameters=rope, linear_attn_config=attention)
    assert rope == {"rope_type": "default"}
    attention["kda_layers"].append(2)
    assert config.linear_attn_config["kda_layers"] == [1]
    config.rope_parameters["rope_theta"] = 123
    assert "rope_theta" not in rope


@pytest.mark.parametrize("field", ["kda_layers", "full_attn_layers"])
def test_kimi_config_requires_explicit_hybrid_layer_lists(field):
    attention = {"kda_layers": [], "full_attn_layers": []}
    del attention[field]
    with pytest.raises(ValueError, match=field):
        KimiLinearConfig(linear_attn_config=attention)
