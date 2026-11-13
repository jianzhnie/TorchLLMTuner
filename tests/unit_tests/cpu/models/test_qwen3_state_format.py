"""Qwen3 format metadata is model-owned and independent of tensor storage."""

import pytest

from llmtuner.models.qwen.qwen3.state import expert_export_keys, normalize_expert_keys


@pytest.mark.parametrize("native", [False, True])
def test_expert_key_normalization_preserves_shard_grouping_without_mutation(native):
    prefix = "model.layers.12.mlp."
    source = {
        prefix + "experts.0.gate_proj.weight": 9,
        prefix + "experts.0.up_proj.weight": 2,
        prefix + "experts.1.gate_proj.weight": 7,
        prefix + "experts.1.up_proj.weight": 5,
        prefix + "experts.0.down_proj.weight": 4,
        "model.norm.weight": 3,
    }
    before = source.copy()
    keys, indices = normalize_expert_keys(source, source, native_ep_layout=native)
    if native:
        expected = {
            prefix + "routed_experts.inner_experts.w1_EFD": 7,
            prefix + "routed_experts.inner_experts.w3_EFD": 2,
            prefix + "routed_experts.inner_experts.w2_EDF": 4,
            "model.norm.weight": 3,
        }
    else:
        expected = {
            prefix + "experts.gate_up_proj": 2,
            prefix + "experts.down_proj": 4,
            "model.norm.weight": 3,
        }
    assert keys == set(expected)
    assert indices == expected
    assert source == before
    assert list(source) == list(before)
    without_indices, mapping = normalize_expert_keys(
        iter(source), None, native_ep_layout=native
    )
    assert without_indices == keys
    assert mapping is None


@pytest.mark.parametrize(
    ("role", "projection"),
    [("w1_EFD", "gate_proj"), ("w3_EFD", "up_proj"), ("w2_EDF", "down_proj")],
)
def test_export_keys_preserve_projection_roles_and_numeric_expert_order(
    role, projection
):
    prefix = "model.layers.7.mlp."
    keys = expert_export_keys(
        prefix + "routed_experts.inner_experts." + role, (3, 4, 8), num_experts=3
    )
    assert keys == (
        prefix + f"experts.0.{projection}.weight",
        prefix + f"experts.1.{projection}.weight",
        prefix + f"experts.2.{projection}.weight",
    )


@pytest.mark.parametrize("shape", [None, (2, 8), (1, 4, 8), (2, 1, 4, 8)])
def test_export_rejects_incomplete_or_invalid_full_expert_stacks(shape):
    with pytest.raises(ValueError, match="requires full expert stack"):
        expert_export_keys(
            "model.layers.0.mlp.routed_experts.inner_experts.w1_EFD",
            shape,
            num_experts=2,
        )


def test_nonexpert_state_passes_through_and_unknown_native_roles_fail():
    assert expert_export_keys("model.norm.weight", None, num_experts=2) is None
    with pytest.raises(ValueError, match="Unknown Qwen3 expert projection"):
        expert_export_keys(
            "model.layers.0.mlp.routed_experts.inner_experts.w1_EDF",
            (2, 4, 8),
            num_experts=2,
        )
