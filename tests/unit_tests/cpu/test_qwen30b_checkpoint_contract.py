"""CPU/meta regression for actual HF-to-native joint-checkpoint FQNs.

These small models validate the generator, not target NPU acceptance.
"""

import torch
from transformers import Qwen3MoeConfig

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from tests.integration_tests.prepare_qwen30b_checkpoint_contract import (
    project_joint_global_schema,
)


def test_joint_projection_uses_actual_expert_conversion_and_dense_tp_plan():
    cfg = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
    )
    with torch.device("meta"):
        model = HFTransformerModel(cfg)
        count, ownership = project_joint_global_schema(
            model,
            ParallelConfig(
                tensor_parallel_size=2,
                expert_parallel_size=4,
                data_parallel_shard_size=2,
                context_parallel_kv_allgather_size=2,
            ),
        )
    assert count == 2
    state = model.state_dict()
    for layer in range(2):
        prefix = f"model.model.layers.{layer}.mlp"
        assert f"{prefix}.experts.gate_up_proj" not in state
        assert f"{prefix}.gate.weight" not in state
        assert state[f"{prefix}.router.gate.weight"].shape == (8, 16)
        for name, shape in (
            ("w1_EFD", (8, 8, 16)),
            ("w3_EFD", (8, 8, 16)),
            ("w2_EDF", (8, 16, 8)),
        ):
            key = f"{prefix}.routed_experts.inner_experts.{name}"
            assert state[key].shape == shape
            assert ownership[key] == {"axis": 0, "degree": 4}
        q = f"model.model.layers.{layer}.self_attn.q_proj.weight"
        o = f"model.model.layers.{layer}.self_attn.o_proj.weight"
        assert ownership[q] == {"axis": 0, "degree": 2}
        assert ownership[o] == {"axis": 1, "degree": 2}
    assert all(t.device.type == "meta" for t in state.values())
