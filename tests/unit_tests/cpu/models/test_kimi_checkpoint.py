"""Full Kimi HF state round-trip through native grouped expert execution."""

import pytest
import torch

from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.models.kimi.checkpoint import convert_kimi_expert_state
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks


@pytest.mark.parametrize("latent", [False, True])
def test_kimi_grouped_export_loads_strictly_in_original_hf_model(latent, tmp_path):
    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        num_shared_experts=1,
        num_expert_group=2,
        topk_group=1,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    source = HFTransformerModel(config).eval()
    original = {key: value.clone() for key, value in source.model.state_dict().items()}
    swap_hf_moe_blocks(source)
    adapter = HFTransformerStateDictAdapter(config, None, kimi_expert_parallel=True)
    exported = adapter.to_hf(source.state_dict())
    assert exported.keys() == original.keys()
    for key in original:
        torch.testing.assert_close(exported[key], original[key], atol=0, rtol=0)
    from safetensors.torch import load_file, save_file

    path = tmp_path / "model.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in exported.items()}, path
    )
    exported = load_file(path)
    restored = HFTransformerModel(config).eval()
    restored.model.load_state_dict(exported, strict=True)
    grouped = HFTransformerModel(config).eval()
    swap_hf_moe_blocks(grouped)
    grouped.load_state_dict(adapter.from_hf(exported), strict=True)
    ids = torch.tensor([1, 2, 3])
    torch.testing.assert_close(source(ids), restored(ids), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(source(ids), grouped(ids), atol=0, rtol=0)
    meta = {key: value.to("meta") for key, value in exported.items()}
    assert all(value.is_meta for value in adapter.from_hf(meta).values())


def test_kimi_export_rejects_local_shards():
    key = "model.layers.0.block_sparse_moe.routed_experts.inner_experts.w1_EFD"
    with pytest.raises(ValueError, match="gather EP shards"):
        convert_kimi_expert_state(
            {key: torch.randn(2, 4, 8)},
            num_experts=4,
            to_hf=True,
        )


def test_kimi_import_rejects_missing_experts():
    with pytest.raises(ValueError, match="missing individual experts"):
        convert_kimi_expert_state(
            {"model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(4, 8)},
            num_experts=4,
            to_hf=False,
        )


def test_kimi_mapping_rejects_key_collisions():
    with pytest.raises(ValueError, match="collision"):
        convert_kimi_expert_state(
            {
                "model.layers.0.block_sparse_moe.router.gate.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.gate.weight": torch.randn(4, 8),
            },
            num_experts=4,
            to_hf=True,
        )
