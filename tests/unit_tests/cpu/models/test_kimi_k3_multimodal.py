"""Explicit Kimi K3 multimodal assembly over the native text decoder."""

import torch

from llmtuner.models.kimi.kimi_k3.configuration import KimiK3Config
from llmtuner.models.kimi.kimi_k3.multimodal import KimiK3ForConditionalGeneration


def small_config() -> KimiK3Config:
    return KimiK3Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "linear_attn_config": {
                "kda_layers": [],
                "full_attn_layers": [1],
                "head_dim": 16,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        },
        vision_config={
            "patch_size": 2,
            "init_pos_emb_height": 2,
            "init_pos_emb_width": 2,
            "init_pos_emb_time": 1,
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
            "vt_hidden_size": 16,
            "vt_intermediate_size": 32,
            "qkv_hidden_size": 32,
            "mm_hidden_size": 16,
            "text_hidden_size": 32,
            "_attn_implementation": "eager",
        },
        media_placeholder_token_id=3,
    )


def test_kimi_k3_multimodal_forward_and_backward() -> None:
    model = KimiK3ForConditionalGeneration(small_config())
    input_ids = torch.tensor([[1, 3, 2]])
    labels = input_ids.clone()

    output = model(
        input_ids=input_ids,
        pixel_values=torch.randn(1, 3, 2, 2),
        grid_thws=torch.tensor([[1, 2, 2]]),
        labels=labels,
    )

    assert output.logits.shape == (1, 3, 32)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.mm_projector is not None
    assert any(
        parameter.grad is not None for parameter in model.vision_tower.parameters()
    )


def test_kimi_k3_config_roundtrip_restores_nested_types(tmp_path) -> None:
    from llmtuner.models.kimi.kimi_k3.configuration import (
        KimiK3VisionConfig,
        KimiLinearConfig,
    )

    small_config().save_pretrained(tmp_path)
    restored = KimiK3Config.from_pretrained(tmp_path)

    assert isinstance(restored.text_config, KimiLinearConfig)
    assert isinstance(restored.vision_config, KimiK3VisionConfig)
    assert KimiK3ForConditionalGeneration(restored) is not None
