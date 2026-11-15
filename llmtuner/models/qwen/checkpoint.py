"""Qwen3.5 model-owned semantic configuration for checkpoint compatibility."""

from typing import Any


def qwen35_text_configuration(config: Any) -> dict[str, object]:
    """Shared dense/MoE attention and recurrent semantics; absent fields stay null."""
    return {
        name: getattr(config, name, None)
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "attn_mask_type",
            "is_causal",
            "rms_norm_eps",
            "max_position_embeddings",
            "rope_theta",
            "rope_scaling",
            "rope_parameters",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "norm_topk_prob",
            "router_aux_loss_coef",
            "tie_word_embeddings",
            "pad_token_id",
        )
    }


def qwen35_multimodal_configuration(config: Any) -> dict[str, object]:
    """Compose text, vision geometry and media token insertion semantics."""
    return {
        "text": qwen35_text_configuration(config.text_config),
        "vision": {
            name: getattr(config.vision_config, name, None)
            for name in (
                "depth",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "hidden_act",
                "in_channels",
                "patch_size",
                "temporal_patch_size",
                "spatial_merge_size",
                "out_hidden_size",
                "num_position_embeddings",
                "deepstack_visual_indexes",
            )
        },
        "media_tokens": {
            name: getattr(config, name, None)
            for name in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
        },
    }
