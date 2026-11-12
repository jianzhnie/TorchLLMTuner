"""Explicit Qwen3 rotary precision without attention monkey patches."""

import torch
from torch import Tensor
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRotaryEmbedding,
    rotate_half,
)


def validate_rotary_precision(config: Qwen3MoeConfig) -> None:
    """Validate supported initialization and precision before allocation."""
    if getattr(config, "rotary_init_device", "constructor") != "constructor":
        raise ValueError("Qwen3-MoE initialization requires constructor RoPE")
    compute = getattr(config, "rotary_compute_dtype", None)
    cache = getattr(config, "rotary_cache_dtype", None)
    if compute not in (None, "float32"):
        raise ValueError("Qwen3 rotary_compute_dtype must be float32 or None")
    if cache not in (None, "float32", "bfloat16"):
        raise ValueError("Qwen3 rotary_cache_dtype must be float32/bfloat16 or None")
    if cache is not None and compute != "float32":
        raise ValueError("Qwen3 rotary cache precision requires FP32 application")


class Qwen3FP32RotaryEmbedding(Qwen3MoeRotaryEmbedding):
    """Keep reference frequency/cache lifecycle, retain FP32 trigonometry."""

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        cosine, sine = super().forward(x.float(), position_ids)
        if getattr(self.config, "rotary_cache_dtype", None) == "bfloat16":
            cosine, sine = cosine.bfloat16(), sine.bfloat16()
        return cosine, sine


def apply_fp32_rotary(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    """Share each FP32 cast across both branches, casting only the final sum."""
    with torch.autocast(device_type=query.device.type, enabled=False):
        cosine, sine = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
        query_fp32, key_fp32 = query.float(), key.float()
        return (
            (query_fp32 * cosine + rotate_half(query_fp32) * sine).to(query.dtype),
            (key_fp32 * cosine + rotate_half(key_fp32) * sine).to(key.dtype),
        )
