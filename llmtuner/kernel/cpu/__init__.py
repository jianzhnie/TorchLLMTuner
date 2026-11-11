"""Register CPU reference kernels."""

from .basic import (
    fusion_attention_cpu,
    moe_token_permute_cpu,
    rms_norm_cpu,
    rotary_mul_cpu,
)

__all__ = [
    "fusion_attention_cpu",
    "moe_token_permute_cpu",
    "rms_norm_cpu",
    "rotary_mul_cpu",
]
