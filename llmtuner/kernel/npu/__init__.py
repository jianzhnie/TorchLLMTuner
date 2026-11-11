"""Register Ascend kernels when torch-npu is available."""

from .basic import (
    fusion_attention_npu,
    moe_token_permute_npu,
    rms_norm_npu,
    rotary_mul_npu,
)

__all__ = [
    "fusion_attention_npu",
    "moe_token_permute_npu",
    "rms_norm_npu",
    "rotary_mul_npu",
]
