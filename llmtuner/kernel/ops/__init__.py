"""Semantic kernel APIs; implementations live in the device backends."""

from .attention import fusion_attention
from .moe import moe_token_permute
from .rms_norm import rms_norm
from .rotary import rotary_mul

__all__ = ["fusion_attention", "moe_token_permute", "rms_norm", "rotary_mul"]
