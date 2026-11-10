"""Explicit DeepSeek V4 model and context-parallel helpers."""

from .context_parallel import build_raw_attention_mask, deepseek_v4_cp_attention_forward
from .modeling import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4SparseMoeBlock,
)

__all__ = [
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
    "build_raw_attention_mask",
    "deepseek_v4_cp_attention_forward",
]
