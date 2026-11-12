"""Explicit dense Llama math with shared RoPE/GQA/SwiGLU execution primitives.

Reference: Transformers 5.11.0 modeling_llama (Apache-2.0).
Llama does not apply Qwen3's per-head Q/K normalization.
"""

from dataclasses import replace

from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .model import Qwen3Model


class LlamaModel(Qwen3Model):
    """Reuse identical dense math; explicitly remove Qwen-specific Q/K norms.

    Supports full-head default RoPE and bias-free SwiGLU. Unsupported math is
    rejected during construction rather than approximated with Qwen semantics.
    No Transformers model class is instantiated by this implementation.
    """

    def __init__(self, config: LlamaConfig) -> None:
        if config.model_type != "llama":
            raise ValueError("LlamaModel requires a llama configuration")
        if config.mlp_bias:
            raise ValueError("Llama prototype does not implement MLP bias")
        if getattr(config, "pretraining_tp", 1) != 1:
            raise ValueError("Llama prototype requires pretraining_tp=1")
        values = config.to_dict()
        values.pop("model_type", None)
        dense = Qwen3Config(**values)
        dense._attn_implementation = config._attn_implementation
        super().__init__(dense)
        for layer in self.layers:
            layer.self_attn.q_norm = nn.Identity()
            layer.self_attn.k_norm = nn.Identity()
        self.contract = replace(
            self.contract,
            architecture="llama",
            tp_gradient_sums=(),
        )
