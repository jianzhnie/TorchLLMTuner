"""Model architecture config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.errors import ConfigError

LM_HEAD_COMPUTE_DTYPES = ("float32", "float16", "bfloat16")


@dataclass
class ModelConfig:
    """Model architecture, with CLI sizes filled from saved HF configurations."""

    model_name_or_path: str = field(
        default="llama",
        metadata={
            "help": "HF architecture name, local model directory, or hub id 'org/name'"
        },
    )
    vocab_size: int = field(default=128, metadata={"help": "Vocabulary size"})
    hidden_size: int = field(default=64, metadata={"help": "Hidden dimension"})
    intermediate_size: int = field(
        default=128, metadata={"help": "FFN inner dimension"}
    )
    num_hidden_layers: int = field(
        default=2, metadata={"help": "Number of decoder layers"}
    )
    num_attention_heads: int = field(
        default=4, metadata={"help": "Number of attention heads"}
    )
    num_key_value_heads: int = field(
        default=4, metadata={"help": "Number of KV heads (GQA)"}
    )
    experts_implementation: str = field(
        default="native",
        metadata={
            "help": "HF MoE expert kernel: native, grouped_mm, batched_mm, or eager. "
            "Overrides require model support; ignored under EP > 1."
        },
    )
    norm_backend: str = field(
        default="native",
        metadata={
            "help": "Qwen3 RMSNorm: native, aten (CPU first-order boundaries), "
            "or joint (dense Qwen3 first-order norm/residual boundary)."
        },
    )
    qkv_projection: str = field(
        default="separate",
        metadata={
            "help": "Qwen3 QKV projection: separate or fused. Fused uses one GEMM "
            "with the existing parameters; tensor parallelism is not yet supported."
        },
    )
    rotary_init_device: str = field(
        default="constructor",
        metadata={
            "help": "RoPE initialization: constructor (HF default) or target "
            "(rebuild dense Qwen3 default frequencies on the training device)."
        },
    )
    attention_backend: str = field(
        default="auto",
        metadata={"help": "Attention backend: auto or eager_flex (CPU/NPU math Flex)."},
    )
    compute_dtype: str | None = field(
        default=None,
        metadata={
            "help": "lm_head forward dtype: float32, float16, bfloat16, or None for "
            "model dtype. Stored weights retain their dtype."
        },
    )
    arch_overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Programmatic architecture overrides for locally built models; "
            "ignored for saved HF configurations."
        },
    )

    def update_architecture(self, architecture: object) -> None:
        """Copy available decoder geometry from a resolved architecture."""
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if hasattr(architecture, name):
                setattr(self, name, getattr(architecture, name))

    def __post_init__(self) -> None:
        if self.attention_backend not in ("auto", "eager_flex"):
            raise ConfigError("model.attention_backend must be 'auto' or 'eager_flex'.")
        if self.rotary_init_device not in ("constructor", "target"):
            raise ConfigError(
                "model.rotary_init_device must be 'constructor' or 'target'."
            )
        if self.qkv_projection not in ("separate", "fused"):
            raise ConfigError("model.qkv_projection must be 'separate' or 'fused'.")
        if self.norm_backend not in ("native", "aten", "joint"):
            raise ConfigError(
                "model.norm_backend must be 'native', 'aten', or 'joint'."
            )
        # Architecture numbers must be sane before any model build: an invalid
        # value here used to surface deep inside HF's for_model or the
        # wrapper's forward, long after the config was accepted.
        require_integer_at_least(
            self,
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            group="model",
        )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError(
                "model.hidden_size must be divisible by num_attention_heads, "
                f"got {self.hidden_size} and {self.num_attention_heads}"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ConfigError(
                "model.num_key_value_heads must be <= num_attention_heads "
                f"({self.num_attention_heads}), got {self.num_key_value_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ConfigError(
                "model.num_attention_heads must be divisible by num_key_value_heads, "
                f"got {self.num_attention_heads} and {self.num_key_value_heads}"
            )
        if (
            self.compute_dtype is not None
            and self.compute_dtype not in LM_HEAD_COMPUTE_DTYPES
        ):
            raise ConfigError(
                f"model.compute_dtype must be one of float32/float16/bfloat16, "
                f"got {self.compute_dtype!r}"
            )
        if self.experts_implementation not in (
            "native",
            "grouped_mm",
            "batched_mm",
            "eager",
        ):
            raise ConfigError(
                f"model.experts_implementation must be one of "
                f"native/grouped_mm/batched_mm/eager, got "
                f"{self.experts_implementation!r}"
            )
