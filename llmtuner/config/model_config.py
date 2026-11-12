"""Model architecture config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmtuner.config.utils import require_integer_at_least
from llmtuner.errors import ConfigError

LM_HEAD_COMPUTE_DTYPES = ("float32", "float16", "bfloat16")


@dataclass
class ModelConfig:
    """Model architecture, with CLI sizes filled from saved HF configurations."""

    model_name_or_path: str = field(
        default="qwen3_moe",
        metadata={
            "help": "Registered architecture, local model directory, or Hub model ID"
        },
    )
    initialization_device: str = field(
        default="cpu",
        metadata={
            "help": "Random initialization: cpu (legacy) or meta (after sharding)"
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
            "help": "HF MoE expert kernel: native, grouped_mm, batched_mm, eager, "
            "torchtitan, or pre_down (Qwen3.5 BF16 score before down projection). "
            "Overrides require model support; EP replacement rejects pre_down "
            "and otherwise uses its own expert backend."
        },
    )
    router_compute_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Router compute: float32 or None for existing behavior. "
            "Qwen3-MoE and Qwen3.5-MoE use FP32 only with explicit opt-in. "
            "Parameter storage dtype and checkpoint keys are unchanged."
        },
    )
    norm_backend: str = field(
        default="native",
        metadata={
            "help": "Explicit Qwen3 RMSNorm: native, torch, or npu "
            "(matching activation/weight dtype). Legacy aten/joint contracts "
            "still require migration and are rejected by the explicit model."
        },
    )
    linear_attention_backend: str = field(
        default="native",
        metadata={
            "help": "Qwen3.5 linear attention recurrence: native or npu_triton "
            "(opt-in BF16, single-document pure FSDP; no compile or cache)."
        },
    )
    gdn_numerical_policy: str = field(
        default="native",
        metadata={
            "help": "GDN arithmetic: native or split_qkv_fp32_norm (explicit "
            "BF16 Qwen3.5 projection/norm policy, independent of recurrence)."
        },
    )
    gdn_deterministic_reduction: bool = field(
        default=False,
        metadata={
            "help": "Explicit deterministic GDN gradient reduction; "
            "requires npu_triton, not full-model determinism acceptance."
        },
    )
    gdn_allow_right_padding: bool = field(
        default=False,
        metadata={
            "help": "Experimental B1 causal right padding; requires "
            "npu_triton + split_qkv_fp32_norm; masked FSDP/Trainer unverified."
        },
    )
    qkv_projection: str = field(
        default="separate",
        metadata={
            "help": "Qwen3/Qwen3-MoE QKV projection: separate or fused. Fused "
            "uses one GEMM; tensor parallelism is not yet supported."
        },
    )
    rotary_init_device: str = field(
        default="constructor",
        metadata={
            "help": "RoPE initialization: constructor (HF default) or target "
            "(rebuild dense Qwen3 default frequencies on the training device)."
        },
    )
    rotary_compute_dtype: str | None = field(
        default=None,
        metadata={
            "help": "RoPE application dtype: float32 or None for the HF default. "
            "FP32 application supports Qwen3/Qwen3-MoE."
        },
    )
    rotary_cache_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Cos/sin cache precision: bfloat16 or Qwen3.5 float32; None "
            "keeps the configured rotary adapter's default."
        },
    )
    attention_backend: str = field(
        default="auto",
        metadata={
            "help": "Attention backend: auto uses the explicit model default. "
            "Legacy flex/eager_flex adapters are not migrated and are rejected."
        },
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
        if not isinstance(self.gdn_deterministic_reduction, bool):
            raise ConfigError("model.gdn_deterministic_reduction must be boolean")
        if (
            self.gdn_deterministic_reduction
            and self.linear_attention_backend != "npu_triton"
        ):
            raise ConfigError("gdn_deterministic_reduction requires npu_triton")
        if not isinstance(self.gdn_allow_right_padding, bool):
            raise ConfigError("model.gdn_allow_right_padding must be boolean")
        if self.gdn_allow_right_padding and (
            self.linear_attention_backend != "npu_triton"
            or self.gdn_numerical_policy != "split_qkv_fp32_norm"
        ):
            raise ConfigError(
                "gdn_allow_right_padding requires npu_triton + split_qkv_fp32_norm"
            )
        if self.gdn_numerical_policy not in ("native", "split_qkv_fp32_norm"):
            raise ConfigError(
                "model.gdn_numerical_policy must be native or split_qkv_fp32_norm"
            )
        if self.router_compute_dtype not in (None, "float32"):
            raise ConfigError("model.router_compute_dtype must be float32 or None")
        if self.linear_attention_backend not in ("native", "npu_triton"):
            raise ConfigError(
                "model.linear_attention_backend must be native or npu_triton"
            )
        if self.initialization_device not in ("cpu", "meta"):
            raise ConfigError("model.initialization_device must be cpu or meta")
        if self.attention_backend not in ("auto", "flex", "eager_flex"):
            raise ConfigError(
                "model.attention_backend must be 'auto', 'flex', or 'eager_flex'."
            )
        if self.rotary_init_device not in ("constructor", "target"):
            raise ConfigError(
                "model.rotary_init_device must be 'constructor' or 'target'."
            )
        if self.rotary_cache_dtype not in (None, "bfloat16", "float32"):
            raise ConfigError(
                "model.rotary_cache_dtype must be bfloat16, float32 or None."
            )
        if self.rotary_cache_dtype and self.rotary_compute_dtype != "float32":
            raise ConfigError("model.rotary_cache_dtype requires FP32 RoPE.")
        if self.rotary_compute_dtype not in (None, "float32"):
            raise ConfigError("model.rotary_compute_dtype must be float32 or None.")
        if self.qkv_projection not in ("separate", "fused"):
            raise ConfigError("model.qkv_projection must be separate or fused.")
        if self.norm_backend not in ("native", "torch", "aten", "joint", "npu"):
            raise ConfigError(
                "model.norm_backend must be native, torch, aten, joint, or npu."
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
            "torchtitan",
            "pre_down",
        ):
            raise ConfigError(
                f"model.experts_implementation must be one of "
                f"native/grouped_mm/batched_mm/eager/torchtitan/pre_down, got "
                f"{self.experts_implementation!r}"
            )
