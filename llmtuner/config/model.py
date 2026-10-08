"""Model architecture config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmtuner.errors import ConfigError


@dataclass
class ModelConfig:
    """Model architecture. None fields are auto-filled from the HF config.

    Offline path: when ``model_name_or_path`` has no '/', build a tiny local model
    with AutoConfig.for_model using the explicit sizes below. Online path: a hub id
    ("org/name") pulls real architecture via AutoConfig.from_pretrained.
    """

    model_name_or_path: str = field(
        default="llama",
        metadata={
            "help": "HF architecture name (offline) or hub id 'org/name' (online)"
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
            "help": "HF experts forward kernel for MoE models: 'native' keeps "
            "the model's built-in kernel; 'grouped_mm' / 'batched_mm' / 'eager' "
            "require a model with a settable experts implementation and raise "
            "otherwise (never silently substituted). Irrelevant under EP>1, "
            "where the swap replaces the whole MoE block."
        },
    )
    compute_dtype: str | None = field(
        default=None,
        metadata={
            "help": "Forward matmul dtype for the lm_head only: 'float32', "
            "'float16', or 'bfloat16'. None (default) leaves the output "
            "projection in the model's own dtype. The stored weight keeps its "
            "dtype -- the cast happens inside the forward, so logits are "
            "scored at the requested precision without an upcast copy of the "
            "weight living in the optimizer."
        },
    )
    arch_overrides: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Extra architecture settings, by name, for fields this "
            "config does not name. The six fields above cover every dense "
            "decoder; anything else a model needs -- an MoE's "
            "n_routed_experts / num_experts_per_tok, MLA's q_lora_rank -- has "
            "no field here. Offline only: they are applied on top of the six "
            "above when building the HF config, and ignored for a hub id or a "
            "local checkpoint directory, where the saved config wins. Supplied "
            "programmatically (a dict does not survive the CLI parser)."
        },
    )

    def __post_init__(self) -> None:
        # Architecture numbers must be sane before any model build: an invalid
        # value here used to surface deep inside HF's for_model or the
        # wrapper's forward, long after the config was accepted.
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(
                    f"model.{name} must be >= 1, got {getattr(self, name)}"
                )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError(
                f"model.hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ConfigError(
                f"model.num_key_value_heads ({self.num_key_value_heads}) cannot "
                f"exceed num_attention_heads ({self.num_attention_heads})"
            )
        if self.compute_dtype is not None and self.compute_dtype not in (
            "float32",
            "float16",
            "bfloat16",
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
