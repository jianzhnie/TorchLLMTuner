"""Configuration for the minimal Qwen3-MoE training experiment."""

import math
from dataclasses import dataclass
from pathlib import Path

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.config import ParallelConfig


@dataclass(kw_only=True, slots=True)
class ExperimentConfig:
    parallel: ParallelConfig
    steps: int = 2
    batch_size: int = 1
    seq_len: int = 16
    learning_rate: float = 1e-3
    seed: int = 42
    checkpoint: Path | None = None
    resume: Path | None = None
    model_config: Path | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.seq_len % self.parallel.tp_size:
            raise ValueError("seq_len must be divisible by TP degree")
        if self.steps < 1 or self.batch_size < 1 or self.seq_len < 2:
            raise ValueError(
                "steps and batch_size must be positive; seq_len must be >= 2"
            )
        if self.parallel.cp_size != 1 or self.parallel.pp_size != 1:
            raise ValueError(
                "This experiment implements TP/EP/FSDP only; CP and PP are rejected"
            )


def small_qwen3_moe_config() -> Qwen3MoeConfig:
    """Return a small randomly initialized model for fast distributed checks."""
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        router_aux_loss_coef=0.0,
        attention_dropout=0.0,
        experts_implementation="eager",
    )


def load_model_config(path: Path | None) -> Qwen3MoeConfig:
    """Read architecture only; never load weights or select a model by inspection."""
    if path is None:
        return small_qwen3_moe_config()
    path = path / "config.json" if path.is_dir() else path
    config = Qwen3MoeConfig.from_json_file(str(path))
    if config.model_type != "qwen3_moe":
        raise ValueError("This experiment requires a qwen3_moe configuration")
    config._experts_implementation = "eager"
    config.router_aux_loss_coef = 0.0
    config.attention_dropout = 0.0
    return config
