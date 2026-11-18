"""Offline auxiliary Trainer recipe; never a target-model acceptance result."""

from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)


def qwen3_config() -> LLMTunerConfig:
    """Use registered Qwen3-MoE dense layers for auxiliary service contracts."""
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            arch_overrides={"head_dim": 16, "mlp_only_layers": [0, 1]},
        ),
        parallel=ParallelConfig(data_parallel_shard_size=-1),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=20,
            seed=42,
            deterministic=True,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )
