"""Real compressor CP Trainer checkpoint full/resume on CPU/Gloo ranks.

Run each phase with torchrun --nproc_per_node=2, arguments: full|resume workdir.
Reuses strict pre-update state and continued-training checks from the PP harness.
Set LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_EMA=dynamic in both phases.
CP2/DP1 by default; set LLMTUNER_TEST_HCA_DP_SIZE=2 and world size 4
for FSDP2. Set LLMTUNER_TEST_CSA=1 to append CSA and
LLMTUNER_TEST_SLIDING=1 to prepend sliding attention. All-HCA by default;
TP/PP/EP1 and AC/compile disabled.
"""

import os
from unittest.mock import patch

from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    FSDPConfig,
    ParallelConfig,
)
from tests.integration_tests import pp_checkpoint_equivalence as harness

base_config = harness._cfg


def config(workdir, *, packed=False):
    if packed:
        raise ValueError("HCA Trainer scenario requires unpacked input")
    cfg = base_config(workdir, packed=False)
    mixed = os.environ.get("LLMTUNER_TEST_CSA") == "1"
    layers = ["heavily_compressed_attention"]
    if mixed:
        layers.append("compressed_sparse_attention")
    if os.environ.get("LLMTUNER_TEST_SLIDING") == "1":
        layers.insert(0, "sliding_attention")
    cfg.model = ModelConfig(
        model_name_or_path="deepseek_v4",
        vocab_size=harness.VOCAB,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=len(layers),
        num_attention_heads=4,
        num_key_value_heads=1,
        arch_overrides={
            "head_dim": 8,
            "partial_rotary_factor": 0.5,
            "q_lora_rank": 8,
            "o_groups": 2,
            "o_lora_rank": 4,
            "sliding_window": 3,
            "layer_types": layers,
            "mlp_layer_types": ["moe"] * len(layers),
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 1,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "hc_mult": 2,
            "compress_rates": {
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 4,
            },
        },
    )
    cfg.parallel = ParallelConfig(
        context_parallel_ulysses_size=2,
        data_parallel_shard_size=int(os.environ.get("LLMTUNER_TEST_HCA_DP_SIZE", "1")),
        fsdp_config=FSDPConfig(
            fsdp_implementation=os.environ.get(
                "LLMTUNER_TEST_FSDP_IMPLEMENTATION", "native"
            )
        ),
        cp_config=ContextParallelConfig(context_parallel_strategy="ulysses"),
    )
    return cfg


if __name__ == "__main__":
    with patch.object(harness, "_cfg", config):
        harness.main()
