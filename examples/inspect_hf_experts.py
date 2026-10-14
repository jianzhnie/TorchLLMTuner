"""Inspect fused expert layouts after a transformers upgrade.

Run from the repository root:
    python examples/inspect_hf_experts.py
    python examples/inspect_hf_experts.py qwen3_moe mixtral

This diagnostic constructs tiny models and prints raw expert shapes and
activation classes. It does not establish routing or numerical equivalence.
"""

from __future__ import annotations

import argparse

import transformers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "families", nargs="*", default=["qwen3_moe", "olmoe", "mixtral", "gpt_oss"]
    )
    args = parser.parse_args()
    failed = False
    for family in args.families:
        try:
            config = transformers.AutoConfig.for_model(
                family,
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                num_experts=8,
                num_experts_per_tok=2,
                num_local_experts=8,
                n_routed_experts=8,
                n_shared_experts=1,
                decoder_sparse_step=1,
                first_k_dense_replace=1,
                n_group=2,
                topk_group=1,
                max_position_embeddings=128,
                vocab_size=128,
            )
            model = transformers.AutoModelForCausalLM.from_config(
                config, experts_implementation="eager"
            )
            experts = model.model.layers[-1].mlp.experts
            gate_up = getattr(experts, "gate_up_proj", None)
            down = getattr(experts, "down_proj", None)
            activation = getattr(experts, "act_fn", None)
            if activation is None:
                activation = getattr(experts, "_act_fn", None)
            print(
                f"{family}: gate_up_proj="
                f"{None if gate_up is None else tuple(gate_up.shape)}, "
                f"down_proj={None if down is None else tuple(down.shape)}, "
                f"activation={type(activation).__name__}"
            )
            del model, experts
        except Exception as error:  # noqa: BLE001 - report each family independently
            print(f"{family}: unavailable ({error})")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
