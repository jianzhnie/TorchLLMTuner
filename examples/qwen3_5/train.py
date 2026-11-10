"""Train random-initialized HF Qwen3.5 text models on Ascend NPUs."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig

from llmtuner import Trainer
from llmtuner.config import (
    CheckpointConfig,
    DataloaderConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.errors import ConfigError


def build_config(args: argparse.Namespace) -> LLMTunerConfig:
    initial_weights = getattr(args, "initial_weights", None)
    if initial_weights and (args.load_step >= 0 or args.layers is not None):
        raise ConfigError(
            "--initial-weights is a full-model step-zero snapshot; do not "
            "combine it with --load-step or --layers"
        )
    compile_backend = getattr(args, "compile_backend", "none")
    if compile_backend not in {"none", "aot_eager", "inductor"}:
        raise ConfigError("--compile-backend must be none, aot_eager or inductor")
    architecture = AutoConfig.from_pretrained(args.model)
    text = getattr(architecture, "text_config", architecture)
    if text.model_type not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise ConfigError(f"Expected Qwen3.5 text architecture, got {text.model_type}")
    # Preserve every published dimension. Layer reduction is an explicit OOM
    # fallback, never the default, and must also truncate the HF layer schedule.
    overrides = text.to_dict()
    overrides.pop("model_type", None)
    if args.layers is not None:
        if not 1 <= args.layers <= text.num_hidden_layers:
            raise ConfigError("--layers must be within the published layer count")
        overrides["num_hidden_layers"] = args.layers
        if "layer_types" in overrides:
            overrides["layer_types"] = overrides["layer_types"][: args.layers]
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path=text.model_type,
            arch_overrides=overrides,
            initialization_device="meta",
            linear_attention_backend=getattr(
                args, "linear_attention_backend", "native"
            ),
            router_compute_dtype=getattr(args, "router_compute_dtype", None),
            experts_implementation=getattr(args, "experts_implementation", "native"),
            gdn_numerical_policy=getattr(args, "gdn_numerical_policy", "native"),
            gdn_allow_right_padding=getattr(args, "gdn_allow_right_padding", False),
            gdn_deterministic_reduction=getattr(
                args, "gdn_deterministic_reduction", False
            ),
            rotary_compute_dtype=getattr(args, "rotary_compute_dtype", None),
            rotary_cache_dtype=getattr(args, "rotary_cache_dtype", None),
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=getattr(args, "dp_shard", -1),
            data_parallel_replicate_size=getattr(args, "dp_replicate", 1),
            tensor_parallel_size=getattr(args, "tp", 1),
            expert_parallel_size=getattr(args, "ep", 1),
            pipeline_parallel_size=getattr(args, "pp", 1),
            context_parallel_kv_allgather_size=getattr(args, "cp", 1),
        ),
        training=TrainingConfig(
            steps=args.steps,
            global_batch_size=args.batch,
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
            max_seq_len=args.sequence_length,
            seed=42,
            deterministic=True,
            activation_checkpoint_mode=getattr(args, "ac", "full"),
            compile=compile_backend != "none",
            compile_config=CompileConfig(
                backend="inductor" if compile_backend == "none" else compile_backend,
                per_block=compile_backend != "none",
            ),
            dump_folder=args.output,
            checkpoint_config=CheckpointConfig(
                enable=True,
                interval=args.checkpoint_interval,
                load_step=args.load_step,
                last_save_model_only=False,
                initial_load_path=initial_weights,
                initial_load_in_hf=bool(initial_weights),
                sharded_tensor_parallel=getattr(args, "checkpoint_layout", "full")
                == "chunk",
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=args.data,
                tokenizer_path=args.model,
                prompt_field="question",
                response_field="solution",
                chat_template_kwargs={"enable_thinking": False},
                max_num_documents=1,
            ),
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Share model, numerical policy and topology options with resume checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dp-shard", type=int, default=-1)
    parser.add_argument("--dp-replicate", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1, help="KV-allgather CP degree")
    parser.add_argument(
        "--ac",
        choices=("none", "full", "selective", "memory_budget", "region"),
        default="full",
    )
    parser.add_argument("--router-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--experts-implementation", choices=("native", "pre_down"), default="native"
    )
    parser.add_argument("--rotary-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--gdn-numerical-policy",
        choices=("native", "split_qkv_fp32_norm"),
        default="native",
    )
    parser.add_argument(
        "--gdn-allow-right-padding",
        action="store_true",
        help="Opt in to B1 right padding with npu_triton + split_qkv_fp32_norm",
    )
    parser.add_argument(
        "--gdn-deterministic-reduction",
        action="store_true",
        help="Use block reduction for native NPU GDN dk/dbeta gradients",
    )
    parser.add_argument("--rotary-cache-dtype", choices=("float32",))
    parser.add_argument(
        "--linear-attention-backend", choices=("native", "npu_triton"), default="native"
    )
    parser.add_argument(
        "--compile-backend", choices=("none", "aot_eager", "inductor"), default="none"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument(
        "--checkpoint-layout", choices=("full", "chunk"), default="full"
    )
    parser.add_argument("--load-step", type=int, default=-1)
    parser.add_argument(
        "--initial-weights",
        help="Absolute path to shared random step-zero HF weights, not a resume",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This example requires torch-npu and visible Ascend devices")
    torch.set_default_dtype(torch.bfloat16)
    Trainer(build_config(args)).train()


if __name__ == "__main__":
    main()
