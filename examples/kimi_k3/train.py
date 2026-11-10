"""Train the local Kimi model from an official K3 text configuration on Ascend."""

from __future__ import annotations

import argparse

import torch
from transformers import PretrainedConfig

from llmtuner import Trainer
from llmtuner.config import (
    CheckpointConfig,
    CompileConfig,
    DataloaderConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.errors import ConfigError
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig


def load_text_overrides(model: str, layers: int | None = None) -> dict:
    raw, _ = PretrainedConfig.get_config_dict(model, local_files_only=True)
    if raw.get("model_type") != "kimi_k3" or not isinstance(
        raw.get("text_config"), dict
    ):
        raise ConfigError("Expected an official Kimi K3 config with text_config")
    text = KimiLinearConfig(**raw["text_config"])
    overrides = text.to_dict()
    overrides.pop("model_type", None)
    if layers is not None:
        if not 1 <= layers <= text.num_hidden_layers:
            raise ConfigError("--layers must be within the published layer count")
        overrides["num_hidden_layers"] = layers
        attention = dict(overrides["linear_attn_config"])
        for key in ("kda_layers", "full_attn_layers"):
            attention[key] = [index for index in attention[key] if index <= layers]
        overrides["linear_attn_config"] = attention
    return overrides


def build_config(args: argparse.Namespace) -> LLMTunerConfig:
    if args.compile_backend not in {"none", "aot_eager", "inductor"}:
        raise ConfigError("--compile-backend must be none, aot_eager or inductor")
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="kimi_linear",
            arch_overrides=load_text_overrides(args.model, args.layers),
            initialization_device="meta",
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=args.dp_shard,
            data_parallel_replicate_size=args.dp_replicate,
            tensor_parallel_size=args.tp,
            expert_parallel_size=args.ep,
            pipeline_parallel_size=args.pp,
            context_parallel_kv_allgather_size=args.cp,
        ),
        optimizer=OptimizerConfig(
            implementation="for-loop",
            state_residency=args.optimizer_state_residency,
        ),
        training=TrainingConfig(
            steps=args.steps,
            global_batch_size=args.batch,
            max_seq_len=args.sequence_length,
            seed=42,
            deterministic=True,
            activation_checkpoint_mode=args.ac,
            compile=args.compile_backend != "none",
            compile_config=CompileConfig(
                backend=(
                    "inductor"
                    if args.compile_backend == "none"
                    else args.compile_backend
                ),
                per_block=args.compile_backend != "none",
            ),
            dump_folder=args.output,
            checkpoint_config=CheckpointConfig(
                enable=True,
                interval=args.checkpoint_interval,
                load_step=args.load_step,
                last_save_model_only=False,
                sharded_expert_parallel=args.ep > 1,
                sharded_tensor_parallel=args.tp > 1,
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=args.data,
                tokenizer_path=args.model,
                tokenizer_backend="kimi_k3",
                prompt_field="question",
                response_field="solution",
                chat_template_kwargs={"thinking": False},
                max_num_documents=1,
            ),
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
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
    parser.add_argument(
        "--compile-backend", choices=("none", "aot_eager", "inductor"), default="none"
    )
    parser.add_argument(
        "--optimizer-state-residency",
        choices=("device", "npu_virtual"),
        default="npu_virtual",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--load-step", type=int, default=-1)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This example requires torch-npu and visible Ascend devices")
    torch.set_default_dtype(torch.bfloat16)
    Trainer(build_config(args)).train()


if __name__ == "__main__":
    main()
