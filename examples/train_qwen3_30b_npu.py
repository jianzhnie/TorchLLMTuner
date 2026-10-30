"""Train the local pretrained Qwen3-30B-A3B checkpoint on Ascend NPUs.

The architecture is read from the local Hugging Face directory:

    /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B

The safetensors shards are loaded through DCP after FSDP has established
the distributed parameter layout. Key coverage is checked strictly before any
training step, so this cannot silently fall back to random weights.

Launch through ``examples/train_qwen3_30b_npu.sh``. The default topology is
8-way FSDP with full activation checkpointing and BF16 model construction.

``LLMTUNER_TENSOR_PARALLEL_SIZE`` / ``LLMTUNER_PIPELINE_PARALLEL_SIZE`` /
``LLMTUNER_NUM_PP_MICROBATCHES`` / ``LLMTUNER_CONTEXT_PARALLEL_SIZE`` select
other topologies without editing this file; they default to 1 (pure FSDP),
except the microbatch count, which follows ``pipeline_parallel_size`` since the
schedule rejects fewer microbatches than stages. The data-parallel shard degree
keeps its ``-1`` "cover the rest of the world" meaning, so the explicit sizes
multiply up to the world size: ``dp_shard * tp * pp * cp == world``. Context
parallelism shards the sequence across ranks and is served by the sdpa
KV-allgather kernel, which requires ``max_seq_len`` to be a multiple of the
context-parallel size.
"""

from __future__ import annotations

import os

import torch

from llmtuner import Trainer
from llmtuner.config import (
    CheckpointConfig,
    DataloaderConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TrainingConfig,
)
from llmtuner.config.parallel_config import FSDPConfig

MODEL_PATH = "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B"
DATASET_PATH = "/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl"


def qwen3_30b_npu_config() -> LLMTunerConfig:
    model_path = os.environ.get("LLMTUNER_QWEN3_30B_PATH", MODEL_PATH)
    dataset_path = os.environ.get("LLMTUNER_DATASET_PATH", DATASET_PATH)
    save_checkpoint = os.environ.get("LLMTUNER_SAVE_CHECKPOINT", "1") == "1"
    cpu_offload = os.environ.get("LLMTUNER_FSDP_CPU_OFFLOAD", "0") == "1"
    tensor_parallel_size = int(os.environ.get("LLMTUNER_TENSOR_PARALLEL_SIZE", "1"))
    expert_parallel_size = int(os.environ.get("LLMTUNER_EXPERT_PARALLEL_SIZE", "1"))
    pipeline_parallel_size = int(os.environ.get("LLMTUNER_PIPELINE_PARALLEL_SIZE", "1"))
    # One microbatch per stage is the floor the schedule accepts, and the only
    # default that works unasked: pp=2 with a single microbatch is rejected by
    # torch's 1F1B schedule constructor (n_microbatches >= num_stages).
    num_pp_microbatches = int(
        os.environ.get(
            "LLMTUNER_NUM_PP_MICROBATCHES", str(max(1, pipeline_parallel_size))
        )
    )
    # CP splits the sequence across ranks; the kv_allgather strategy is the one
    # the sdpa kernel implements (see parallel/context_parallel/sdpa_attention).
    context_parallel_size = int(os.environ.get("LLMTUNER_CONTEXT_PARALLEL_SIZE", "1"))
    # Deterministic kernels cost throughput; the default matches the shape of
    # this example (a training run, not a bit-reproducibility harness). Turn it
    # on when comparing per-step loss curves against another implementation.
    deterministic = os.environ.get("LLMTUNER_DETERMINISTIC", "0") == "1"
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"Qwen3-30B-A3B config.json not found under {model_path}"
        )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"training dataset not found: {dataset_path}")

    return LLMTunerConfig(
        model=ModelConfig(model_name_or_path=model_path),
        parallel=ParallelConfig(
            data_parallel_replicate_size=int(
                os.environ.get("LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE", "1")
            ),
            data_parallel_shard_size=int(
                os.environ.get("LLMTUNER_DATA_PARALLEL_SHARD_SIZE", "-1")
            ),
            expert_parallel_size=expert_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_kv_allgather_size=context_parallel_size,
            pp_config=PipelineParallelConfig(num_pp_microbatches=num_pp_microbatches),
            fsdp_config=FSDPConfig(
                fsdp_cpu_offload=cpu_offload,
                fsdp_cpu_offload_pin_memory=(
                    os.environ.get("LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY", "1") == "1"
                ),
            ),
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.1,
            implementation="for-loop" if cpu_offload else "fused",
        ),
        training=TrainingConfig(
            global_batch_size=int(os.environ.get("LLMTUNER_GLOBAL_BATCH_SIZE", "8")),
            max_seq_len=int(os.environ.get("LLMTUNER_MAX_SEQ_LEN", "4096")),
            steps=int(os.environ.get("LLMTUNER_STEPS", "100")),
            seed=42,
            deterministic=deterministic,
            activation_checkpoint_mode="full",
            checkpoint_config=CheckpointConfig(
                enable=True,
                # Keep training checkpoints local: full TP gather can exhaust
                # device memory before HF initial loading even begins.
                sharded_tensor_parallel=(
                    os.environ.get(
                        "LLMTUNER_SHARDED_TENSOR_CHECKPOINT",
                        "1" if tensor_parallel_size > 1 else "0",
                    )
                    == "1"
                ),
                sharded_expert_parallel=(
                    os.environ.get(
                        "LLMTUNER_SHARDED_EXPERT_CHECKPOINT",
                        "1" if expert_parallel_size > 1 else "0",
                    )
                    == "1"
                ),
                interval=int(os.environ.get("LLMTUNER_CHECKPOINT_INTERVAL", "500")),
                load_step=int(os.environ.get("LLMTUNER_CHECKPOINT_LOAD_STEP", "-1")),
                load_only=not save_checkpoint,
                initial_load_path=model_path,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                # This is a training checkpoint, not an export artifact: keep
                # optimizer, scheduler, dataloader, and counters so the final
                # step can be resumed safely.
                last_save_model_only=False,
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=dataset_path,
                tokenizer_path=model_path,
                prompt_field=os.environ.get("LLMTUNER_PROMPT_FIELD", "question"),
                response_field=os.environ.get("LLMTUNER_RESPONSE_FIELD", "solution"),
                packing="first_fit",
                # Keep the global packed rows identical across DP degrees.
                # Requires PP=1 and accumulation=1; multiple local rows use
                # block-causal attention to preserve their isolation.
                global_context_rows=(
                    os.environ.get("LLMTUNER_GLOBAL_CONTEXT_ROWS", "0") == "1"
                ),
                # 1 = one document per sequence (plain causal attention, no mask
                # materialized). A larger value packs several documents into a
                # sequence, which needs the block-causal document mask; that is
                # built densely on this backend (see
                # HFTransformerModel._apply_attention) instead of via flex,
                # which Ascend does not have.
                max_num_documents=int(
                    os.environ.get("LLMTUNER_MAX_NUM_DOCUMENTS", "1")
                ),
            ),
            metrics_config=MetricsConfig(
                log_freq=1,
                enable_tensorboard=os.environ.get("LLMTUNER_ENABLE_TENSORBOARD", "0")
                == "1",
            ),
            dump_folder=os.environ.get(
                "LLMTUNER_DUMP_FOLDER",
                "/home/jianzhnie/llmtuner/llm/TorchLLMTuner/outputs/qwen3-30b-npu",
            ),
        ),
    )


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError(
            "This example requires torch-npu and a visible Ascend NPU. "
            "Run it through examples/train_qwen3_30b_npu.sh."
        )
    torch.set_default_dtype(torch.bfloat16)
    trainer_class = Trainer
    if os.environ.get("LLMTUNER_AUDIT_TRAINING", "0") == "1":
        from examples.npu_training_audit import AuditedNPUTrainer

        trainer_class = AuditedNPUTrainer
    trainer_class(qwen3_30b_npu_config()).train()


if __name__ == "__main__":
    main()
