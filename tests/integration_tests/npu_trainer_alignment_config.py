"""Configuration helpers for opt-in Qwen3-30B Trainer alignment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

QWEN3_30B = "qwen3_30b_a3b"
MODEL_PATH = "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B"
DATASET_PATH = "/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl"
TARGET_PARALLELISM = {
    "data_parallel_replicate_degree": 1,
    "data_parallel_shard_degree": 8,
    "tensor_parallel_degree": 1,
    "expert_parallel_degree": 1,
    "pipeline_parallel_degree": 1,
    "context_parallel_degree": 1,
}


def selected_model(environ: Mapping[str, str] | None = None) -> str | None:
    environ = os.environ if environ is None else environ
    model = environ.get("LLMTUNER_ALIGNMENT_MODEL")
    if model not in (None, QWEN3_30B):
        raise ValueError(
            f"Unsupported LLMTUNER_ALIGNMENT_MODEL={model!r}; expected {QWEN3_30B!r}"
        )
    return model


def is_qwen3_30b(config: Mapping[str, Any]) -> bool:
    return config["model_spec"]["flavor"] == "30B-A3B"


def construction_dtype(captured_30b: bool):
    import torch

    return torch.bfloat16 if captured_30b else None


def baseline_cli_args(
    argv: Sequence[str], environ: Mapping[str, str] | None = None
) -> list[str]:
    args = list(argv)
    if selected_model(environ) != QWEN3_30B:
        return args
    if any(arg == "--module" or arg.startswith("--module=") for arg in args):
        raise ValueError("30B alignment owns --module; do not override it")
    if any(arg == "--config" or arg.startswith("--config=") for arg in args):
        raise ValueError("30B alignment owns --config; do not override it")
    return ["--module", "qwen3", "--config", QWEN3_30B, *args]


def limo_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["solution"]},
    ]


def read_hf_architecture(hf_assets_path: str | Path) -> dict[str, Any]:
    path = Path(hf_assets_path)
    architecture = json.loads((path / "config.json").read_text())
    if architecture.get("model_type") != "qwen3_moe":
        raise ValueError("30B alignment requires a local Qwen3-MoE HF directory")
    if architecture.get("tie_word_embeddings") is not False:
        raise ValueError("30B alignment requires an independent lm_head")
    if not (path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"HF weight index missing under {path}")
    return architecture


def _disable_moe_balancing(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    disabled = []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            current = getattr(value, field.name)
            path = f"{prefix}.{field.name}" if prefix else field.name
            if field.name == "load_balance_coeff" and current is not None:
                disabled.append({"path": path, "previous": current, "current": None})
                setattr(value, field.name, None)
            elif field.name == "aux_loss" and current is not None:
                disabled.append(
                    {"path": path, "previous": repr(current), "current": None}
                )
                setattr(value, field.name, None)
            else:
                disabled.extend(_disable_moe_balancing(current, path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            disabled.extend(_disable_moe_balancing(item, f"{prefix}.{index}"))
    return disabled


def ensure_baseline_timeout_api() -> None:
    import logging

    import torch

    if hasattr(torch.distributed, "set_timeout"):
        return

    def set_timeout_compat(timeout, group=None):
        if group is None:
            group = torch.distributed.distributed_c10d._get_default_group()
        backend = group._get_backend(torch.device("npu"))
        backend._set_default_timeout(timeout)
        logging.warning(
            "Explicit baseline probe: HCCL _set_default_timeout(%s)", timeout
        )

    torch.distributed.set_timeout = set_timeout_compat


def configure_baseline_30b(config: Any, world_size: int) -> list[dict[str, Any]]:
    ensure_baseline_timeout_api()
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.hf_datasets.text_datasets import ChatDataLoader

    if config.model_spec.flavor != "30B-A3B":
        raise ValueError("30B alignment must derive from qwen3_30b_a3b")
    if world_size != 8:
        raise ValueError(f"30B alignment requires WORLD_SIZE=8, got {world_size}")
    model_path = os.environ.get("LLMTUNER_QWEN3_30B_PATH", MODEL_PATH)
    dataset_path = os.environ.get("LLMTUNER_DATASET_PATH", DATASET_PATH)
    architecture = read_hf_architecture(model_path)
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(f"LIMO dataset not found: {dataset_path}")

    config.hf_assets_path = model_path
    for name, degree in TARGET_PARALLELISM.items():
        setattr(config.parallelism, name, degree)
    config.parallelism.spmd_backend = "partial_dtensor"
    config.training.local_batch_size = 1
    config.training.global_batch_size = 8
    config.training.seq_len = 4096
    config.training.steps = 1
    config.training.dtype = "bfloat16"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "float32"
    config.training.disable_cuda_graphs = True
    config.activation_checkpoint = FullAC.Config()
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.debug.moe_force_load_balance = False
    config.optimizer.implementation = "for-loop"
    config.lr_scheduler.decay_ratio = 0.0
    config.metrics.log_freq = 1
    config.loss = CrossEntropyLoss.Config(global_vocab_size=architecture["vocab_size"])
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": dataset_path, "split": "train"},
        sample_processor=limo_messages,
    )
    config.checkpoint.enable = True
    config.checkpoint.load_only = True
    config.checkpoint.load_step = -1
    config.checkpoint.initial_load_path = model_path
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_in_hf_quantized = False
    disabled = _disable_moe_balancing(config.model_spec.model)
    if not any(item["path"].endswith("load_balance_coeff") for item in disabled):
        raise ValueError("30B baseline did not expose load_balance_coeff to disable")
    config.__post_init__()
    return disabled


def _enabled_balancing(value: Any, prefix: str = "") -> list[str]:
    enabled = []
    if isinstance(value, Mapping):
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else name
            if name in {"load_balance_coeff", "aux_loss"} and item is not None:
                enabled.append(path)
            else:
                enabled.extend(_enabled_balancing(item, path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            enabled.extend(_enabled_balancing(item, f"{prefix}.{index}"))
    return enabled


def validate_baseline_30b_dict(config: Mapping[str, Any], world_size: int) -> None:
    if not is_qwen3_30b(config) or world_size != 8:
        raise ValueError("Replay requires the 8-rank Qwen3-30B-A3B capture")
    enabled = _enabled_balancing(config["model_spec"]["model"])
    if enabled:
        raise ValueError(f"Captured MoE balancing/aux loss was not disabled: {enabled}")
    for name, degree in TARGET_PARALLELISM.items():
        if config["parallelism"][name] != degree:
            raise ValueError(
                f"Unexpected captured {name}: {config['parallelism'][name]}"
            )
    if config["parallelism"]["spmd_backend"] != "partial_dtensor":
        raise ValueError("30B capture must use the partial_dtensor FSDP path")
    training = config["training"]
    expected_training = {
        "local_batch_size": 1,
        "global_batch_size": 8,
        "seq_len": 4096,
        "steps": 1,
        "dtype": "bfloat16",
        "mixed_precision_param": "bfloat16",
        "mixed_precision_reduce": "float32",
    }
    for name, expected in expected_training.items():
        if training[name] != expected:
            raise ValueError(f"Unexpected captured training.{name}: {training[name]}")
    if config["debug"]["seed"] != 42 or not config["debug"]["deterministic"]:
        raise ValueError("30B capture must use deterministic seed 42")
    if config["optimizer"]["implementation"] != "for-loop":
        raise ValueError("30B capture must use AdamW for-loop")
    if (
        config["loss"].get("global_vocab_size") is None
        or "num_chunks" in config["loss"]
    ):
        raise ValueError("30B capture must use non-chunked cross entropy")


def build_candidate_30b_config(
    baseline: Mapping[str, Any], output: Path, world_size: int
):
    from llmtuner.config.checkpoint_config import CheckpointConfig
    from llmtuner.config.data_config import DataloaderConfig
    from llmtuner.config.llmtuner_config import LLMTunerConfig
    from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
    from llmtuner.config.model_config import ModelConfig
    from llmtuner.config.optimizer_config import OptimizerConfig
    from llmtuner.config.parallel_config import (
        ExpertParallelConfig,
        FSDPConfig,
        ParallelConfig,
    )
    from llmtuner.config.training_config import TrainingConfig

    validate_baseline_30b_dict(baseline, world_size)
    model_path = baseline["hf_assets_path"]
    dataset_path = baseline["dataloader"]["load_dataset_kwargs"]["data_files"]
    architecture = read_hf_architecture(model_path)
    param_groups = baseline["optimizer"]["param_groups"]
    if len(param_groups) != 1 or param_groups[0]["pattern"] != ".*":
        raise ValueError("30B replay requires one catch-all baseline optimizer group")
    group = param_groups[0]
    if group["optimizer_name"] != "AdamW":
        raise ValueError("30B replay requires baseline AdamW")
    options = group["optimizer_kwargs"]
    scheduler = baseline["lr_scheduler"]
    if scheduler["decay_ratio"] is None:
        raise ValueError(
            "Captured scheduler decay_ratio=None has no candidate equivalent"
        )
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
            vocab_size=architecture["vocab_size"],
            hidden_size=architecture["hidden_size"],
            intermediate_size=architecture["intermediate_size"],
            num_hidden_layers=architecture["num_hidden_layers"],
            num_attention_heads=architecture["num_attention_heads"],
            num_key_value_heads=architecture["num_key_value_heads"],
            experts_implementation="torchtitan",
            norm_backend="torch",
            qkv_projection="fused",
            rotary_compute_dtype="float32",
            attention_backend="flex",
        ),
        parallel=ParallelConfig(
            data_parallel_replicate_size=1,
            data_parallel_shard_size=8,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_kv_allgather_size=1,
            fsdp_config=FSDPConfig(fsdp_reduce_dtype="float32"),
            ep_config=ExpertParallelConfig(router_aux_loss_coef=0.0),
        ),
        optimizer=OptimizerConfig(
            optimizer_type="AdamW",
            learning_rate=options["lr"],
            weight_decay=options["weight_decay"],
            betas=list(options["betas"]),
            eps=options["eps"],
            implementation=baseline["optimizer"]["implementation"],
            lr_scheduler_config=LRSchedulerConfig(
                warmup_steps=scheduler["warmup_steps"],
                total_steps=scheduler["total_steps"],
                decay_ratio=scheduler["decay_ratio"],
                decay_type=scheduler["decay_type"],
                min_lr_factor=scheduler["min_lr_factor"],
            ),
        ),
        training=TrainingConfig(
            steps=baseline["training"]["steps"],
            max_seq_len=4096,
            global_batch_size=8,
            dump_folder=str(output),
            seed=42,
            deterministic=True,
            activation_checkpoint_mode="full",
            chunked_loss_num_chunks=1,
            checkpoint_config=CheckpointConfig(
                enable=True,
                load_only=True,
                initial_load_path=model_path,
                initial_load_model_only=True,
                initial_load_in_hf=True,
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=dataset_path,
                tokenizer_path=model_path,
                prompt_field="question",
                response_field="solution",
            ),
            max_norm=baseline["training"]["max_norm"],
        ),
    )


def debug_candidate_config(
    baseline: Mapping[str, Any], output: Path, steps: int, world: int
):
    from llmtuner.config.llmtuner_config import LLMTunerConfig
    from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
    from llmtuner.config.model_config import ModelConfig
    from llmtuner.config.optimizer_config import OptimizerConfig
    from llmtuner.config.training_config import TrainingConfig

    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=3072,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=8,
            norm_backend=os.environ.get("LLMTUNER_ALIGNMENT_NORM_BACKEND", "native"),
            qkv_projection=os.environ.get(
                "LLMTUNER_ALIGNMENT_QKV_PROJECTION", "separate"
            ),
            rotary_init_device=os.environ.get(
                "LLMTUNER_ALIGNMENT_ROTARY_INIT_DEVICE", "constructor"
            ),
            attention_backend=os.environ.get(
                "LLMTUNER_ALIGNMENT_ATTENTION_BACKEND", "auto"
            ),
            arch_overrides={
                "head_dim": 128,
                "tie_word_embeddings": True,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            },
        ),
        optimizer=OptimizerConfig(
            learning_rate=8e-4,
            weight_decay=0.1,
            betas=[0.9, 0.95],
            implementation=baseline["optimizer"]["implementation"],
            lr_scheduler_config=LRSchedulerConfig(
                warmup_steps=2, total_steps=steps, decay_ratio=0.8
            ),
        ),
        training=TrainingConfig(
            steps=steps,
            max_seq_len=32,
            global_batch_size=world,
            dump_folder=str(output),
            deterministic=True,
        ),
    )


def call_after_load(load: Callable, capture: Callable, *args, **kwargs):
    result = load(*args, **kwargs)
    capture()
    return result


def map_capture_to_candidate_hf(
    state: dict, candidate_keys: set[str], num_experts: int
):
    import torch

    fused_pattern = re.compile(r"(.+\.mlp\.experts\.)(gate_up_proj|down_proj)")
    for candidate_name in candidate_keys:
        match = fused_pattern.fullmatch(candidate_name)
        if match is None:
            continue
        prefix, projection = match.groups()
        source_names = [
            f"{prefix}{expert}.{role}_proj.weight"
            for expert in range(num_experts)
            for role in (("gate", "up") if projection == "gate_up_proj" else ("down",))
        ]
        if any(name not in state for name in source_names):
            raise ValueError(f"Missing HF expert projections for {candidate_name}")
        if projection == "gate_up_proj":
            value = torch.stack(
                [
                    torch.cat(
                        (
                            state[f"{prefix}{expert}.gate_proj.weight"],
                            state[f"{prefix}{expert}.up_proj.weight"],
                        ),
                        dim=0,
                    )
                    for expert in range(num_experts)
                ]
            )
        else:
            value = torch.stack(
                [
                    state[f"{prefix}{expert}.down_proj.weight"]
                    for expert in range(num_experts)
                ]
            )
        for name in source_names:
            del state[name]
        state[candidate_name] = value
    return state


def map_candidate_to_canonical_hf(state: dict):
    fused_pattern = re.compile(r"(.+\.mlp\.experts\.)(gate_up_proj|down_proj)")
    for name in list(state):
        match = fused_pattern.fullmatch(name)
        if match is None:
            continue
        prefix, projection = match.groups()
        fused = state.pop(name)
        for expert, expert_value in enumerate(fused.unbind(0)):
            if projection == "gate_up_proj":
                if expert_value.shape[0] % 2:
                    raise ValueError(f"Invalid fused gate/up shape for {name}")
                gate, up = expert_value.chunk(2, dim=0)
                state[f"{prefix}{expert}.gate_proj.weight"] = gate
                state[f"{prefix}{expert}.up_proj.weight"] = up
            else:
                state[f"{prefix}{expert}.down_proj.weight"] = expert_value
    return state


def tensor_fingerprint(value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        raw = local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {
            "shape": list(local.shape),
            "dtype": str(local.dtype),
            "sha256": hashlib.sha256(raw.data).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {name: tensor_fingerprint(item) for name, item in value.items()}
    if isinstance(value, tuple | list):
        return [tensor_fingerprint(item) for item in value]
    return value


def input_fingerprints(batches: Sequence[Mapping], forwards: Sequence[Mapping]):
    result = []
    for batch, forward in zip(batches, forwards, strict=True):
        values = batch["kwargs"]
        result.append(
            {
                "input": tensor_fingerprint(values["input_dict"]["input"]),
                "labels": tensor_fingerprint(values["labels"]),
                "global_valid_tokens": tensor_fingerprint(
                    values["global_valid_tokens"]
                ),
                "forward_input": tensor_fingerprint(forward["args"][0]),
                "positions": tensor_fingerprint(forward["kwargs"]["positions"]),
            }
        )
    return result


def summed_cross_entropy(logits, labels):
    import torch.nn.functional as F

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.flatten(),
        reduction="sum",
    )


def mesh_manifest(parallel_dims: Any) -> dict[str, Any]:
    if parallel_dims is None:
        return {
            "dp_replicate": 1,
            "dp_shard": 1,
            "tp": 1,
            "ep": 1,
            "pp": 1,
            "cp": 1,
            "world_size": 1,
            "meshes": {},
        }
    result = {
        name: getattr(parallel_dims, name)
        for name in ("dp_replicate", "dp_shard", "tp", "ep", "pp", "cp", "world_size")
    }
    meshes = {}
    for name, mesh in getattr(parallel_dims, "_global_meshes", {}).items():
        meshes[name] = {
            "shape": list(mesh.shape),
            "axis_names": list(mesh.mesh_dim_names),
            "ranks": mesh.mesh.detach().cpu().tolist(),
        }
    result["meshes"] = meshes
    return result


def optimizer_manifest(trainer: Any) -> dict[str, Any]:
    container = getattr(trainer, "optimizers", getattr(trainer, "optimizer", None))
    optimizers = list(getattr(container, "optimizers", [container]))
    result = []
    for optimizer in optimizers:
        groups = []
        for group in optimizer.param_groups:
            values = {
                name: value
                for name, value in group.items()
                if name not in {"params", "param_names"}
            }
            values["num_params"] = len(group["params"])
            groups.append(values)
        result.append({"type": type(optimizer).__name__, "param_groups": groups})
    scheduler = getattr(
        trainer, "lr_schedulers", getattr(trainer, "lr_scheduler", None)
    )
    return {
        "optimizers": result,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "lr_metrics": scheduler.get_metrics() if scheduler is not None else {},
    }


def expected_dtensor_shard(value: Any, expected: Any):
    import torch
    from torch.distributed.tensor import Replicate, Shard

    for axis, placement in enumerate(value.placements):
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, Shard):
            raise ValueError(f"Unsupported final-weight placement: {placement}")
        coordinate = value.device_mesh.get_coordinate()[axis]
        expected = torch.chunk(
            expected, value.device_mesh.size(axis), dim=placement.dim
        )[coordinate]
    return expected


def iter_local_state(state: dict[str, Any], baseline: dict[str, Any]):
    from torch.distributed.tensor import DTensor

    if state.keys() != baseline.keys():
        raise ValueError(
            "Local state key mismatch: "
            f"missing={sorted(baseline.keys() - state.keys())[:8]}, "
            f"unexpected={sorted(state.keys() - baseline.keys())[:8]}"
        )
    for name, value in state.items():
        expected = baseline[name]
        if isinstance(value, DTensor):
            expected = expected_dtensor_shard(value, expected)
            value = value.to_local()
        value = value.detach().cpu().clone()
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"Local state metadata mismatch for {name}")
        yield name, value, expected
