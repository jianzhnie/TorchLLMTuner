"""Configuration helpers for opt-in Qwen3-30B Trainer alignment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

QWEN3_30B = "qwen3_30b_a3b"
BASELINE_30B_MODULE = "torchtitan_npu.models.qwen3.config_registry"
BASELINE_30B_CONFIG = "sft_qwen3_30ba3b_gsm8k"
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
TARGET_LAYOUTS = {
    "dp8": (8, 1, 1),
    "dp4-tp2": (4, 2, 1),
    "tp2-ep4": (4, 2, 4),
    "dp8-ep8": (8, 1, 8),
}


def target_parallelism(environ: Mapping[str, str] | None = None) -> dict[str, int]:
    environ = os.environ if environ is None else environ
    name = environ.get("LLMTUNER_ALIGNMENT_LAYOUT", "dp8")
    if name not in TARGET_LAYOUTS:
        raise ValueError(f"Unsupported alignment layout {name!r}")
    dp, tp, ep = TARGET_LAYOUTS[name]
    return {
        **TARGET_PARALLELISM,
        "data_parallel_shard_degree": dp,
        "tensor_parallel_degree": tp,
        "expert_parallel_degree": ep,
    }


def alignment_runtime_manifest(module_names: Sequence[str]) -> dict[str, Any]:
    """Identify actual imported sources, independently of configured search paths."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for distribution in ("torch", "torch-npu", "torchtitan", "attn-gym", "spmd-types"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    sources = {}
    for name in module_names:
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise ValueError(f"Alignment source module is not loaded: {name}")
        path = Path(filename).resolve()
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    torch_module = sys.modules.get("torch")
    timeout_api = getattr(
        getattr(torch_module, "distributed", None), "set_timeout", None
    )
    npu = getattr(torch_module, "npu", None)
    return {
        "python": sys.version,
        "versions": versions,
        "sources": sources,
        "npu_allocator_config": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
        "npu_allow_hf32": {
            name: getattr(npu, name).allow_hf32
            for name in ("matmul", "conv", "aclnn")
            if hasattr(npu, name)
        },
        "diagnostic_timeout_patch": bool(
            getattr(timeout_api, "_llmtuner_diagnostic_patch", False)
        ),
    }


def selected_model(environ: Mapping[str, str] | None = None) -> str | None:
    environ = os.environ if environ is None else environ
    model = environ.get("LLMTUNER_ALIGNMENT_MODEL")
    if model not in (None, QWEN3_30B):
        raise ValueError(
            f"Unsupported LLMTUNER_ALIGNMENT_MODEL={model!r}; expected {QWEN3_30B!r}"
        )
    return model


def validate_capture_runtime(manifest: Mapping[str, Any]) -> None:
    runtime = manifest.get("runtime", {})
    if runtime.get("diagnostic_timeout_patch") is not False:
        raise ValueError(
            "Formal 30B replay requires a capture with explicit unpatched "
            "runtime provenance; diagnostic or legacy captures are insufficient"
        )


def validate_capture_hf32(manifest: Mapping[str, Any], expected: bool) -> None:
    settings = manifest.get("runtime", {}).get("npu_allow_hf32", {})
    if any(
        type(settings.get(name)) is not bool for name in ("matmul", "conv", "aclnn")
    ):
        raise ValueError("Capture lacks actual NPU HF32 runtime settings; recapture")
    if any(settings[name] != expected for name in ("matmul", "conv", "aclnn")):
        raise ValueError("Captured NPU HF32 runtime differs from configuration")


def is_qwen3_30b(config: Mapping[str, Any]) -> bool:
    return config["model_spec"]["flavor"] == "30B-A3B"


def construction_dtype(captured_30b: bool):
    import torch

    return torch.bfloat16 if captured_30b else None


def baseline_config_snapshot(config: Any) -> dict[str, Any]:
    """Retain model arithmetic omitted by TorchTitan 2.12's Trainer serializer."""
    result = config.to_dict()
    result["model_spec"] = {
        **result["model_spec"],
        "model": config.model_spec.model.to_dict(),
    }
    attention = {
        (
            type(layer.attention.inner_attention).__qualname__.split(".")[-2],
            layer.attention.mask_type,
        )
        for layer in config.model_spec.model.layers
    }
    supported = {
        ("ScaledDotProductAttention", "causal"): "sdpa",
        ("FlexAttention", "block_causal"): "flex",
    }
    if len(attention) != 1 or next(iter(attention)) not in supported:
        raise ValueError(f"Unsupported baseline attention contract: {attention}")
    implementation, mask_type = next(iter(attention))
    result["alignment_attention"] = {
        "backend": supported[(implementation, mask_type)],
        "mask_type": mask_type,
    }
    return result


def captured_attention(config: Mapping[str, Any]) -> Mapping[str, str]:
    settings = config.get("alignment_attention", {})
    if (settings.get("backend"), settings.get("mask_type")) not in {
        ("sdpa", "causal"),
        ("flex", "block_causal"),
    }:
        raise ValueError("Capture lacks a supported explicit attention contract")
    return settings


def effective_forward_positions(forward: Mapping[str, Any]):
    """Resolve the model's default sequential RoPE positions without changing it."""
    import torch

    positions = forward["kwargs"].get("positions")
    if positions is not None:
        return positions
    inputs = forward["args"][0]
    return torch.arange(inputs.shape[-1], dtype=torch.int64).expand(inputs.shape)


def baseline_allow_hf32(config: Mapping[str, Any]) -> bool:
    training = config["training"]
    extension = training.get("extension", {})
    values = [
        source["allow_hf32"]
        for source in (training, extension)
        if "allow_hf32" in source
    ]
    if not values or any(type(value) is not bool for value in values):
        raise ValueError("Capture must explicitly record boolean allow_hf32")
    if len(set(values)) != 1:
        raise ValueError("Conflicting captured allow_hf32 settings")
    return values[0]


def configure_baseline_hf32(training, environ=None) -> None:
    """Apply an explicit diagnostic HF32 variant to every owning config field."""
    environ = os.environ if environ is None else environ
    hf32_override = environ.get("LLMTUNER_ALIGNMENT_ALLOW_HF32")
    if hf32_override is not None:
        if hf32_override not in ("0", "1"):
            raise ValueError("LLMTUNER_ALIGNMENT_ALLOW_HF32 must be 0 or 1")
        sources = [training, getattr(training, "extension", None)]
        owners = [source for source in sources if hasattr(source, "allow_hf32")]
        if not owners:
            raise ValueError("Baseline has no configurable allow_hf32 field")
        for source in owners:
            source.allow_hf32 = hf32_override == "1"


def baseline_learning_rate(extra_metrics: Mapping[str, Any]) -> float:
    values = [extra_metrics[key] for key in ("lr/AdamW", "lr") if key in extra_metrics]
    if not values or any(value != values[0] for value in values):
        raise ValueError("Missing or conflicting captured AdamW learning rate")
    return values[0]


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
    return [
        "--module",
        BASELINE_30B_MODULE,
        "--config",
        BASELINE_30B_CONFIG,
        *args,
    ]


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


def alignment_steps(environ: Mapping[str, str] | None = None) -> int:
    """Select a positive update count without changing existing single-step runs."""
    environ = os.environ if environ is None else environ
    value = environ.get("LLMTUNER_ALIGNMENT_STEPS", "1")
    try:
        steps = int(value)
    except ValueError as error:
        raise ValueError(
            "LLMTUNER_ALIGNMENT_STEPS must be a positive integer"
        ) from error
    if steps < 1:
        raise ValueError("LLMTUNER_ALIGNMENT_STEPS must be a positive integer")
    return steps


def configure_baseline_30b(config: Any, world_size: int) -> list[dict[str, Any]]:
    try:
        from torchtitan.components.loss import CrossEntropyLoss
    except ImportError:
        # Torch 2.12's TorchTitan uses the model spec's fixed CE loss builder.
        CrossEntropyLoss = None
    try:
        from torchtitan.distributed.activation_checkpoint import FullAC

        full_ac = FullAC.Config()
    except ImportError:
        from torchtitan.config import ActivationCheckpointConfig

        full_ac = ActivationCheckpointConfig(mode="full")
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
    topology = target_parallelism()
    for name, degree in topology.items():
        setattr(config.parallelism, name, degree)
    if hasattr(config.parallelism, "spmd_backend"):
        config.parallelism.spmd_backend = "partial_dtensor"
    config.training.local_batch_size = 8 // topology["data_parallel_shard_degree"]
    config.training.global_batch_size = 8
    config.training.seq_len = 4096
    config.training.steps = alignment_steps()
    configure_baseline_hf32(config.training)
    config.training.dtype = "bfloat16"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "float32"
    if hasattr(config.training, "disable_cuda_graphs"):
        config.training.disable_cuda_graphs = True
    config.activation_checkpoint = full_ac
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.debug.moe_force_load_balance = False
    config.optimizer.implementation = "for-loop"
    if hasattr(config.optimizer, "swap_optimizer"):
        config.optimizer.swap_optimizer = False
    config.lr_scheduler.decay_ratio = 0.0
    config.metrics.log_freq = 1
    if CrossEntropyLoss is not None:
        config.loss = CrossEntropyLoss.Config(
            global_vocab_size=architecture["vocab_size"]
        )
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
    topology = {key: config["parallelism"][key] for key in TARGET_PARALLELISM}
    allowed = [
        target_parallelism({"LLMTUNER_ALIGNMENT_LAYOUT": name})
        for name in TARGET_LAYOUTS
    ]
    if topology not in allowed:
        raise ValueError(f"Unexpected captured parallelism: {topology}")
    spmd_backend = config["parallelism"].get("spmd_backend")
    if spmd_backend not in (None, "partial_dtensor"):
        raise ValueError("30B capture must use the partial_dtensor FSDP path")
    training = config["training"]
    if type(training["steps"]) is not int or training["steps"] < 1:
        raise ValueError("Captured training.steps must be a positive integer")
    expected_training = {
        "local_batch_size": 8 // topology["data_parallel_shard_degree"],
        "global_batch_size": 8,
        "seq_len": 4096,
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
    loss = config.get("loss", {})
    if loss and (loss.get("global_vocab_size") is None or "num_chunks" in loss):
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
    attention = captured_attention(baseline)
    model_path = baseline["hf_assets_path"]
    dataset_path = baseline["dataloader"]["load_dataset_kwargs"]["data_files"]
    architecture = read_hf_architecture(model_path)
    optimizer = baseline["optimizer"]
    if "param_groups" in optimizer:
        param_groups = optimizer["param_groups"]
        if len(param_groups) != 1 or param_groups[0]["pattern"] != ".*":
            raise ValueError(
                "30B replay requires one catch-all baseline optimizer group"
            )
        group = param_groups[0]
        if group["optimizer_name"] != "AdamW":
            raise ValueError("30B replay requires baseline AdamW")
        options = group["optimizer_kwargs"]
        learning_rate = options["lr"]
        betas = options["betas"]
        eps = options["eps"]
        weight_decay = options["weight_decay"]
    else:
        if optimizer.get("name") != "AdamW":
            raise ValueError("30B replay requires baseline AdamW")
        learning_rate = optimizer["lr"]
        betas = (optimizer["beta1"], optimizer["beta2"])
        eps = optimizer["eps"]
        weight_decay = optimizer["weight_decay"]
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
            norm_backend="npu",
            qkv_projection="separate",
            rotary_compute_dtype="float32",
            rotary_cache_dtype="bfloat16",
            attention_backend="auto" if attention["backend"] == "sdpa" else "flex",
        ),
        parallel=ParallelConfig(
            data_parallel_replicate_size=1,
            data_parallel_shard_size=baseline["parallelism"][
                "data_parallel_shard_degree"
            ],
            tensor_parallel_size=baseline["parallelism"]["tensor_parallel_degree"],
            expert_parallel_size=baseline["parallelism"]["expert_parallel_degree"],
            pipeline_parallel_size=1,
            context_parallel_kv_allgather_size=1,
            fsdp_config=FSDPConfig(fsdp_reduce_dtype="float32"),
            ep_config=ExpertParallelConfig(router_aux_loss_coef=0.0),
        ),
        optimizer=OptimizerConfig(
            optimizer_type="AdamW",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=list(betas),
            eps=eps,
            implementation=optimizer["implementation"],
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
            grad_norm_precision="parameter",
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
                sharded_tensor_parallel=(
                    baseline["parallelism"]["tensor_parallel_degree"] > 1
                ),
                sharded_expert_parallel=(
                    baseline["parallelism"]["expert_parallel_degree"] > 1
                ),
            ),
            dataloader_config=DataloaderConfig(
                dataset="local_jsonl_sft",
                dataset_path=dataset_path,
                tokenizer_path=model_path,
                prompt_field="question",
                response_field="solution",
                max_num_documents=1 if attention["mask_type"] == "causal" else None,
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
        native = re.fullmatch(
            r"(.+\.mlp\.)routed_experts\.inner_experts\.(w1_EFD|w3_EFD|w2_EDF)",
            candidate_name,
        )
        if native is not None:
            prefix, role = native.groups()
            projection = {"w1_EFD": "gate", "w3_EFD": "up", "w2_EDF": "down"}[role]
            names = [
                f"{prefix}experts.{expert}.{projection}_proj.weight"
                for expert in range(num_experts)
            ]
            if any(name not in state for name in names):
                raise ValueError(f"Missing HF expert projections for {candidate_name}")
            state[candidate_name] = torch.stack([state.pop(name) for name in names])
            continue
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
    for name in candidate_keys:
        if ".mlp.router.gate." in name:
            source = name.replace(".mlp.router.gate.", ".mlp.gate.")
            if source in state:
                state[name] = state.pop(source)
    return state


def localize_parallel_reference(
    state: dict[str, Any], model: Any, *, parameter_prefix: str = ""
) -> dict[str, Any]:
    """Slice canonical CPU references using the live physical TP/EP ownership.

    DTensor DP placements are handled separately by iter_local_state. This does
    not distribute tensors or change the model and never moves a full tensor to
    an accelerator merely to verify its local checkpoint shard.
    """
    import torch
    import torch.distributed as dist

    from llmtuner.parallel.expert_parallel.checkpointing import expert_shard_map
    from llmtuner.parallel.tensor_parallel.checkpointing import _index_layouts

    state = dict(state)
    layouts, _ = _index_layouts([model])
    for name, (dimension, packed, group) in layouts.items():
        if parameter_prefix:
            if not name.startswith(parameter_prefix):
                raise ValueError(f"TP parameter outside reference namespace: {name}")
            name = name.removeprefix(parameter_prefix)
        if name not in state:
            raise ValueError(f"Missing TP reference {name}")
        value = state[name]
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        if packed:
            value = torch.cat(
                [
                    half.chunk(world, dimension)[rank]
                    for half in value.chunk(2, dimension)
                ],
                dim=dimension,
            )
        else:
            value = value.chunk(world, dimension)[rank]
        state[name] = value
    for name, group in expert_shard_map([model]).items():
        if parameter_prefix:
            if not name.startswith(parameter_prefix):
                raise ValueError(f"EP parameter outside reference namespace: {name}")
            name = name.removeprefix(parameter_prefix)
        if name not in state:
            raise ValueError(f"Missing EP reference {name}")
        state[name] = state[name].chunk(dist.get_world_size(group), 0)[
            dist.get_rank(group)
        ]
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
                "positions": tensor_fingerprint(effective_forward_positions(forward)),
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
    import torch.distributed as dist

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
    # Inspect existing group handles only; constructing a new mesh here could
    # change collective ordering and would not audit the trainer's groups.
    groups = {}
    if dist.is_initialized():
        result["rank"] = dist.get_rank()
        views = {
            **getattr(parallel_dims, "_global_meshes", {}),
            **getattr(parallel_dims, "_single_axis_meshes", {}),
        }
        views.update(
            {
                "/".join(names): mesh
                for names, mesh in getattr(
                    parallel_dims, "_multi_axis_meshes", {}
                ).items()
            }
        )
        for name, mesh in views.items():
            groups[name] = {
                axis: dist.get_process_group_ranks(mesh.get_group(axis))
                for axis in mesh.mesh_dim_names
            }
    result["local_process_groups"] = groups
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
    get_metrics = getattr(scheduler, "get_metrics", None)
    return {
        "optimizers": result,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "lr_metrics": get_metrics() if callable(get_metrics) else None,
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
        # DTensor pads the rank space with empty shards when torch.chunk
        # returns fewer chunks than mesh ranks (including zero-sized tensors).
        chunks = torch.chunk(expected, value.device_mesh.size(axis), dim=placement.dim)
        expected = (
            chunks[coordinate]
            if coordinate < len(chunks)
            else expected.narrow(placement.dim, expected.size(placement.dim), 0)
        )
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
