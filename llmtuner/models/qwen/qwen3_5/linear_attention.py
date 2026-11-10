"""Opt-in recurrence dispatch inside the unchanged HF Qwen3.5 MoE module.

Only the chunk recurrence callable is replaced. Projections, convolution,
decay, gated normalization, parameters and checkpoint keys remain HF-owned.
The initial contract is BF16 35B-A3B geometry, eager pure FSDP, single-row
unpacked training, with no cache. CPU tests validate routing and rejection;
they do not establish NPU, fullAC or end-to-end numerical acceptance.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from types import FunctionType, MethodType
from typing import Any

import torch
from torch import nn

from ....datasets.contracts import Batch
from ....errors import EnvironmentUnsupportedError, UnsupportedCombinationError
from .padding_metadata import (
    validate_gdn_forward_metadata,
    validate_gdn_padding_config,
    validate_gdn_sequence_metadata,
    validate_qwen35_gdn_geometry,
)


def validate_linear_attention_config(config) -> None:
    validate_gdn_padding_config(config)
    backend = getattr(config, "linear_attention_backend", "native")
    deterministic = getattr(config, "gdn_deterministic_reduction", False)
    if not isinstance(deterministic, bool):
        raise UnsupportedCombinationError("gdn_deterministic_reduction must be bool")
    if deterministic and backend != "npu_triton":
        raise UnsupportedCombinationError(
            "gdn_deterministic_reduction requires linear_attention_backend=npu_triton"
        )
    if backend == "native":
        return
    if backend != "npu_triton":
        raise UnsupportedCombinationError(
            f"Unknown linear attention backend: {backend}"
        )
    validate_qwen35_gdn_geometry(config, backend=backend)
    if getattr(config, "attn_mask_type", "causal") != "causal":
        raise UnsupportedCombinationError(
            "npu_triton does not support packed documents"
        )


def validate_linear_attention_training(cfg, config, *, device, parallel_dims) -> None:
    """Reject unsupported runs before allocating the HF model or sharding it."""
    validate_linear_attention_config(config)
    if getattr(cfg.model, "linear_attention_backend", "native") == "native":
        if getattr(cfg.model, "gdn_deterministic_reduction", False):
            raise UnsupportedCombinationError(
                "gdn_deterministic_reduction requires "
                "linear_attention_backend=npu_triton"
            )
        return
    if device.type != "npu":
        raise UnsupportedCombinationError("npu_triton requires an NPU training device")
    if any(
        getattr(cfg.parallel, f"{axis}_size") != 1 for axis in ("tp", "cp", "ep", "pp")
    ):
        raise UnsupportedCombinationError(
            "npu_triton supports pure FSDP only; no TP/CP/EP/PP"
        )
    if parallel_dims is None or parallel_dims.dp_shard <= 1:
        raise UnsupportedCombinationError("npu_triton requires sharded FSDP")
    if cfg.training.compile or cfg.training.activation_checkpoint_mode not in (
        "none",
        "full",
    ):
        raise UnsupportedCombinationError(
            "npu_triton supports eager training with none/full AC only"
        )
    if cfg.training.activation_swap or cfg.parallel.fsdp_config.fsdp_cpu_offload:
        raise UnsupportedCombinationError(
            "npu_triton offload/activation swap is not validated"
        )
    if (
        cfg.dataloader.global_context_rows
        or cfg.dataloader.attention_mask_type() != "causal"
    ):
        raise UnsupportedCombinationError(
            "npu_triton does not support packed/global context rows"
        )
    # Each local microbatch must contain one independent recurrence stream.
    if (
        cfg.training.global_batch_size
        != parallel_dims.dp_shard * parallel_dims.dp_replicate
    ):
        raise UnsupportedCombinationError(
            "npu_triton requires one batch row per DP rank"
        )
    if cfg.training.max_seq_len <= 0 or cfg.training.max_seq_len % 64:
        raise UnsupportedCombinationError(
            "npu_triton sequence length must be a positive multiple of 64"
        )
    if torch.get_default_dtype() != torch.bfloat16:
        raise UnsupportedCombinationError(
            "npu_triton requires BF16 parameter construction (default dtype)"
        )


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    *,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    _configured_deterministic_reduction=False,
    **kwargs,
):
    """HF chunk API -> local first-order NPU op; never silently drop metadata."""
    if initial_state is not None or output_final_state:
        raise UnsupportedCombinationError(
            "npu_triton does not support cache/final state"
        )
    if not use_qk_l2norm_in_kernel:
        raise UnsupportedCombinationError("npu_triton requires QK L2 normalization")
    if cu_seqlens is not None or kwargs:
        raise UnsupportedCombinationError(
            "npu_triton does not support packed/extra recurrence metadata"
        )
    # The optional accelerator dependency is never imported on the native path.
    op = import_module("llmtuner.accelerator.gated_delta").gated_delta_rule
    if _configured_deterministic_reduction:
        return op(query, key, value, g, beta, deterministic_reduction=True), None
    return op(query, key, value, g, beta), None


def _configured_chunk_rule(deterministic, *, forward_metadata=False):
    """Bind installation policy; HF call metadata cannot override it."""

    def recurrence(*args, **kwargs):
        if {
            "deterministic_reduction",
            "_configured_deterministic_reduction",
        } & kwargs.keys():
            raise UnsupportedCombinationError(
                "GDN deterministic reduction is fixed by model configuration"
            )
        if forward_metadata:
            # New HF forwards propagate these already-validated observation and
            # position fields to recurrence; they do not alter its arithmetic.
            for key in (
                "position_ids",
                "cache_position",
                "use_cache",
                "is_causal",
                "output_router_logits",
                "output_attentions",
                "output_hidden_states",
            ):
                kwargs.pop(key, None)
        return chunk_gated_delta_rule(
            *args, _configured_deterministic_reduction=deterministic, **kwargs
        )

    return recurrence


def validate_gdn_inputs(module, args, kwargs):
    """Guard direct HF calls too, including single-token cache bypass paths."""
    hidden = validate_gdn_forward_metadata(module, args, kwargs)
    if hidden.dtype != torch.bfloat16 or hidden.device.type != "npu":
        raise UnsupportedCombinationError("npu_triton requires BF16 NPU activations")


def _module_level_forward(module, hf, deterministic):
    """Bind new HF function dispatch locally, without changing global HF code.

    HF 5.22 moved recurrence from instance fields to module-level functions.
    Reuse its original forward bytecode with a private dispatch namespace and
    reapply its child-weight hooks. Parameters, projections and cache guards
    remain owned by the original HF instance.
    """
    forward = inspect.unwrap(type(module).forward)
    if (
        "forward" in vars(module)
        or forward.__closure__
        or not {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"}
        <= set(forward.__code__.co_names)
    ):
        raise UnsupportedCombinationError("Unrecognized HF GDN function dispatch")
    namespace = dict(forward.__globals__)
    namespace["torch_chunk_gated_delta_rule"] = _configured_chunk_rule(
        deterministic, forward_metadata=True
    )
    namespace["causal_conv1d_fn"] = inspect.unwrap(hf.causal_conv1d_fn)
    local = FunctionType(
        forward.__code__, namespace, forward.__name__, forward.__defaults__
    )
    local.__kwdefaults__ = forward.__kwdefaults__
    return MethodType(hf.force_accelerate_hooks("conv1d")(local), module)


def apply_linear_attention_backend(model: nn.Module) -> None:
    config = model.config
    validate_linear_attention_config(config)
    if getattr(config, "linear_attention_backend", "native") == "native":
        return
    hf = import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    modules = [m for m in model.modules() if type(m) is hf.Qwen3_5MoeGatedDeltaNet]
    if not modules:
        raise UnsupportedCombinationError(
            "npu_triton found no native HF GatedDeltaNet modules"
        )
    # Validate all instances before mutating any instance. No global HF patches.
    forwards = {}
    for module in modules:
        if (
            type(module.norm) is not hf.Qwen3_5MoeRMSNormGated
            or getattr(module, "causal_conv1d_fn", None) is not None
        ):
            raise UnsupportedCombinationError(
                "npu_triton requires native torch HF norm and convolution"
            )
        if not hasattr(module, "chunk_gated_delta_rule"):
            forwards[module] = _module_level_forward(
                module, hf, getattr(config, "gdn_deterministic_reduction", False)
            )
    try:
        import_module("triton")
        import_module("llmtuner.accelerator.gated_delta")
    except ImportError as exc:
        raise EnvironmentUnsupportedError(
            "npu_triton requires the independent GDN op and Ascend-compatible Triton"
        ) from exc
    for module in modules:
        module._llmtuner_gdn_allow_right_padding = getattr(
            config, "gdn_allow_right_padding", False
        )
        if module in forwards:
            module.forward = forwards[module]
        else:
            module.chunk_gated_delta_rule = _configured_chunk_rule(
                getattr(config, "gdn_deterministic_reduction", False)
            )
        module.register_forward_pre_hook(validate_gdn_inputs, with_kwargs=True)


def validate_linear_attention_batch(
    input_dict: Batch | dict[str, Any],
    *,
    allow_right_padding: bool = False,
    validate_metadata: bool = True,
) -> None:
    """Reject boundaries before preprocessing can flatten independent rows."""
    if isinstance(input_dict, Batch):
        inputs = input_dict.input_ids
        positions = padding = None
    else:
        allowed = {"input", "labels", "positions", "padding_mask", "num_valid_tokens"}
        if input_dict.keys() - allowed:
            raise UnsupportedCombinationError(
                "npu_triton does not support extra batch metadata"
            )
        inputs = input_dict["input"]
        positions, padding = input_dict.get("positions"), input_dict.get("padding_mask")
    if inputs.ndim not in (1, 2) or (inputs.ndim == 2 and inputs.shape[0] != 1):
        raise UnsupportedCombinationError(
            "npu_triton requires one independent batch row"
        )
    validate_linear_attention_sequence(
        inputs.reshape(-1),
        positions,
        None,
        padding,
        allow_right_padding=allow_right_padding,
        validate_metadata=validate_metadata,
    )


def validate_linear_attention_sequence(
    inputs,
    positions,
    masks,
    padding,
    *,
    allow_right_padding: bool = False,
    validate_metadata: bool = True,
) -> None:
    validate_gdn_sequence_metadata(
        inputs,
        positions,
        masks,
        padding,
        allow_right_padding=allow_right_padding,
        validate_metadata=validate_metadata,
    )
