"""Explicit Qwen3.5 GDN numerical policy, independent of recurrence dispatch.

The opt-in splits original QKV parameter views into three projections and moves
gated-norm weight multiplication before the BF16 cast. No parameters or state
keys are added. Evidence is isolated first-order, unpadded BF16 comparisons;
this is not full-model, padding, compilation or training acceptance.
"""

from __future__ import annotations

from importlib import import_module
from types import MethodType

import torch
from torch import nn

from ...datasets.contracts import Batch
from ...errors import UnsupportedCombinationError
from ..hf.padding_metadata import (
    validate_gdn_forward_metadata,
    validate_gdn_padding_config,
    validate_gdn_sequence_metadata,
    validate_qwen35_gdn_geometry,
)

POLICY = "split_qkv_fp32_norm"


def validate_gdn_numerical_config(config) -> None:
    validate_gdn_padding_config(config)
    policy = getattr(config, "gdn_numerical_policy", "native")
    if policy == "native":
        return
    if policy != POLICY:
        raise UnsupportedCombinationError(f"Unknown GDN numerical policy: {policy}")
    validate_qwen35_gdn_geometry(config, backend=POLICY)
    if (
        getattr(config, "attn_mask_type", "causal") != "causal"
        or getattr(config, "qkv_projection", "separate") != "separate"
        or getattr(config, "norm_backend", "native") != "native"
    ):
        raise UnsupportedCombinationError(
            f"{POLICY} requires causal documents and native norm/separate QKV options"
        )


def validate_gdn_numerical_training(cfg, config, *, device, parallel_dims) -> None:
    """Fail before construction/sharding; recurrence backend remains independent."""
    if getattr(cfg.model, "gdn_numerical_policy", "native") == "native":
        return
    validate_gdn_numerical_config(config)
    if device.type != "npu":
        raise UnsupportedCombinationError(f"{POLICY} training requires NPU")
    if any(
        getattr(cfg.parallel, f"{axis}_size") != 1 for axis in ("tp", "cp", "ep", "pp")
    ):
        raise UnsupportedCombinationError(
            f"{POLICY} supports local/DP/FSDP, not TP/CP/EP/PP"
        )
    if (
        cfg.training.compile
        or cfg.training.activation_checkpoint_mode not in ("none", "full")
        or cfg.training.activation_swap
        or cfg.parallel.fsdp_config.fsdp_cpu_offload
        or getattr(getattr(cfg.parallel, "ep_config", None), "fixed_router", False)
    ):
        raise UnsupportedCombinationError(
            f"{POLICY} requires eager none/full AC without offload or MoE replacement"
        )
    if (
        cfg.dataloader.global_context_rows
        or cfg.dataloader.attention_mask_type() != "causal"
    ):
        raise UnsupportedCombinationError(
            f"{POLICY} packed/global context is unverified"
        )
    dp = (
        1
        if parallel_dims is None
        else parallel_dims.dp_shard * parallel_dims.dp_replicate
    )
    if cfg.training.global_batch_size != dp:
        raise UnsupportedCombinationError(f"{POLICY} requires one row per DP rank")
    if cfg.training.max_seq_len <= 0 or cfg.training.max_seq_len % 64:
        raise UnsupportedCombinationError(
            f"{POLICY} requires sequence length divisible by 64"
        )
    if torch.get_default_dtype() != torch.bfloat16:
        raise UnsupportedCombinationError(
            f"{POLICY} requires BF16 parameter construction"
        )


def validate_gdn_numerical_sequence(
    inputs, positions, masks, padding, *, allow_right_padding: bool = False
) -> None:
    if padding is not None and (
        padding.dtype != torch.bool or padding.numel() != inputs.numel()
    ):
        raise UnsupportedCombinationError(
            f"{POLICY} padding metadata must match tokens"
        )
    validate_gdn_sequence_metadata(
        inputs,
        positions,
        masks,
        padding,
        allow_right_padding=allow_right_padding,
    )


def validate_gdn_numerical_batch(batch, *, allow_right_padding: bool = False) -> None:
    """Validate row/document boundaries before input preparation flattens them."""
    if isinstance(batch, Batch):
        inputs, positions, padding = batch.input_ids, None, None
    else:
        if batch.keys() - {
            "input",
            "labels",
            "positions",
            "padding_mask",
            "num_valid_tokens",
        }:
            raise UnsupportedCombinationError(
                f"{POLICY} extra batch metadata is unverified"
            )
        inputs = batch["input"]
        positions, padding = batch.get("positions"), batch.get("padding_mask")
    if inputs.ndim not in (1, 2) or (inputs.ndim == 2 and inputs.shape[0] != 1):
        raise UnsupportedCombinationError(
            f"{POLICY} requires one independent batch row"
        )
    validate_gdn_numerical_sequence(
        inputs.reshape(-1),
        positions,
        None,
        padding,
        allow_right_padding=allow_right_padding,
    )


def _validate_forward(module, args, kwargs):
    hidden = validate_gdn_forward_metadata(module, args, kwargs)
    if torch.compiler.is_compiling():
        raise UnsupportedCombinationError(f"{POLICY} requires eager execution")
    if hidden.dtype != torch.bfloat16 or hidden.device.type != "npu":
        raise UnsupportedCombinationError(f"{POLICY} requires BF16 NPU activations")
    if hidden.shape[-1] != 2048:
        raise UnsupportedCombinationError(f"{POLICY} requires width 2048")
    if any(p.dtype != torch.bfloat16 for p in module.parameters()):
        raise UnsupportedCombinationError(f"{POLICY} requires BF16 parameters")


def forward_split_qkv(module, hidden_states):
    """Original HF parameter slices resolved each forward after FSDP unshard."""
    return torch.cat(
        [
            nn.functional.linear(hidden_states, weight)
            for weight in module.weight.split((2048, 2048, 4096), dim=0)
        ],
        dim=-1,
    )


def forward_fp32_weight_norm(module, hidden_states, gate=None):
    dtype = hidden_states.dtype
    hidden = hidden_states.float()
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + module.variance_epsilon)
    hidden = (module.weight.float() * hidden).to(dtype)
    return (hidden * nn.functional.silu(gate.float())).to(dtype)


def configure_gdn_numerics(model: nn.Module) -> None:
    """Install one atomic opt-in on native HF instances; native is a no-op.

    Validate all targets before mutation. Reject fused/custom norm/projection
    overrides rather than silently replacing another numerical policy. Optional
    recurrence packages and reference/test code are never imported here.
    """
    config = model.config
    validate_gdn_numerical_config(config)
    if getattr(config, "gdn_numerical_policy", "native") == "native":
        if any(
            getattr(m, "_llmtuner_gdn_numerical_policy", None) for m in model.modules()
        ):
            raise UnsupportedCombinationError(
                "GDN numerical policy cannot be removed in place"
            )
        return
    hf = import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    modules = [m for m in model.modules() if type(m) is hf.Qwen3_5MoeGatedDeltaNet]
    if not modules:
        raise UnsupportedCombinationError(f"{POLICY} requires native HF GDN modules")
    pending = []
    for module in modules:
        owned = (
            getattr(module.in_proj_qkv.forward, "__func__", None) is forward_split_qkv
            and getattr(module.norm.forward, "__func__", None)
            is forward_fp32_weight_norm
            and _validate_forward in module._forward_pre_hooks.values()
        )
        if (
            type(module.norm) is not hf.Qwen3_5MoeRMSNormGated
            or type(module.in_proj_qkv) is not nn.Linear
            or module.in_proj_qkv.bias is not None
            or module.in_proj_qkv.weight.shape != (8192, 2048)
            or module.norm.weight.shape != (128,)
            or (
                not owned
                and (
                    "forward" in vars(module.in_proj_qkv)
                    or "forward" in vars(module.norm)
                )
            )
            or any(p.dtype != torch.bfloat16 for p in module.parameters())
        ):
            raise UnsupportedCombinationError(
                f"{POLICY} requires unmodified native BF16 QKV/norm parameters"
            )
        if not owned:
            pending.append(module)
    for module in modules:
        module._llmtuner_gdn_allow_right_padding = getattr(
            config, "gdn_allow_right_padding", False
        )
    for module in pending:
        module.in_proj_qkv.forward = MethodType(forward_split_qkv, module.in_proj_qkv)
        module.norm.forward = MethodType(forward_fp32_weight_norm, module.norm)
        module.register_forward_pre_hook(_validate_forward, with_kwargs=True)
        module._llmtuner_gdn_numerical_policy = POLICY
