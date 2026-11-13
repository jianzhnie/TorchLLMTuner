# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Kimi Delta Attention state context parallelism (KCP)."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.distributed import get_comm_device, get_world_size
from ...models.contracts import DecoderStructure
from ...models.kimi.kimi_k3.delta_attention import (
    KimiDeltaAttention,
    forward_context_parallel_kda,
    validate_context_parallel_kda,
)
from ...models.kimi.kimi_k3.latent_attention import KimiLatentAttention
from ..plan import ParallelPlan, PlanTarget
from .attention import HeadToSeq, SeqToHead

try:
    from fla.modules.conv.causal_conv1d import causal_conv1d
    from triton_ascend_kernels.attention.fla.kda import chunk_kda
    from triton_ascend_kernels.attention.fla.kda.cp import build_cp_context
except ImportError as error:
    causal_conv1d = chunk_kda = build_cp_context = None
    KCP_IMPORT_ERROR = error


def _kcp_runtime():
    """Require the optional KCP kernel set before applying this strategy."""
    if any(kernel is None for kernel in (causal_conv1d, chunk_kda, build_cp_context)):
        raise ImportError(
            "KCP requires `fla` and `triton_ascend_kernels` with KDA context "
            "parallel support. Install the Ascend KCP runtime or use another "
            "context_parallel_strategy."
        ) from KCP_IMPORT_ERROR
    return causal_conv1d, chunk_kda, build_cp_context


def kcp_kda_forward(
    original_forward: Callable[..., torch.Tensor], group: dist.ProcessGroup
) -> Callable[..., torch.Tensor]:
    """Bind model-owned KDA mathematics to a sequence-state parallel context."""

    @wraps(original_forward)
    def forward(
        module: KimiDeltaAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        with synchronize_local_errors("KCP KDA input validation", group=group):
            kernel_size = validate_context_parallel_kda(
                module,
                hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params,
                cu_seqlens=kwargs.get("cu_seqlens"),
            )
        local_sequence = hidden_states.shape[1]
        if get_world_size(group) > 1:
            metadata = torch.tensor(
                [
                    local_sequence,
                    kernel_size,
                    module.num_heads,
                    module.head_k_dim,
                    module.head_dim,
                ],
                dtype=torch.int64,
                device=get_comm_device(group),
            )
            bounds = torch.stack((metadata, -metadata))
            dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=group)
            if bounds[0, 0].item() != -bounds[1, 0].item():
                raise ValueError(
                    "KCP KDA requires equal local sequence lengths on every CP "
                    "rank; repartition the sequence before state communication"
                )
            if not torch.equal(bounds[0, 1:], -bounds[1, 1:]):
                raise ValueError(
                    "KCP KDA requires matching convolution kernel size, head "
                    "count and head dimensions on every CP rank"
                )
        causal_conv1d, chunk_kda, build_cp_context = _kcp_runtime()
        context = build_cp_context(
            cu_seqlens=torch.tensor(
                [0, local_sequence * group.size()],
                dtype=torch.long,
                device=hidden_states.device,
            ),
            group=group,
            conv1d_kernel_size=kernel_size,
        )

        return forward_context_parallel_kda(
            module,
            hidden_states,
            context=context,
            causal_conv1d=causal_conv1d,
            chunk_kda=chunk_kda,
        )

    return forward


def kcp_mla_attention(group):
    """Ulysses for expanded Kimi MLA heads and contiguous causal token shards.

    Adapted from FSDPTurbo 0af4eca distributed/context_parallel/ulysses/
    ulysses_attention.py. Exchange autograd uses the existing CP primitives;
    output returns in the HF [batch, sequence, heads, value_dim] convention.
    """

    def forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        with synchronize_local_errors("KCP MLA local mask validation", group=group):
            batch, heads, local_sequence, _ = query.shape
            if (
                key.ndim != 4
                or value.ndim != 4
                or key.shape != query.shape
                or value.shape[:3] != query.shape[:3]
            ):
                raise ValueError(
                    "KCP MLA requires aligned Q/K/V batch, heads and sequence"
                )
            if local_sequence == 0:
                raise ValueError("KCP MLA requires a nonempty local sequence")
            if heads % group.size():
                raise ValueError("KCP MLA heads must be divisible by the CP degree")
            valid = torch.ones(
                batch, local_sequence, device=query.device, dtype=torch.bool
            )
            # The decoder constructs a local causal mask. Recover padding from its
            # diagonal, then verify the whole mask before rebuilding global causality.
            # Arbitrary bias or document masks must never be silently discarded.
            if attention_mask is not None:
                if attention_mask.ndim != 4 or attention_mask.shape != (
                    batch,
                    1,
                    local_sequence,
                    local_sequence,
                ):
                    raise ValueError("KCP MLA requires a local square causal mask")
                allowed = (
                    attention_mask
                    if attention_mask.dtype == torch.bool
                    else attention_mask == 0
                )
                valid = allowed[:, 0].diagonal(dim1=-2, dim2=-1)
                causal = torch.ones(
                    local_sequence,
                    local_sequence,
                    device=query.device,
                    dtype=torch.bool,
                ).tril()
                expected = causal[None, None] & valid[:, None, None, :]
                if not torch.equal(allowed, expected):
                    raise ValueError("KCP MLA supports causal padding masks only")
                if attention_mask.dtype != torch.bool and not torch.all(
                    (attention_mask == 0) | torch.isneginf(attention_mask)
                ):
                    raise ValueError("KCP MLA cannot discard additive attention bias")
        # Both mask gathering and Ulysses all-to-all require equal payload
        # geometry. Compare fixed-size metadata before exchanging those tensors.
        if get_world_size(group) > 1:
            metadata = torch.tensor(
                [batch, heads, local_sequence, query.shape[-1], value.shape[-1]],
                dtype=torch.int64,
                device=get_comm_device(group),
            )
            bounds = torch.stack((metadata, -metadata))
            dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=group)
            if not torch.equal(bounds[0], -bounds[1]):
                raise ValueError(
                    "KCP MLA requires matching batch, heads, sequence and "
                    "head dimensions on every CP rank"
                )
        masks = [torch.empty_like(valid) for _ in range(group.size())]
        dist.all_gather(masks, valid.contiguous(), group=group)
        valid = torch.cat(masks, dim=-1)
        query, key, value = (SeqToHead.apply(x, group) for x in (query, key, value))
        length = query.shape[-2]
        causal = torch.ones(
            length, length, device=query.device, dtype=torch.bool
        ).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling
        )
        return HeadToSeq.apply(output, group).transpose(1, 2).contiguous(), None

    return forward


def resolve_kcp_plan(model: nn.Module, *, degree: int) -> ParallelPlan:
    """Bind KCP targets without loading kernels or modifying model forwards.

    Explicit models consume declared attention slots. Structural detection is
    retained only for existing raw-module callers. Aliases install once.
    """
    if type(degree) is not int or degree < 1:
        raise ValueError("KCP degree must be a positive integer")
    structure = getattr(model, "decoder_structure", None)
    if structure is not None:
        if not isinstance(structure, DecoderStructure):
            raise TypeError("KCP requires a model-owned DecoderStructure")
        # Preserve first-occurrence order while installing each shared module
        # once. Declaration binding itself retains every decoder slot.
        attentions = tuple(dict.fromkeys(structure.bind_attentions(model)))
        layouts = {
            module: getattr(module, "context_parallel_layout", None)
            for module in attentions
        }
        if any(
            layout not in ("kda_state", "mla_ulysses") for layout in layouts.values()
        ):
            raise ValueError("KCP requires declared Kimi KDA/MLA attention modules")
        modules = [m for m in attentions if layouts[m] == "kda_state"]
        mla_modules = [m for m in attentions if layouts[m] == "mla_ulysses"]
    else:
        required = ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias")
        attentions = tuple(model.modules())
        if any(
            (
                all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))
                or all(
                    hasattr(module, name)
                    for name in ("kv_a_proj_with_mqa", "kv_b_proj")
                )
            )
            and not all(hasattr(module, name) for name in required)
            and not isinstance(module, KimiLatentAttention)
            for module in attentions
        ):
            raise ValueError("KCP does not support hybrid KDA/softmax attention models")
        modules = [
            module
            for module in attentions
            if all(hasattr(module, name) for name in required)
        ]
        mla_modules = [m for m in attentions if isinstance(m, KimiLatentAttention)]
    if not modules and not (structure is not None and mla_modules):
        raise ValueError("KCP selected but no Kimi Delta Attention module was found")
    for module in mla_modules:
        if module.num_heads % degree:
            raise ValueError("KCP MLA heads must be divisible by the CP degree")
    paths = {
        id(module): path for path, module in reversed(tuple(model.named_modules()))
    }
    return ParallelPlan(
        stage="cp",
        degree=degree,
        implementation="kcp",
        targets=tuple(
            PlanTarget(paths[id(module)], "attention", module, layout)
            for layout, targets in (
                ("kda_state", modules),
                ("mla_ulysses", mla_modules),
            )
            for module in targets
        ),
        operations=("bind_kda_context", "bind_mla_exchange"),
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )


def apply_kcp(
    model: nn.Module,
    cp_mesh: DeviceMesh,
    *,
    resolved_plan: ParallelPlan | None = None,
) -> int:
    """Prepare all KCP wrappers, then install them without stacking wrappers.

    Runtime/group resources remain explicit. A pre-resolved plan belongs to the
    current local instance and must be rebound after PP or parameter conversion.
    """
    degree = cp_mesh.size() if cp_mesh is not None else 1
    plan = resolved_plan or resolve_kcp_plan(model, degree=degree)
    if plan.stage != "cp" or plan.implementation != "kcp" or plan.degree != degree:
        raise ValueError("KCP plan does not match the requested execution")
    plan.validate_binding(model)
    # Metadata checks precede optional runtime loading and wrapper creation.
    # A frozen plan does not freeze model-owned role declarations.
    kda_count = 0
    seen: set[int] = set()
    for target in plan.targets:
        module = target.module
        if target.kind != "attention" or module is None:
            raise ValueError("KCP plan requires bound attention targets")
        if id(module) in seen:
            raise ValueError("KCP plan contains duplicate attention targets")
        seen.add(id(module))
        if plan.decoder_structure is not None and (
            getattr(module, "context_parallel_layout", None) != target.layout
        ):
            raise ValueError(f"Stale KCP attention layout: {target.path}")
        if target.layout == "mla_ulysses":
            if module.num_heads % degree:
                raise ValueError("KCP MLA heads must be divisible by the CP degree")
        elif target.layout == "kda_state":
            kda_count += 1
        else:
            raise ValueError(f"Unsupported KCP target layout: {target.layout}")
    if plan.decoder_structure is not None:
        expected = {
            id(module) for module in plan.decoder_structure.bind_attentions(model)
        }
        if seen != expected:
            raise ValueError("KCP plan must cover all declared attention slots")
    if not kda_count and (plan.decoder_structure is None or not seen):
        raise ValueError("KCP plan requires at least one KDA target")
    _kcp_runtime()
    group = cp_mesh.get_group()
    pending = []
    for target in plan.targets:
        module = target.module
        if target.layout == "mla_ulysses":
            pending.append((module, "_kcp_attention", kcp_mla_attention(group)))
        else:
            wrapped = kcp_kda_forward(type(module).forward, group)
            pending.append((module, "forward", wrapped.__get__(module, type(module))))
    for module, attribute, forward in pending:
        setattr(module, attribute, forward)
    return kda_count


__all__ = ["apply_kcp", "kcp_kda_forward", "resolve_kcp_plan"]
