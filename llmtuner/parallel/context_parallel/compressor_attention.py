# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Ulysses attention over sliding KV and rank-owned HCA/CSA entries.

The caller projects local queries/sliding KV and supplies compressed windows.
Outputs follow the HF [batch, local_sequence, heads, dim] layout.
"""

from __future__ import annotations

from functools import partial
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from ...accelerator.collectives import synchronize_local_errors
from ...config.parallel_config import ParallelConfig
from ...models.contracts import DecoderStructure, registered_submodule
from ..plan import ParallelPlan, PlanTarget
from .attention import HeadToSeq, SeqToHead
from .compression_windows import (
    CompressionWindowExchange,
    compress_csa_windows,
    compress_hca_windows,
    select_csa_windows,
)

try:
    from ...models.deepseek.deepseek_v4.context_parallel import (
        compressor_attention_math,
    )
    from ...models.deepseek.deepseek_v4.context_parallel import (
        compressor_decoder_forward as compressor_decoder_forward,
    )
except ImportError as error:
    compressor_attention_math = compressor_decoder_forward = None
    DEEPSEEK_IMPORT_ERROR = error


def compressor_ulysses_attention(
    query,
    sliding_kv,
    compressed_kv,
    *,
    sinks,
    compress_rate,
    sliding_window,
    scaling,
    group,
    dropout=0.0,
    training=True,
    selected_windows=None,
):
    """Attend with a per-head sink and globally indexed compressed causality.

    Query is BHSD; both KV inputs are shared single-head BHSD. Complete Compressor
    windows are owned by their final token's rank, with unequal counts allowed.
    Padding, CSA indexing and KV cache require separate model-level contracts.
    """
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    with synchronize_local_errors("Compressor attention input validation", group=group):
        if query.dtype not in dtypes:
            raise ValueError(
                "Compressor attention requires floating point query tensors"
            )
        for value in (sliding_kv, compressed_kv):
            if value.dtype != query.dtype or value.device != query.device:
                raise ValueError("Compressor query and KV must share dtype and device")
        if sinks.device != query.device:
            raise ValueError("Compressor sinks must be on the query device")
        batch, heads, length, width = query.shape
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if heads % size or length == 0:
            raise ValueError(
                "Compressor Ulysses requires divisible heads and nonempty shards"
            )
        if (compress_rate is not None and compress_rate <= 0) or sliding_window <= 0:
            raise ValueError(
                "Compressor compression rate and sliding window must be positive"
            )
        first = rank * length // compress_rate if compress_rate is not None else 0
        last = (rank + 1) * length // compress_rate if compress_rate is not None else 0
        if sliding_kv.shape != (batch, 1, length, width):
            raise ValueError("Compressor sliding KV must be shared [B, 1, local_S, D]")
        if compressed_kv.shape != (batch, 1, last - first, width):
            raise ValueError(
                "Compressor compressed KV does not match global window ownership"
            )
        if sinks.shape != (heads,):
            raise ValueError("Compressor sinks must contain one value per query head")
        if selected_windows is not None:
            total = length * size // compress_rate if compress_rate is not None else 0
            if (
                selected_windows.ndim != 3
                or selected_windows.shape[:2] != (batch, length)
                or selected_windows.dtype != torch.long
                or selected_windows.device != query.device
                or torch.any(selected_windows < -1)
                or torch.any(selected_windows >= total)
            ):
                raise ValueError(
                    "CSA selection must contain local global-window indices"
                )
    metadata = torch.tensor(
        [
            batch,
            heads,
            length,
            width,
            0 if compress_rate is None else compress_rate,
            sliding_window,
            dtypes.index(query.dtype),
            -1 if selected_windows is None else selected_windows.shape[-1],
        ],
        device=query.device,
        dtype=torch.long,
    )
    peers = [torch.empty_like(metadata) for _ in range(size)]
    dist.all_gather(peers, metadata, group=group)
    if any(not torch.equal(peer, metadata) for peer in peers):
        raise ValueError(
            "Compressor attention requires matching CP shapes and window settings"
        )
    local_heads = heads // size
    query = SeqToHead.apply(query, group)
    sliding = SeqToHead.apply(sliding_kv.expand(-1, heads, -1, -1).contiguous(), group)
    # Every peer receives this owner's entries for its query heads. Shared KV is
    # transmitted once per destination, with backward summing all head consumers.
    owned = compressed_kv[:, 0].transpose(0, 1).contiguous()
    send = owned.repeat(size, 1, 1)
    counts = [
        ((peer + 1) * length // compress_rate - peer * length // compress_rate)
        if compress_rate is not None
        else 0
        for peer in range(size)
    ]
    received = CompressionWindowExchange.apply(
        send, [last - first] * size, counts, group
    )
    compressed = received.transpose(0, 1).unsqueeze(1)
    full_length = length * size
    positions = torch.arange(full_length, device=query.device)
    sliding_allowed = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - sliding_window
    )
    indices = torch.arange(compressed.shape[2], device=query.device)
    compressed_allowed = (
        indices[None, :] < (positions[:, None] + 1) // compress_rate
        if compress_rate is not None
        else torch.zeros(full_length, 0, device=query.device, dtype=torch.bool)
    )
    if selected_windows is not None:
        # Selection is discrete metadata, shared across heads but query-sharded.
        selections = [torch.empty_like(selected_windows) for _ in range(size)]
        dist.all_gather(selections, selected_windows.contiguous(), group=group)
        selected = torch.cat(selections, dim=1)
        visible = torch.zeros(
            batch,
            full_length,
            compressed.shape[2] + 1,
            device=query.device,
            dtype=torch.bool,
        )
        safe = torch.where(selected >= 0, selected, compressed.shape[2])
        visible.scatter_(-1, safe, True)
        compressed_allowed = compressed_allowed[None] & visible[..., :-1]
    else:
        compressed_allowed = compressed_allowed[None].expand(batch, -1, -1)
    allowed = torch.cat(
        (sliding_allowed[None].expand(batch, -1, -1), compressed_allowed), dim=-1
    )
    kv = torch.cat((sliding, compressed.expand(-1, local_heads, -1, -1)), dim=2)
    logits = (query @ kv.transpose(-1, -2)) * scaling
    logits = logits.masked_fill(~allowed[:, None], float("-inf"))
    sink = sinks[rank * local_heads : (rank + 1) * local_heads]
    sink = sink[None, :, None, None].expand(batch, -1, full_length, 1)
    # Sink contributes to the softmax denominator but has zero value.
    logits = torch.cat((logits, sink), dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    weights = logits.softmax(dim=-1)[..., :-1]
    weights = F.dropout(weights, p=dropout, training=training)
    output = weights.to(kv.dtype) @ kv
    return HeadToSeq.apply(output, group).transpose(1, 2).contiguous()


def compressor_attention_forward(group):
    """Build an instance forward for stateless DeepSeek V4 compressor attention.

    Projection/rotary/output math follows Transformers 5.9.0
    modeling_deepseek_v4.py (Apache-2.0). This factory does not install itself;
    the model integration must enforce contiguous inputs and supported layers.
    """

    def forward(
        module,
        hidden_states,
        position_embeddings,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if compressor_attention_math is None:
            raise ImportError(
                "DeepSeek V4 CP requires a compatible transformers build"
            ) from DEEPSEEK_IMPORT_ERROR
        with synchronize_local_errors(
            "Compressor module input validation", group=group
        ):
            if module.layer_type not in (
                "heavily_compressed_attention",
                "compressed_sparse_attention",
                "sliding_attention",
            ):
                raise ValueError("Compressor CP forward requires HCA or CSA layers")
            if past_key_values is not None or attention_mask is not None:
                raise ValueError(
                    "Compressor CP constructs its own masks and rejects KV cache"
                )
            batch, length, _ = hidden_states.shape
            expected = torch.arange(length, device=hidden_states.device)
            expected = expected + dist.get_rank(group) * length
            if position_ids.shape != (batch, length) or not torch.equal(
                position_ids, expected[None].expand(batch, -1)
            ):
                raise ValueError(
                    "Compressor CP requires contiguous global position_ids"
                )
        return compressor_attention_math(
            module,
            hidden_states,
            position_embeddings,
            position_ids,
            compress_csa_windows=partial(compress_csa_windows, group=group),
            select_csa_windows=partial(select_csa_windows, group=group),
            compress_hca_windows=partial(compress_hca_windows, group=group),
            attention=partial(compressor_ulysses_attention, group=group),
        )

    return forward


def resolve_compressor_plan(
    model: torch.nn.Module, *, degree: int, cfg: ParallelConfig
) -> ParallelPlan:
    """Resolve stateless Compressor targets without changing model ownership."""
    if type(degree) is not int or degree < 1:
        raise ValueError("Compressor CP degree must be a positive integer")
    if not callable(getattr(model, "set_cp_mesh", None)):
        raise TypeError("Compressor CP requires model-owned set_cp_mesh semantics")
    structure = getattr(model, "decoder_structure", None)
    if not isinstance(structure, DecoderStructure):
        raise TypeError("Compressor CP requires an explicit decoder structure")
    decoder = registered_submodule(model, structure.decoder)
    if not hasattr(decoder, "layers"):
        raise TypeError(
            f"Declared compressor decoder {structure.decoder!r} must expose layers"
        )
    attentions = tuple(dict.fromkeys(structure.bind_attentions(model)))
    if cfg.cp_config.context_parallel_strategy != "ulysses":
        raise ValueError(
            "DeepSeek V4 Compressor CP currently requires strategy='ulysses'"
        )
    if cfg.tp_size != 1 or cfg.pp_size != 1:
        raise ValueError(
            "DeepSeek V4 Compressor CP TP/PP composition is not implemented"
        )
    if cfg.cp_config.context_parallel_load_balancer is not None:
        raise ValueError(
            "Compressor CP requires contiguous shards with no load balancer"
        )
    if getattr(decoder.config, "attn_mask_type", "causal") != "causal":
        raise ValueError("Compressor CP does not support packed document attention")
    if not attentions or any(
        getattr(attention, "layer_type", None)
        not in (
            "heavily_compressed_attention",
            "compressed_sparse_attention",
            "sliding_attention",
        )
        for attention in attentions
    ):
        raise ValueError("Compressor CP encountered an unsupported attention layer")
    for attention in attentions:
        if attention.num_heads % degree:
            raise ValueError(
                "Compressor CP requires query heads divisible by the CP degree"
            )
        if attention.attention_dropout != 0:
            raise ValueError(
                "Compressor CP dropout random-stream equivalence is not implemented"
            )
    paths = {
        id(module): path for path, module in reversed(tuple(model.named_modules()))
    }
    return ParallelPlan(
        stage="cp",
        degree=degree,
        implementation="compressor",
        targets=(PlanTarget(structure.decoder, "decoder", decoder, "stateless"),)
        + tuple(
            PlanTarget(paths[id(module)], "attention", module, module.layer_type)
            for module in attentions
        ),
        operations=("bind_window_exchange", "install_stateless_decoder"),
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )


def apply_compressor_context_parallel(
    model: torch.nn.Module,
    *,
    cp_mesh: DeviceMesh,
    cfg: ParallelConfig,
    resolved_plan: ParallelPlan | None = None,
) -> torch.nn.Module:
    """Consume current bindings and prepare all forwards before installation.

    Standalone callers may omit the plan. Assembly resolves after TP/EP so the
    snapshot describes the modules and parameters that will actually execute.
    """
    degree = cp_mesh.size()
    plan = (
        resolve_compressor_plan(model, degree=degree, cfg=cfg)
        if resolved_plan is None
        else resolved_plan
    )
    if (
        plan.stage != "cp"
        or plan.implementation != "compressor"
        or plan.degree != degree
    ):
        raise ValueError("Compressor CP plan does not match the execution mesh")
    plan.validate_binding(model)
    if (
        len(plan.targets) < 2
        or plan.targets[0].kind != "decoder"
        or plan.targets[0].layout != "stateless"
        or any(
            target.kind != "attention"
            or target.layout != getattr(target.module, "layer_type", None)
            for target in plan.targets[1:]
        )
    ):
        raise ValueError("Compressor CP plan has invalid decoder or attention targets")
    attention_ids = [id(target.module) for target in plan.targets[1:]]
    if len(set(attention_ids)) != len(attention_ids):
        raise ValueError("Compressor CP plan contains duplicate attention targets")
    if plan.decoder_structure is not None:
        structure = plan.decoder_structure
        expected = {id(module) for module in structure.bind_attentions(model)}
        if (
            set(attention_ids) != expected
            or plan.targets[0].path != structure.decoder
            or plan.targets[0].module
            is not registered_submodule(model, structure.decoder)
        ):
            raise ValueError("Compressor CP plan must cover its declared decoder slots")
    decoder = plan.targets[0].module
    attentions = [target.module for target in plan.targets[1:]]
    if compressor_attention_math is None or compressor_decoder_forward is None:
        raise ImportError(
            "DeepSeek V4 CP requires a compatible transformers build"
        ) from DEEPSEEK_IMPORT_ERROR
    group = cp_mesh.get_group()
    pending = [
        (
            attention,
            MethodType(compressor_attention_forward(group), attention),
        )
        for attention in attentions
    ]
    decoder_forward = MethodType(compressor_decoder_forward, decoder)
    for attention, forward in pending:
        attention.forward = forward
    decoder.forward = decoder_forward
    model.set_cp_mesh(cp_mesh, strategy="compressor_ulysses")
    return model
