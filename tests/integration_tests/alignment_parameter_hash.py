"""Bounded local parameter hashes in explicit Qwen3 canonical coordinates."""

import hashlib
import json
import re
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard


def canonical_parameter_views(name, local):
    """Yield logical projection views; never concatenate packed expert storage."""
    name = re.sub(r"^part0\.", "", name)
    if re.match(r"^part\d+\.", name):
        raise ValueError("parameter hash requires one model part")
    name = name.replace("._checkpoint_wrapped_module", "")
    name = name.removeprefix("model.model.").removeprefix("model.")
    if any(
        token in name
        for token in (
            "qkv_linear.wqkv",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        )
    ):
        return
    expert = re.fullmatch(
        r"layers\.(\d+)\.(?:moe\.routed_experts\.inner_experts\.(w1_EFD|w2_EDF|w3_EFD)|mlp\.experts\.(gate_up_proj|down_proj))",
        name,
    )
    if expert:
        layer, native, packed = expert.groups()
        if local.ndim != 3:
            raise ValueError("expert hash requires three-dimensional local storage")
        roles = (
            (
                ("gate", local[:, : local.shape[1] // 2]),
                ("up", local[:, local.shape[1] // 2 :]),
            )
            if packed == "gate_up_proj"
            else (
                (
                    {"w1_EFD": "gate", "w2_EDF": "down", "w3_EFD": "up"}.get(
                        native, "down"
                    ),
                    local,
                ),
            )
        )
        if packed == "gate_up_proj" and local.shape[1] % 2:
            raise ValueError("packed expert hash requires even gate/up dimension")
        for role, value in roles:
            yield f"layers.{layer}.experts.{role}", value
        return
    for source, target in (
        ("attention.wo", "attention.o_proj"),
        ("self_attn.o_proj", "attention.o_proj"),
        ("self_attn.q_norm", "attention.q_norm"),
        ("self_attn.k_norm", "attention.k_norm"),
        ("moe.router.gate", "router"),
        ("mlp.gate", "router"),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("embed_tokens", "tok_embeddings"),
    ):
        name = name.replace(source, target)
    if name == "output.weight":
        name = "lm_head.weight"
    yield name, local


def parameter_chunk_hashes(name, parameter, *, chunk_elements=262144):
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    offset = 0
    local = parameter.detach()
    placements = []
    ranks = []
    if isinstance(parameter, DTensor):
        placements = [str(p) for p in parameter.placements]
        ranks = parameter.device_mesh.mesh.tolist()
        # This diagnostic is specifically DP8. Fail before hashing unsupported
        # geometry rather than silently assigning wrong canonical coordinates.
        if parameter.device_mesh.ndim != 1 or len(parameter.placements) != 1:
            raise ValueError("parameter hash requires a one-dimensional DP mesh")
        placement = parameter.placements[0]
        if isinstance(placement, Shard) and placement.dim == 0:
            length, offset = placement._local_shard_size_and_offset(
                parameter.shape[0],
                parameter.device_mesh.size(),
                parameter.device_mesh.get_local_rank(),
            )
            if length != parameter.to_local().shape[0]:
                raise ValueError(
                    "local shard length disagrees with canonical ownership"
                )
        elif not isinstance(placement, Replicate):
            raise ValueError("parameter hash requires Shard(0) or Replicate")
        local = parameter.to_local().detach()
    rows = []
    for canonical, view in canonical_parameter_views(name, local):
        expert = ".experts." in canonical
        outer = range(view.shape[0]) if expert else (None,)
        for expert_index in outer:
            value = view[expert_index] if expert else view
            if value.ndim not in (1, 2):
                raise ValueError(f"unsupported canonical hash shape for {canonical}")
            width = value.shape[1] if value.ndim == 2 else 1
            block_rows = max(1, chunk_elements // max(width, 1))
            for start in range(0, value.shape[0], block_rows):
                end = min(start + block_rows, value.shape[0])
                cpu = value[start:end].contiguous().cpu()
                payload = cpu.view(torch.uint8).numpy().tobytes()
                rows.append(
                    {
                        "canonical": canonical,
                        "expert": offset + expert_index if expert else None,
                        "row_start": start if expert else offset + start,
                        "row_end": end if expert else offset + end,
                        "shape": list(cpu.shape),
                        "dtype": str(cpu.dtype),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "finite": bool(torch.isfinite(cpu).all()),
                    }
                )
    covered = sum(torch.Size(row["shape"]).numel() for row in rows)
    normalized_name = name.replace("._checkpoint_wrapped_module", "")
    skipped_qkv = any(
        token in normalized_name
        for token in (
            "qkv_linear.wqkv",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        )
    )
    if not skipped_qkv and covered != local.numel():
        raise ValueError(f"incomplete local hash coverage for {name}")
    return {
        "name": name,
        "local_numel": local.numel(),
        "covered_numel": covered,
        "excluded_reason": "QKV covered by independent full-local audit"
        if skipped_qkv
        else None,
        "global_shape": list(parameter.shape),
        "local_shape": list(local.shape),
        "placements": placements,
        "mesh_ranks": ranks,
        "chunks": rows,
    }


def save_parameter_hashes(parameters, names, path: Path):
    """Write a JSONL record per parameter with bounded tensor staging."""
    with path.open("x") as stream:
        for parameter in parameters:
            record = parameter_chunk_hashes(names[id(parameter)], parameter)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
