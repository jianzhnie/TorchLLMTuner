"""Qwen3 external expert keys and canonical packed/native projection roles.

Pure format rules shared by checkpoint readers and export adapters. This module
performs no storage IO, collective, tensor allocation or model construction.
"""

import re
from collections.abc import Iterable, Mapping, Sequence

NATIVE_PROJECTIONS = {
    "gate_proj": "w1_EFD",
    "up_proj": "w3_EFD",
    "down_proj": "w2_EDF",
}
HF_PROJECTIONS = {name: projection for projection, name in NATIVE_PROJECTIONS.items()}


def grouped_expert_key(key: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(
        r"(.+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
    )
    if match is None:
        return None
    prefix, expert, projection = match.groups()
    target = "down_proj" if projection == "down_proj" else "gate_up_proj"
    return prefix + target, int(expert), projection


def native_ep_expert_key(key: str) -> str:
    """Map an individual HF projection to its native EP stack FQN."""
    mapped = grouped_expert_key(key)
    if mapped is None:
        return key
    projection = mapped[2]
    role = NATIVE_PROJECTIONS[projection]
    return key.rsplit("experts.", 1)[0] + "routed_experts.inner_experts." + role


def external_expert_projection(key: str) -> tuple[str, str] | None:
    """Return the external expert-list prefix and projection for a native stack.

    Nonexpert keys pass through at the caller. Unknown native projection roles
    are errors rather than a guessed checkpoint layout.
    """
    match = re.fullmatch(
        r"(.+\.mlp\.)routed_experts\.inner_experts\.(w[123]_[A-Z]+)", key
    )
    if match is None:
        return None
    prefix, role = match.groups()
    if role not in HF_PROJECTIONS:
        raise ValueError(f"Unknown Qwen3 expert projection: {key}")
    return prefix + "experts.", HF_PROJECTIONS[role]


def normalize_expert_keys(
    keys: Iterable[str],
    shard_indices: Mapping[str, int] | None,
    *,
    native_ep_layout: bool,
) -> tuple[set[str], dict[str, int] | None]:
    """Normalize external expert keys without reading or copying tensor storage.

    Gate/up and individual expert entries intentionally share a packed target.
    The smallest source shard index retains the existing export grouping rule.
    Inputs stay untouched, including their iteration order and shard mapping.
    """
    grouped_keys: set[str] = set()
    grouped_indices: dict[str, int] = {}
    for key in keys:
        mapped = grouped_expert_key(key)
        target = mapped[0] if mapped is not None else key
        if native_ep_layout:
            target = native_ep_expert_key(key)
        grouped_keys.add(target)
        if shard_indices is not None:
            source_index = shard_indices[key]
            grouped_indices[target] = min(
                grouped_indices.get(target, source_index), source_index
            )
    return grouped_keys, None if shard_indices is None else grouped_indices


def expert_export_keys(
    key: str, shape: Sequence[int] | None, *, num_experts: int
) -> tuple[str, ...] | None:
    """Describe the individual HF keys for a complete native expert stack.

    None denotes nonexpert state. Runtime EP shards must first be gathered by
    the layout adapter; partial stacks cannot describe a complete external file.
    Tensor validation is represented by shape=None, keeping this module free of
    tensor, checkpoint-storage and distributed dependencies.
    """
    projection = external_expert_projection(key)
    if projection is None:
        return None
    if shape is None or len(shape) != 3 or shape[0] != num_experts:
        raise ValueError(f"Qwen3 HF export requires full expert stack: {key}")
    prefix, role = projection
    return tuple(f"{prefix}{index}.{role}.weight" for index in range(num_experts))
