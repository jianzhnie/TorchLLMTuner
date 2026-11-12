"""Qwen3 external expert keys and canonical packed/native projection roles.

Pure format rules shared by checkpoint readers and export adapters. This module
performs no storage IO, collective, tensor allocation or model construction.
"""

import re

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
