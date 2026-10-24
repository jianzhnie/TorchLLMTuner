"""Translate full Kimi expert tensors between native EP and HF layouts.

EP collection/slicing belongs to parallel/expert_parallel/checkpointing.py.
These transforms operate on full expert tensors, including meta load templates.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from typing import Any

import torch

from ...accelerator.collectives import synchronize_local_errors

_BLOCK = re.compile(r"^(.*\.block_sparse_moe\.)(.*)$")
_PROJECTIONS = {"w1_EFD": "w1", "w2_EDF": "w2", "w3_EFD": "w3"}
_LATENT = ("routed_expert_down_proj.", "routed_expert_norm.", "routed_expert_up_proj.")


def convert_kimi_expert_state(
    state: dict[str, Any],
    *,
    num_experts: int,
    to_hf: bool,
    coordinate_errors: bool = False,
) -> dict[str, Any]:
    """Map grouped weights and replicated state without dropping keys.

    ``coordinate_errors`` requires all world ranks to participate. It coordinates
    metadata validation only; failures inside tensor communication remain fatal.
    """
    result = {}
    stacks: dict[str, dict[int, torch.Tensor]] = {}
    splits: list[tuple[list[str], torch.Tensor]] = []

    def put(key, value):
        if key in result:
            raise ValueError(f"Kimi checkpoint key mapping collision: {key!r}")
        result[key] = value

    # Validate only keys and tensor metadata before any tensor operation.
    # DTensor unbind/stack below may communicate and must stay outside.
    with (
        synchronize_local_errors("Kimi checkpoint mapping")
        if coordinate_errors
        else nullcontext()
    ):
        for key, value in state.items():
            match = _BLOCK.match(key)
            if match is None:
                put(key, value)
                continue
            prefix, suffix = match.groups()
            if to_hf:
                if suffix.startswith("routed_experts.inner_experts."):
                    role = suffix.removeprefix("routed_experts.inner_experts.")
                    if role in _PROJECTIONS:
                        if not isinstance(value, torch.Tensor) or value.ndim != 3:
                            raise ValueError(
                                f"Kimi grouped weight {key!r} must be rank 3"
                            )
                        if value.shape[0] != num_experts:
                            raise ValueError(
                                f"Kimi grouped weight {key!r} must contain all "
                                f"{num_experts} experts; gather EP shards before export"
                            )
                        keys = [
                            f"{prefix}experts.{index}.{_PROJECTIONS[role]}.weight"
                            for index in range(num_experts)
                        ]
                        for target in keys:
                            put(target, None)
                        splits.append((keys, value))
                        continue
                if suffix == "router.gate.weight":
                    suffix = "gate.weight"
                elif suffix == "expert_bias_E":
                    suffix = "gate.e_score_correction_bias"
                elif suffix.startswith(
                    tuple(f"routed_experts.{name}" for name in _LATENT)
                ):
                    suffix = suffix.removeprefix("routed_experts.")
            else:
                expert = re.fullmatch(r"experts\.(\d+)\.(w[123])\.weight", suffix)
                if expert:
                    index, projection = expert.groups()
                    index = int(index)
                    if not 0 <= index < num_experts:
                        raise ValueError(
                            f"Kimi checkpoint expert index out of range: {key}"
                        )
                    if not isinstance(value, torch.Tensor) or value.ndim != 2:
                        raise ValueError(
                            f"Kimi individual weight {key!r} must be rank 2"
                        )
                    role = next(
                        role
                        for role, name in _PROJECTIONS.items()
                        if name == projection
                    )
                    target = f"{prefix}routed_experts.inner_experts.{role}"
                    experts = stacks.setdefault(target, {})
                    if index in experts:
                        raise ValueError(
                            f"Kimi checkpoint duplicate expert index {index} "
                            f"for {target}"
                        )
                    experts[index] = value
                    continue
                if suffix == "gate.weight":
                    suffix = "router.gate.weight"
                elif suffix == "gate.e_score_correction_bias":
                    suffix = "expert_bias_E"
                elif suffix.startswith(_LATENT):
                    suffix = f"routed_experts.{suffix}"
            put(prefix + suffix, value)
        for target, experts in stacks.items():
            if set(experts) != set(range(num_experts)):
                raise ValueError(
                    f"Kimi checkpoint is missing individual experts for {target}"
                )
            shape = experts[0].shape
            if any(value.shape != shape for value in experts.values()):
                raise ValueError(f"Kimi expert weight shapes disagree for {target}")
            put(target, None)
    for keys, value in splits:
        for target, weight in zip(keys, value.unbind(0), strict=True):
            result[target] = weight
    for target, experts in stacks.items():
        result[target] = torch.stack([experts[index] for index in range(num_experts)])
    return result


def convert_kimi_optimizer_state(
    state: dict[str, Any], *, num_experts: int, to_hf: bool
) -> dict[str, Any]:
    """Translate flat optimizer/EMA state without losing per-parameter metadata.

    Moments/EMA tensors stack along E, but Adam's scalar step and parameter
    group options must agree across individual experts before they can become
    one grouped Parameter. Reject incompatible histories rather than selecting
    one expert's step or hyperparameters. Inputs must contain full EP tensors.
    """
    result: dict[str, Any] = {}
    pending: dict[tuple[str, str], dict[int, Any]] = {}
    reverse = {name: role for role, name in _PROJECTIONS.items()}

    def put(key: str, value: Any) -> None:
        if key in result:
            raise ValueError(f"Kimi optimizer key mapping collision: {key!r}")
        result[key] = value

    for key, value in state.items():
        match = re.fullmatch(r"(state|param_groups)\.(.*\.block_sparse_moe\.)(.*)", key)
        if match is None:
            put(key, value)
            continue
        category, prefix, suffix = match.groups()
        if to_hf:
            expert = re.fullmatch(
                r"routed_experts\.inner_experts\.(w[123]_[A-Z]+)\.(.+)", suffix
            )
            if expert and expert[1] in _PROJECTIONS:
                role, field = expert.groups()
                tensor_state = (
                    category == "state"
                    and isinstance(value, torch.Tensor)
                    and value.ndim > 0
                )
                if tensor_state and (value.ndim != 3 or value.shape[0] != num_experts):
                    raise ValueError(
                        f"Kimi optimizer tensor {key!r} requires full (E,O,I) state"
                    )
                for index in range(num_experts):
                    put(
                        f"{category}.{prefix}experts.{index}.{_PROJECTIONS[role]}.weight.{field}",
                        value[index]
                        if tensor_state
                        else value.clone()
                        if isinstance(value, torch.Tensor)
                        else value,
                    )
                continue
            suffix = suffix.replace("router.gate.weight.", "gate.weight.", 1)
            if suffix.startswith("expert_bias_E."):
                suffix = "gate.e_score_correction_bias." + suffix.removeprefix(
                    "expert_bias_E."
                )
            if suffix.startswith(tuple(f"routed_experts.{name}" for name in _LATENT)):
                suffix = suffix.removeprefix("routed_experts.")
        else:
            expert = re.fullmatch(r"experts\.(\d+)\.(w[123])\.weight\.(.+)", suffix)
            if expert:
                index, projection, field = expert.groups()
                index = int(index)
                if not 0 <= index < num_experts:
                    raise ValueError(f"Kimi optimizer expert index out of range: {key}")
                target = (
                    f"{category}.{prefix}routed_experts.inner_experts."
                    f"{reverse[projection]}.{field}"
                )
                values = pending.setdefault((target, category), {})
                if index in values:
                    raise ValueError(
                        f"Kimi optimizer duplicate expert index {index} for {target}"
                    )
                values[index] = value
                continue
            suffix = suffix.replace("gate.weight.", "router.gate.weight.", 1)
            if suffix.startswith("gate.e_score_correction_bias."):
                suffix = "expert_bias_E." + suffix.removeprefix(
                    "gate.e_score_correction_bias."
                )
            if suffix.startswith(_LATENT):
                suffix = f"routed_experts.{suffix}"
        put(f"{category}.{prefix}{suffix}", value)
    for (target, category), values in pending.items():
        if set(values) != set(range(num_experts)):
            raise ValueError(
                f"Kimi optimizer state missing individual experts for {target}"
            )
        ordered = [values[index] for index in range(num_experts)]
        first = ordered[0]
        if category == "state" and isinstance(first, torch.Tensor) and first.ndim > 0:
            if any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape != first.shape
                or value.dtype != first.dtype
                for value in ordered
            ):
                raise ValueError(f"Kimi expert moment tensors disagree for {target}")
            put(target, torch.stack(ordered))
        else:
            for value in ordered[1:]:
                same = (
                    isinstance(value, torch.Tensor)
                    and value.shape == first.shape
                    and value.dtype == first.dtype
                    and torch.equal(value, first)
                    if isinstance(first, torch.Tensor)
                    else type(value) is type(first) and value == first
                )
                if not same:
                    raise ValueError(
                        "Kimi expert scalar state or parameter groups disagree "
                        f"for {target}"
                    )
            # The caller may continue training the source runtime. Its scalar
            # step tensor must not become the destination optimizer's counter.
            put(target, first.clone() if isinstance(first, torch.Tensor) else first)
    return result
