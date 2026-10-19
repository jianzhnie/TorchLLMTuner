"""Load Kimi checkpoints whose individual/grouped expert layout changed.

This explicit migration reads full CPU tensors before conversion. Ordinary
same-layout recovery keeps the sharded DCP path and its memory behavior.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, distribute_tensor

from ...accelerator.collectives import synchronize_local_errors
from ...models.kimi.checkpoint import (
    convert_kimi_expert_state,
    convert_kimi_optimizer_state,
)
from ...utils.logging import get_logger
from .checkpoint_keys import (
    DATALOADER,
    EMA,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    RNG_STATE,
    TRAIN_STATE,
)

logger = get_logger(__name__)


class KimiSourceLoadPlanner(DefaultLoadPlanner):
    """Write scalar byte records back to the flat source dictionary."""

    def __init__(self, destination):
        super().__init__(flatten_state_dict=False)
        self.destination = destination

    def load_bytes(self, read_item, value):
        loaded = torch.load(value, weights_only=False)
        self.destination[read_item.dest_index.fqn] = loaded
        self.original_state_dict[read_item.dest_index.fqn] = loaded


def load_changed_kimi_layout(
    checkpoint_id: str, *, states: dict[str, Any], adapter: Any
) -> bool:
    """Load changed Kimi expert layouts, returning False for ordinary loads."""
    if (
        getattr(getattr(adapter, "model_config", None), "model_type", None)
        != "kimi_linear"
    ):
        return False
    reader = dcp.FileSystemReader(checkpoint_id)
    with synchronize_local_errors("Kimi checkpoint source layout"):
        metadata = reader.read_metadata()
        keys = metadata.state_dict_metadata
        grouped = any(
            ".block_sparse_moe.routed_experts.inner_experts." in key for key in keys
        )
        individual = any(".block_sparse_moe.experts." in key for key in keys)
        if grouped and individual:
            raise ValueError(
                "Kimi checkpoint mixes individual and grouped expert layouts"
            )
        target_grouped = adapter.kimi_expert_parallel
        changed = (grouped or individual) and grouped != target_grouped
    if not changed:
        return False
    # Build source destinations, not live target-shaped buffers. DCP otherwise
    # plans reads against nonexistent target FQNs and cannot migrate layouts.
    with synchronize_local_errors("Kimi checkpoint migration allocation"):
        flat = {}
        paths = metadata.planner_data
        if not isinstance(paths, dict):
            raise ValueError("Kimi layout migration requires checkpoint planner paths")
        selected = {}
        for key, entry in keys.items():
            path = paths[key]
            if path[0] in {OPTIMIZER, EMA} and path[0] not in states:
                continue
            if path[0] in states or str(path[0]).startswith("model."):
                selected[key] = path
                flat[key] = (
                    torch.empty(entry.size, dtype=entry.properties.dtype, device="cpu")
                    if isinstance(entry, TensorStorageMetadata)
                    else None
                )
    dcp.load(
        flat,
        storage_reader=reader,
        planner=KimiSourceLoadPlanner(flat),
    )
    # These templates may gather EP/TP tensors. Keep their collectives outside
    # the local-only error boundary and reuse their full logical shapes below.
    templates = {
        category: states[category].state_dict()
        for category in (OPTIMIZER, EMA)
        if category in states
    }
    with synchronize_local_errors("Kimi checkpoint layout conversion"):
        loaded = {}
        for key, path in selected.items():
            set_element(loaded, path, flat[key])
        model = {key: value for key, value in loaded.items() if key not in states}
        model = convert_kimi_expert_state(
            model,
            num_experts=adapter.model_config.num_experts,
            to_hf=not target_grouped,
        )
        for category in (OPTIMIZER, EMA):
            if category in states:
                if category not in loaded:
                    raise ValueError(
                        f"Kimi checkpoint missing required {category} state"
                    )
                loaded[category] = convert_kimi_optimizer_state(
                    loaded[category],
                    num_experts=adapter.model_config.num_experts,
                    to_hf=not target_grouped,
                )
        missing_categories = set(states) - {MODEL} - loaded.keys()
        if RNG_STATE in missing_categories:
            # Match the ordinary DCP loader's legacy contract. An existing but
            # malformed RNG category still goes through strict validation.
            missing_categories.remove(RNG_STATE)
            logger.warning(
                "Legacy checkpoint has no RNG state; random training "
                "cannot resume with exact stream continuity."
            )
        if missing_categories:
            raise ValueError(
                "Kimi layout migration missing training state categories: "
                f"{sorted(missing_categories)}"
            )
        if RNG_STATE in states and RNG_STATE in loaded:
            states[RNG_STATE].validate_state_dict(loaded[RNG_STATE])
        # Reject malformed full model state before any distribute_tensor
        # collective or parameter copy. All ranks participate in this local
        # validation boundary, including those owning no offending parameter.
        states[MODEL].validate_state_dict(model)
        for category, template in templates.items():
            values = loaded[category]
            expected_values = dict(template)
            for key, expected in template.items():
                if key.startswith("state.") and key.endswith(".exp_avg_sq"):
                    fqn = key[len("state.") : -len(".exp_avg_sq")]
                    if values.get(f"param_groups.{fqn}.amsgrad", False):
                        expected_values[f"state.{fqn}.max_exp_avg_sq"] = expected
            for key, expected in expected_values.items():
                # AMSGrad is selected by the saved group, not the fresh runtime.
                if key.endswith(".max_exp_avg_sq"):
                    fqn = key[len("state.") : -len(".max_exp_avg_sq")]
                    if not values.get(f"param_groups.{fqn}.amsgrad", False):
                        continue
                if key not in values:
                    raise ValueError(f"Kimi {category} missing required key {key!r}")
                value = values[key]
                if isinstance(expected, torch.Tensor):
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(f"Kimi {category} {key!r} must be a tensor")
                    if value.shape != expected.shape:
                        raise ValueError(
                            f"Kimi {category} {key!r} has shape {tuple(value.shape)}, "
                            f"expected {tuple(expected.shape)}"
                        )

    # Model first, optimizer before scheduler, RNG last. Each object's existing
    # validator remains authoritative for keys, topology and shape contracts.
    # DCP normally fills live DTensor destinations directly. Source-layout
    # migration instead read ordinary full tensors, so rebuild each dense
    # destination's mesh/placements before its copy. EP expert slicing stays
    # owned by the existing state wrappers.
    def place(values, template):
        result = dict(values)
        for key, expected in template.items():
            value = result.get(key)
            if isinstance(expected, DTensor) and isinstance(value, torch.Tensor):
                result[key] = distribute_tensor(
                    value.to(expected.device), expected.device_mesh, expected.placements
                )
        return result

    model = place(model, states[MODEL].state_dict())
    for category in (OPTIMIZER, EMA):
        if category in states:
            loaded[category] = place(loaded[category], templates[category])
    states[MODEL].load_state_dict(model)
    if OPTIMIZER in states:
        states[OPTIMIZER].load_state_dict(loaded[OPTIMIZER])
    for category, state in states.items():
        if category in (MODEL, OPTIMIZER, RNG_STATE):
            continue
        if category not in loaded:
            raise ValueError(f"Kimi checkpoint missing required {category} state")
        # These callbacks are local. Coordinate failures before peers can
        # enter EMA or RNG collectives; EMA's own EP/TP communication must
        # remain outside this local-only boundary.
        with (
            synchronize_local_errors(f"Kimi checkpoint {category} restoration")
            if category in (LR_SCHEDULER, DATALOADER, TRAIN_STATE)
            else nullcontext()
        ):
            state.load_state_dict(loaded[category])
    with synchronize_local_errors("Kimi checkpoint RNG restoration"):
        if RNG_STATE in states and RNG_STATE in loaded:
            states[RNG_STATE].load_state_dict(loaded[RNG_STATE])
    return True
