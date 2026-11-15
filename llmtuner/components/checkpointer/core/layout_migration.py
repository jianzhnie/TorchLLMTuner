"""Restore native checkpoints using model-declared source layout migrations.

This explicit migration reads full CPU tensors before conversion. Ordinary
same-layout recovery keeps the sharded DCP path and its memory behavior.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, distribute_tensor

from ....accelerator.collectives import synchronize_local_errors
from ....models.state import SourceLayoutMigration
from ....utils.module_names import canonical_fqn
from ...optimizer.state_dict import validate_full_optimizer_state
from .checkpoint_keys import (
    CHECKPOINT_FORMAT,
    DATALOADER,
    EMA,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    RNG_STATE,
    TRAIN_STATE,
)
from .state_adaptation import (
    restore_categories_before_rng,
    restore_rng_state,
    select_local_preflight_states,
    validate_local_preflight_selection,
    validate_local_state_preflight,
)


class SourceLayoutLoadPlanner(DefaultLoadPlanner):
    """Write scalar byte records back to the flat source dictionary."""

    def __init__(self, destination):
        super().__init__(flatten_state_dict=False)
        self.destination = destination

    def load_bytes(self, read_item, value):
        loaded = torch.load(value, weights_only=False)
        self.destination[read_item.dest_index.fqn] = loaded
        self.original_state_dict[read_item.dest_index.fqn] = loaded


def source_destinations(
    metadata: Any, states: dict[str, Any], *, layout_name: str = "Source"
) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    name = layout_name
    paths = metadata.planner_data
    if not isinstance(paths, dict):
        raise ValueError(f"{name} layout migration requires checkpoint planner paths")
    flat = {}
    selected = {}
    destinations = {}
    training_categories = {
        CHECKPOINT_FORMAT,
        OPTIMIZER,
        EMA,
        LR_SCHEDULER,
        DATALOADER,
        TRAIN_STATE,
        RNG_STATE,
    }
    for key, entry in metadata.state_dict_metadata.items():
        path = paths[key]
        # Preserve storage FQNs for DCP reads, but reconstruct the stable
        # model/optimizer/EMA paths expected by the layout converters.
        root = canonical_fqn(str(path[0]), wrapper_segments=("_orig_mod",))
        # DCP stores model state as flat tensor FQNs, alongside nested training
        # categories. Explicit models also own roots such as lm_head.weight;
        # the historical wrapper's model.* prefix is not a model contract.
        model_tensor = (
            len(path) == 1
            and isinstance(entry, TensorStorageMetadata)
            and root not in training_categories
        )
        model_extra_state = root == "_extra_state" or root.endswith("._extra_state")
        if model_extra_state:
            # Extra state can flatten into nested dictionaries or lists. Only
            # its module FQN contains wrapper segments; payload keys are data.
            path = (root, *path[1:])
        elif root in (OPTIMIZER, EMA) or model_tensor:
            path = tuple(
                canonical_fqn(segment, wrapper_segments=("_orig_mod",))
                if isinstance(segment, str)
                else segment
                for segment in path
            )
        if path[0] in {OPTIMIZER, EMA} and path[0] not in states:
            continue
        if path[0] in states or model_tensor or model_extra_state:
            destination = tuple(path)
            if destination in destinations:
                raise ValueError(
                    f"Ambiguous compiled {name} checkpoint keys: "
                    f"{destinations[destination]!r} and {key!r}"
                )
            destinations[destination] = key
            selected[key] = path
            flat[key] = (
                torch.empty(entry.size, dtype=entry.properties.dtype, device="cpu")
                if isinstance(entry, TensorStorageMetadata)
                else None
            )
    return flat, selected


def _agree_migration_choice(choice: object) -> None:
    """Require all restore participants to select the same migration branch."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    choices = [None] * dist.get_world_size()
    dist.all_gather_object(choices, choice)
    if any(candidate != choice for candidate in choices):
        raise ValueError(
            "Checkpoint ranks selected inconsistent source layout migration"
        )


def load_changed_source_layout(
    checkpoint_id: str,
    *,
    states: dict[str, Any],
    adapter: Any,
    ema_to_reset: Any | None = None,
) -> bool:
    """Apply full-source conversion, returning False for ordinary loads.

    The caller may supply an excluded EMA owner to reseed from restored model
    weights. Its local reset runs before RNG restoration and only on migration.
    """
    with synchronize_local_errors("Checkpoint source migration selection"):
        state_format = getattr(adapter, "state_format", None)
        resolve = getattr(state_format, "source_layout_migration", None)
        migration = resolve() if callable(resolve) else None
        if migration is not None and not isinstance(migration, SourceLayoutMigration):
            raise TypeError("Model format must declare a SourceLayoutMigration or None")
    _agree_migration_choice(None if migration is None else migration.name)
    if migration is None:
        return False
    name = migration.name
    with synchronize_local_errors(f"{name} checkpoint source layout"):
        reader = dcp.FileSystemReader(checkpoint_id)
        metadata = reader.read_metadata()
        names = tuple(
            canonical_fqn(key, wrapper_segments=("_orig_mod",))
            for key in metadata.state_dict_metadata
        )
        changed = migration.requires_migration(names)
        if type(changed) is not bool:
            raise TypeError("Source migration selection must return a boolean")
    _agree_migration_choice(changed)
    if not changed:
        return False
    # Build source destinations, not live target-shaped buffers. DCP otherwise
    # plans reads against nonexistent target FQNs and cannot migrate layouts.
    with synchronize_local_errors(f"{name} checkpoint migration allocation"):
        flat, selected = source_destinations(metadata, states, layout_name=name)
    dcp.load(
        flat,
        storage_reader=reader,
        planner=SourceLayoutLoadPlanner(flat),
    )
    with synchronize_local_errors(f"{name} checkpoint source records"):
        loaded = {}
        for key, path in selected.items():
            set_element(loaded, path, flat[key])
        missing_categories = set(states) - {MODEL} - loaded.keys()
        if missing_categories:
            raise ValueError(
                f"{name} layout migration missing training state categories: "
                f"{sorted(missing_categories)}"
            )

    # Direct callers need the same metadata preflight as backend-managed loads.
    # Do not create fresh reader buffers: the isolated source records are ready.
    with synchronize_local_errors(f"{name} checkpoint metadata validator selection"):
        preflight_states = select_local_preflight_states(states)
    validate_local_preflight_selection(preflight_states)
    validate_local_state_preflight(
        preflight_states,
        loaded,
    )

    # These templates may gather EP/TP tensors. Keep their collectives outside
    # the local-only error boundary and reuse their full logical shapes below.
    templates = {
        category: states[category].state_dict()
        for category in (OPTIMIZER, EMA)
        if category in states
    }
    with synchronize_local_errors(f"{name} checkpoint layout conversion"):
        model = {key: value for key, value in loaded.items() if key not in states}
        model = migration.convert_model(model)
        for category in templates:
            loaded[category] = migration.convert_optimizer(loaded[category])
        # Reject malformed full model state before any distribute_tensor
        # collective or parameter copy. All ranks participate in this local
        # validation boundary, including those owning no offending parameter.
        states[MODEL].validate_state_dict(model)
        for category, template in templates.items():
            validate_full_optimizer_state(
                loaded[category], template, context=f"{name} {category}"
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
    for category in restore_categories_before_rng(states):
        # Only built-in metadata callbacks are local. Model, optimizer, EMA
        # and custom states may communicate and keep their own boundaries.
        with (
            synchronize_local_errors(f"{name} checkpoint {category} restoration")
            if category in (LR_SCHEDULER, DATALOADER, TRAIN_STATE)
            else nullcontext()
        ):
            states[category].load_state_dict(
                model if category == MODEL else loaded[category]
            )
    with synchronize_local_errors("Checkpoint EMA cold start"):
        if ema_to_reset is not None:
            ema_to_reset.reset_from_model()
    restore_rng_state(
        states,
        loaded,
        phase_prefix=f"{name} checkpoint",
    )
    return True
