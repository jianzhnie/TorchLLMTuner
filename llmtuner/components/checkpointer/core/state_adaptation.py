"""Shared state-dict adaptation at checkpoint format boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from ....accelerator.collectives import synchronize_local_errors
from .checkpoint_keys import (
    CHECKPOINT_FORMAT,
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    RNG_STATE,
    TRAIN_STATE,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def restore_rng_state(
    states: dict[str, Any],
    loaded: dict[str, Any],
    *,
    phase_prefix: str = "Checkpoint",
) -> None:
    """Validate RNG on every rank before any generator is restored.

    Call after all other restore callbacks and EMA cold start. These callbacks
    must be local; restore failure is coordinated but does not roll back streams.
    """
    with synchronize_local_errors(f"{phase_prefix} RNG validation"):
        rng = states.get(RNG_STATE)
        if rng is not None:
            incoming = loaded[RNG_STATE]
            validate = getattr(rng, "validate_state_dict", None)
            if callable(validate):
                validate(incoming)
    with synchronize_local_errors(f"{phase_prefix} RNG restoration"):
        if rng is not None:
            rng.load_state_dict(incoming)


def restore_categories_before_rng(states: dict[str, Any]) -> tuple[str, ...]:
    """Order explicit restore callbacks without relying on registration order.

    RNG is restored separately after every callback and any EMA cold start.
    Other state retains registration order once model and optimizer are loaded.
    This describes callback dependencies, not a storage backend's IO order.
    """
    return tuple(
        category for category in (MODEL, OPTIMIZER) if category in states
    ) + tuple(
        category for category in states if category not in (MODEL, OPTIMIZER, RNG_STATE)
    )


def materialize_local_checkpoint_states(states: dict[str, Any]) -> dict[str, Any]:
    """Snapshot built-in collective-free states before communicating callbacks.

    The caller coordinates failures around this local operation. Model,
    optimizer, EMA and custom states remain untouched for backend conversion.
    Independent snapshots also keep load readers away from live metadata.
    """
    result = dict(states)
    for category in (
        CHECKPOINT_FORMAT,
        LR_SCHEDULER,
        DATALOADER,
        TRAIN_STATE,
        RNG_STATE,
    ):
        state = result.get(category)
        if isinstance(state, Stateful):
            result[category] = deepcopy(state.state_dict())
    return result


def select_local_preflight_states(states: dict[str, Any]) -> dict[str, Any]:
    """Select ordered collective-free validators without creating read buffers."""
    return {
        category: states[category]
        for category in (
            CHECKPOINT_FORMAT,
            RNG_STATE,
            DATALOADER,
            TRAIN_STATE,
            LR_SCHEDULER,
        )
        if category in states
        and callable(getattr(states[category], "validate_state_dict", None))
    }


def prepare_local_state_preflight(
    states: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select local compatibility validators and isolated read destinations.

    Only format, progress, data, RNG and scheduler metadata participate. Their state
    methods must be collective-free; model weights and optimizer moments stay
    with backend layout adapters. Copy even custom state dictionaries so a
    reader cannot mutate live metadata/tensors before validation succeeds.
    The caller owns cross-rank error coordination around this local operation.
    """
    selected = select_local_preflight_states(states)
    destination = {
        category: deepcopy(state.state_dict()) for category, state in selected.items()
    }
    return selected, destination


def validate_local_state_preflight(
    states: dict[str, Any],
    destination: dict[str, Any],
    *,
    error_group: ProcessGroup | None = None,
) -> None:
    """Validate isolated metadata before loading live training state.

    Every selected category must be present in the loaded destination.
    Every rank must pass the same ordered state categories. Each local validator
    is coordinated separately before the next one runs.
    """
    for category, state in states.items():
        phase = "RNG" if category == RNG_STATE else category
        with synchronize_local_errors(
            f"Checkpoint {phase} preflight validation", group=error_group
        ):
            state.validate_state_dict(destination[category])


def validate_local_preflight_selection(
    states: dict[str, Any], *, error_group: ProcessGroup | None = None
) -> None:
    """Agree on ordered validators before branching into subset readers.

    The selected state categories must agree even when every rank registered
    the same state names: injected services can expose different validation
    capabilities. Metadata values themselves may legitimately differ by rank.
    """
    if not dist.is_initialized() or dist.get_world_size(error_group) <= 1:
        return
    selection = tuple(states)
    selections = [None] * dist.get_world_size(error_group)
    dist.all_gather_object(selections, selection, group=error_group)
    if any(candidate != selection for candidate in selections):
        raise RuntimeError(
            "Checkpoint ranks selected inconsistent local preflight validators; "
            "use matching validation capabilities and state order on every rank"
        )


def prepare_hf_model_state(
    adapter: Any,
    native_state: dict[str, Any],
    *,
    error_group: ProcessGroup | None = None,
) -> dict[str, Any]:
    """Prepare reader destinations before any peer enters the HF reader.

    Built-in adapters coordinate key mapping and coverage internally. Injected
    collective-free adapters get a local error boundary here; adapters doing
    collectives must declare and own their preparation coordination.
    """
    with (
        nullcontext()
        if getattr(adapter, "coordinates_hf_preparation", False)
        else synchronize_local_errors(
            "HF checkpoint destination preparation", group=error_group
        )
    ):
        return adapter.to_hf(native_state)


def adapt_hf_model_state(
    adapter: Any,
    model_state: Any,
    hf_state: dict[str, Any],
    *,
    error_group: ProcessGroup | None = None,
) -> dict[str, Any]:
    """Convert HF keys and validate the native destination before loading.

    Model-owned conversion can perform its own tensor-layout collectives, so it must
    remain outside rank-local error coordination. Other adapters are local and
    must agree on failures before model load callbacks run. The preceding
    reader may already have written into live tensors; this is not a rollback
    boundary. ``error_group`` coordinates conversion/validation failures only;
    storage IO and model-format tensor collectives keep their own ownership.
    """
    with (
        nullcontext()
        if getattr(adapter, "coordinates_state_conversion", False)
        else synchronize_local_errors(
            "HF checkpoint state adaptation", group=error_group
        )
    ):
        native_state = adapter.from_hf(hf_state)

    with synchronize_local_errors("HF checkpoint model validation", group=error_group):
        validate = getattr(model_state, "validate_state_dict", None)
        if callable(validate):
            validate(native_state)
    return native_state
