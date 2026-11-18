"""Shared state-dict adaptation at checkpoint format boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from ....accelerator.collectives import synchronize_local_errors
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

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def finish_checkpoint_restore(
    registered: dict[str, Any],
    selected: dict[str, Any],
    loaded: dict[str, Any],
    *,
    restore_rng: bool,
) -> None:
    """Cold-start excluded EMA after model load, then restore RNG last.

    EMA reset only copies local tensors; coordinate failures before any peer
    restores generators or returns to training. HF model imports omit RNG.
    """
    with synchronize_local_errors("Checkpoint EMA cold start"):
        if MODEL in selected and EMA in registered and EMA not in selected:
            registered[EMA].reset_from_model()
    if restore_rng:
        restore_rng_state(selected, loaded)


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


def restore_checkpoint_callbacks(
    states: dict[str, Any],
    destination: dict[str, Any],
    *,
    loaded: dict[str, Any] | None = None,
    phase_prefix: str = "Checkpoint",
) -> None:
    """Restore dependencies before RNG; coordinate only local metadata callbacks.

    Collective-bearing model, optimizer, EMA and custom callbacks own their
    error boundaries. Non-Stateful entries must retain the reader's identity.
    RNG and excluded EMA are handled by the caller after these callbacks.
    """
    loaded = destination if loaded is None else loaded
    # Snapshot callback order: model, optimizer, then other registration order.
    # RNG runs separately after every callback and excluded EMA cold start.
    categories = tuple(
        category for category in (MODEL, OPTIMIZER) if category in states
    ) + tuple(
        category for category in states if category not in (MODEL, OPTIMIZER, RNG_STATE)
    )
    for category in categories:
        target = states[category]
        if isinstance(target, Stateful):
            with (
                synchronize_local_errors(f"{phase_prefix} {category} restoration")
                if category
                in (CHECKPOINT_FORMAT, LR_SCHEDULER, DATALOADER, TRAIN_STATE)
                else nullcontext()
            ):
                target.load_state_dict(destination[category])
        elif loaded[category] is not target:
            raise TypeError(
                f"Cannot restore non-Stateful checkpoint state {category!r} of type "
                f"{type(target).__name__}"
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


@contextmanager
def local_state_preflight(states: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Coordinate isolated destinations, backend reading, then local validation."""
    with synchronize_local_errors("Checkpoint local preflight preparation"):
        selected, destination = prepare_local_state_preflight(states)
    validate_local_preflight_selection(selected)
    yield destination
    validate_local_state_preflight(selected, destination)


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


def prepare_hf_export(
    adapter: Any, state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int] | None]:
    """Prefer export-specific hooks, resolving legacy hooks only when absent."""
    try:
        convert = adapter.to_hf_export
    except AttributeError:
        convert = adapter.to_hf
    converted = convert(state)
    try:
        indices = adapter.hf_export_fqn_to_index_mapping
    except AttributeError:
        indices = adapter.fqn_to_index_mapping
    return converted, indices


@contextmanager
def hf_model_restore(
    adapter: Any, model_state: Any, native_state: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Share HF restore ordering while the backend owns reader IO.

    Collective-bearing adapters own preparation coordination; local adapters
    agree on failures before reader IO. Reader failures skip conversion and model
    callbacks, without rolling back tensors already written by the reader.
    """
    error_group = getattr(adapter, "error_group", None)
    with (
        nullcontext()
        if getattr(adapter, "coordinates_hf_preparation", False)
        else synchronize_local_errors(
            "HF checkpoint destination preparation", group=error_group
        )
    ):
        hf_state = adapter.to_hf(native_state)
    yield hf_state
    restored = adapt_hf_model_state(
        adapter, model_state, hf_state, error_group=error_group
    )
    model_state.load_state_dict(restored)


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
