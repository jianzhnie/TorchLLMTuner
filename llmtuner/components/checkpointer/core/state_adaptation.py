"""Shared state-dict adaptation at checkpoint format boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from ....accelerator.collectives import synchronize_local_errors

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


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
