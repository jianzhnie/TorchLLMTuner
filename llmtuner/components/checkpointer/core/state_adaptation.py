"""Shared state-dict adaptation at checkpoint format boundaries."""

from contextlib import nullcontext
from typing import Any

from ....accelerator.collectives import synchronize_local_errors


def adapt_hf_model_state(
    adapter: Any, model_state: Any, hf_state: dict[str, Any]
) -> dict[str, Any]:
    """Convert HF keys and validate the native destination before loading.

    Kimi EP conversion performs its own tensor-layout collectives, so it must
    remain outside rank-local error coordination. Other adapters are local and
    must agree on failures before model load callbacks run. The preceding
    reader may already have written into live tensors; this is not a rollback
    boundary.
    """
    with (
        nullcontext()
        if getattr(adapter, "kimi_expert_parallel", False)
        else synchronize_local_errors("HF checkpoint state adaptation")
    ):
        native_state = adapter.from_hf(hf_state)

    with synchronize_local_errors("HF checkpoint model validation"):
        validate = getattr(model_state, "validate_state_dict", None)
        if callable(validate):
            validate(native_state)
    return native_state
