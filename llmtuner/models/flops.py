"""Select model-owned FLOPs estimates through architecture registration."""

from typing import Any

from ..utils.flops import generic_flops_per_token
from .registry import get_model_spec


def flops_per_token(arch: Any, *, seq_len: int) -> int:
    """Estimate model work without constructing a model or allocating tensors."""
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
        raise ValueError(f"seq_len must be a non-negative integer, got {seq_len!r}")
    spec = get_model_spec(getattr(arch, "model_type", ""))
    estimate = spec.flops_estimator if spec is not None else None
    if estimate is None:
        estimate = generic_flops_per_token
    return estimate(arch, seq_len=seq_len)


def flops_estimation_note(arch: Any, *, seq_len: int) -> str | None:
    """Return the model's accounting caveat; callers own logging and rank policy."""
    spec = get_model_spec(getattr(arch, "model_type", ""))
    if spec is None or spec.flops_note is None:
        return None
    return spec.flops_note(arch, seq_len=seq_len)
