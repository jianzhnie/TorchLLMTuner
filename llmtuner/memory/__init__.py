"""Memory optimizations applied to model execution.

This package owns transformations whose primary purpose is to reduce peak
device memory.  Activation checkpointing trades additional computation for
saved activations; region policies define the rematerialization boundaries
used by that transformation.

Training services such as optimizers, checkpoint persistence, metrics,
profiling, tokenization, and loss definitions live in ``llmtuner.components``.
Parallel topology and sharding remain in ``llmtuner.parallel``.
"""

from __future__ import annotations

from ..utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "apply_ac": "activation_checkpoint",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve memory optimization entry points on first use."""
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
