"""Parallelism: one ``apply_*`` per dimension, composed by one entry point.

Each family has one same-named public entry module. Supporting modules contain
mechanisms only: TP linear kernels, FSDP plans, CP kernels and input sharding,
and EP model conversion. External callers can use the
family packages; implementation code imports from the defining modules.

The composition entry calls each transformation in the declared stage order.
Each entry decides whether it is needed from the configuration and topology.
FSDP may still be required with a singleton DP shard axis when CP, replication,
or expert sharding is active; callers must not skip it based only on DP degree.

Pipeline parallelism has moved to ``experiments/pipeline_parallel``. Production
assembly returns a transformed module with its final local layout.
"""

from __future__ import annotations

from ..utils.lazy_exports import resolve_export

# Keep submodule imports independent of unrelated parallel implementations.
# Context parallelism pulls in the model stack and its spmd dependencies;
# resolving public entries lazily avoids loading that stack for every import.
_EXPORT_SOURCES = {
    "ParallelPlan": "plan",
    "PlanTarget": "plan",
    "apply_cp": "context_parallel.context_parallel",
    "apply_ep": "expert_parallel.expert_parallel",
    "apply_fsdp": "fully_shard_parallel.fully_shard_parallel",
    "apply_tp": "tensor_parallel.tensor_parallel",
    "parallelize_model": "parallelize",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve an ``apply_*`` entry point on first touch (PEP 562)."""
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
