"""Tensor parallelism public API.

``tensor_parallel.py`` is the entry and structural plan engine; ``sharding.py``
owns declarations and patterns, ``linear.py`` owns projection execution, and
``accelerator/linear_collectives.py`` provides collective/GEMM primitives.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "ColumnParallelLinear": "linear",
    "ColwiseLinearNoGather": "linear",
    "RowParallelLinear": "linear",
    "ShardingConfig": "sharding",
    "apply_tp": "tensor_parallel",
    "colwise": "sharding",
    "rowwise": "sharding",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
