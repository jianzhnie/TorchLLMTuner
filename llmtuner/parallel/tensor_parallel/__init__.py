"""Tensor parallelism public API.

``tensor_parallel.py`` is the entry and plan engine; ``linear_ops.py`` contains
the fused collective/GEMM primitives it realizes plans into.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "ColumnParallelLinear": "tensor_parallel",
    "ColwiseLinearNoGather": "tensor_parallel",
    "RowParallelLinear": "tensor_parallel",
    "ShardingConfig": "tensor_parallel",
    "apply_tp": "tensor_parallel",
    "colwise": "tensor_parallel",
    "rowwise": "tensor_parallel",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
