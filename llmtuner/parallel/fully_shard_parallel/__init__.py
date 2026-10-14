"""FSDP public API.

``fully_shard_parallel.py`` is the runtime entry, ``plan.py`` owns pattern and
prefetch plans, and ``fine_grained.py`` selects the PyTorch 2.14 strategy.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "apply_fsdp": "fully_shard_parallel",
    "apply_fsdp_to_decoder": "fully_shard_parallel",
    "disable_fsdp_gradient_division": "fully_shard_parallel",
    "enable_fsdp_symm_mem": "fully_shard_parallel",
    "fsdp_shard_size": "fully_shard_parallel",
    "get_fsdp_reshard_after_forward_policy": "fully_shard_parallel",
    "iter_fsdp_modules": "fully_shard_parallel",
    "iter_transformer_layers": "fully_shard_parallel",
    "resolve_fsdp_mesh": "fully_shard_parallel",
    "resolve_sparse_fsdp_mesh": "fully_shard_parallel",
    "apply_fsdp_by_pattern": "plan",
    "set_fsdp_prefetch_windows": "plan",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
