"""Context parallelism: the CP flex kernel, input sharding, and wiring.

``context_parallel.py`` is the public entry and wires CP onto a model.
``attention.py`` holds attention redistribution, while
``input_sharding.py`` owns batch and mask sharding.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "CPFlexKernel": "attention",
    "DualCPFlexKernel": "attention",
    "apply_cp": "context_parallel",
    "shard_attention_mask_for_cp": "input_sharding",
    "shard_batch_for_cp": "input_sharding",
    "shard_batch_for_tp": "input_sharding",
    "shard_padding_mask_for_cp": "input_sharding",
    "shard_padding_mask_for_tp": "input_sharding",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
