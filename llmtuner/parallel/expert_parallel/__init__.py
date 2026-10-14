"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

``expert_parallel.py`` is the public entry. ``hf_inspection.py`` inspects HF
layouts, ``block_conversion.py`` constructs native MoE blocks,
``model_replacement.py`` replaces them in a complete model, and
``checkpointing.py`` converts expert checkpoint state. ``token_dispatch.py``
implements local and distributed token dispatch and combine operations.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "restore_fp32_state_buffers": "block_conversion",
    "apply_ep": "expert_parallel",
    "swap_hf_moe_blocks": "model_replacement",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
