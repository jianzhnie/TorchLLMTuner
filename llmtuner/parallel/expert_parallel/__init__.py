"""Expert parallelism: swap HF MoE blocks for the EP-capable llmtuner MoE.

EP core idea (MoE): shard experts across ranks; an all-to-all routes each token
to its expert's rank and back. ``swap.py`` is the weight-moving swap itself;
``apply.py`` wires it onto a model given the EP process group.

The two entry points are re-exported lazily (PEP 562): ``swap.py`` pulls in the
whole MoE stack (``models/common/moe`` and its custom ops), and lighter
submodules like ``ckpt.py`` -- imported by the checkpointer and optimizer
containers -- must stay loadable on hosts whose torch predates those ops.
"""

from __future__ import annotations

from ...utils.lazy_exports import export_names, resolve_export

_EXPORT_SOURCES = {
    "apply_ep": "apply",
    "swap_hf_moe_blocks": "swap",
}

__all__ = export_names(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return export_names(_EXPORT_SOURCES)
