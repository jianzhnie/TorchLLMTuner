"""Accelerator layer: device discovery, communication primitives, and their
configuration functions.

Members:

* ``device.py`` -- llmtuner's backend-neutral device module (NPU/CUDA/MLU/MUSA
  discovery, distributed-backend selection, per-vendor predicates).
* ``collectives.py`` -- PG timeouts (``set_pg_timeouts``), the in-place
  ``all_reduce`` used for loss/token denominators, and EP-aware
  ``clip_grad_norm_``.
* ``spmd_context.py`` -- the ambient SPMD mesh context (TLS mesh stack and
  by-name process-group queries) that trainer and ``models/common`` read.
* ``dist_utils.py`` -- process-group bootstrap and rank/world queries, vendored
  from OpenMMLab's ``mmengine.dist`` and de-mmengine'd to depend only on torch
  and ``.device`` (multi-launcher ``init_dist``, ``cast_data_device``).

``dist_utils`` is re-exported here lazily (PEP 562): importing this package or
a sibling submodule (``llmtuner.accelerator.device`` ...) does not pay for it
unless a toolbox name is actually touched. ``collectives`` /
``spmd_context`` are imported as submodules -- re-exporting them would make
``import llmtuner.accelerator`` pull in the parallel and trainer layers and
close an import cycle. Topology construction (``build_parallel_dims`` /
``build_mesh``) lives with ``ParallelDims`` in
``llmtuner/parallel/parallel_dims.py``; the trainer bootstraps its PG via
``dist_utils.init_dist_pytorch``.
"""

from __future__ import annotations

from ..utils.lazy_exports import export_names, resolve_export

_EXPORT_SOURCES = {
    "barrier": "dist_utils",
    "cast_data_device": "dist_utils",
    "get_backend": "dist_utils",
    "get_comm_device": "dist_utils",
    "get_data_device": "dist_utils",
    "get_dist_info": "dist_utils",
    "get_rank": "dist_utils",
    "get_world_size": "dist_utils",
    "infer_launcher": "dist_utils",
    "init_dist": "dist_utils",
    "is_distributed": "dist_utils",
    "is_main_process": "dist_utils",
    "master_only": "dist_utils",
}

__all__ = export_names(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve toolbox names on first touch (PEP 562 lazy re-export)."""
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return export_names(_EXPORT_SOURCES)
