"""Accelerator layer: device discovery, communication primitives, and their
configuration functions.

Members:

* ``device.py`` -- llmtuner's backend-neutral device module (NPU/CUDA/MLU/MUSA
  discovery, distributed-backend selection, per-vendor predicates).
* ``collectives.py`` -- PG timeouts (``set_pg_timeouts``), the in-place
  ``all_reduce`` used for loss/token denominators, and EP-aware
  ``clip_grad_norm_``.
* ``spmd.py`` -- the ambient SPMD mesh context (TLS mesh stack and
  by-name process-group queries) that trainer and ``models/common`` read.
* ``distributed.py`` -- process-group bootstrap and rank/world queries, vendored
  from OpenMMLab's ``mmengine.dist`` and de-mmengine'd to depend only on torch.
* ``tensors.py`` -- nested tensor device inspection and movement.

``distributed`` is re-exported here lazily (PEP 562): importing this package or
a sibling submodule (``llmtuner.accelerator.device`` ...) does not pay for it
unless a toolbox name is actually touched. ``collectives`` /
``spmd`` is imported as a submodule -- re-exporting it would make
``import llmtuner.accelerator`` pull in the parallel and trainer layers and
close an import cycle. Topology construction (``build_parallel_topology`` /
``build_mesh``) lives with ``ParallelTopology`` in
``llmtuner/parallel/topology.py``; the trainer bootstraps its PG via
``distributed.init_dist_pytorch``.
"""

from __future__ import annotations

from ..utils.lazy_exports import export_names, resolve_export

_EXPORT_SOURCES = {
    "barrier": "distributed",
    "cast_data_device": "tensors",
    "get_backend": "distributed",
    "get_comm_device": "distributed",
    "get_data_device": "tensors",
    "get_dist_info": "distributed",
    "get_rank": "distributed",
    "get_world_size": "distributed",
    "infer_launcher": "distributed",
    "init_dist": "distributed",
    "is_distributed": "distributed",
    "is_main_process": "distributed",
    "master_only": "distributed",
}

__all__ = export_names(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve toolbox names on first touch (PEP 562 lazy re-export)."""
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return export_names(_EXPORT_SOURCES)
