"""Parallelism: one ``apply_*`` per dimension, composed by one entry point.

Each family has one same-named public entry module. Supporting modules contain
mechanisms only: TP linear kernels, FSDP plans, CP kernels and input sharding,
EP model conversion, and PP stage splitting. Callers import the family package
instead of reaching into those implementation files.

Every ``apply_*`` is a no-op when its degree is 1. That is what lets the same
trainer code run from step 0 (single device) through step 4 (full hybrid
parallelism) without an ``if degree > 1`` at any call site -- and it is why
``parallelize_hf_transformers`` can apply all of them unconditionally.

Pipeline parallelism lives in ``pipeline_parallel/``: ``pipeline.py`` computes
the stage split and ``pipeline_parallel.py`` builds this rank's schedule.
``parallelize_hf_transformers`` dispatches to it when ``pp > 1`` and returns a
``PipelineParallelSetup`` instead of a model.
"""

from __future__ import annotations

from ..utils.lazy_exports import export_names, resolve_export

# Lazy (PEP 562): ``matrix`` is read by ``llmtuner/config``, which must stay
# importable without the engine layer (context_parallel pulls in the model
# stack and its spmd surface). Eager re-exports would make any submodule
# import -- matrix included -- pay for the whole engine.
_EXPORT_SOURCES = {
    "apply_cp": "context_parallel",
    "apply_ep": "expert_parallel",
    "apply_fsdp": "fully_shard_parallel",
    "apply_tp": "tensor_parallel",
    "parallelize_hf_transformers": "parallelize",
}

__all__ = export_names(_EXPORT_SOURCES)


def __getattr__(name: str):
    """Resolve an ``apply_*`` entry point on first touch (PEP 562)."""
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return export_names(_EXPORT_SOURCES)
