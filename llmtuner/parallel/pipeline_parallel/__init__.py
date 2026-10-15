"""Pipeline parallelism public API and stage/schedule assembly.

Public symbols are loaded on request; importing a supporting module does not
load the runtime entry or optional communication kernels.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "PipelineParallelSetup": "pipeline_parallel",
    "apply_pp": "pipeline_parallel",
    "build_pipeline_schedule": "pipeline_parallel",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
