"""Fine-grained FSDP strategies and lifecycle implementation."""

from ....utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "CustomFSDPStrategy": "fine_grained",
    "NativeFSDPStrategy": "fine_grained",
    "fully_shard_fine_grained": "fine_grained",
    "get_fsdp_strategy": "fine_grained",
    "FineGrainedFSDPParamGroup": "fine_grained_param_group",
    "FineGrainedFSDPState": "fine_grained_state",
    "HookAllGatherState": "fine_grained_param_group",
    "HookReduceScatterState": "fine_grained_param_group",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
