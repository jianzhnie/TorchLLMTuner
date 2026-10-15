"""Training optimizer services, loaded only when requested.

Parameter grouping lives in optimizer.py, state initialization and FQN
serialization in state_dict.py, schedules in lr_scheduler.py, and EMA in ema.py.
Implementation code imports the defining module directly.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "EMA": "ema",
    "LRSchedulersContainer": "lr_scheduler",
    "OptimizersContainer": "optimizer",
    "build_lr_scheduler": "lr_scheduler",
    "get_flat_optim_state_dict": "state_dict",
    "init_optim_state": "state_dict",
    "load_flat_optim_state_dict": "state_dict",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
