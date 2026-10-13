"""Training optimizer services.

``optimizer.py`` groups optimizers across model parts; ``state_dict.py``
initializes and serializes optimizer state by parameter name.
``lr_scheduler.py`` builds learning-rate schedules, and ``ema.py`` maintains
checkpointable moving averages. The exports below form the public API.
"""

from .ema import EMA
from .lr_scheduler import LRSchedulersContainer, build_lr_scheduler
from .optimizer import OptimizersContainer
from .state_dict import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

__all__ = [
    "EMA",
    "LRSchedulersContainer",
    "OptimizersContainer",
    "build_lr_scheduler",
    "get_flat_optim_state_dict",
    "init_optim_state",
    "load_flat_optim_state_dict",
]
