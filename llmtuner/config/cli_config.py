"""Expose scalar config fields to HfArgumentParser through dataclass views.

The parser receives PARSER_GROUPS in a fixed order. Programmatic-only fields
are hidden with init=False in subclasses that preserve defaults and validation.
This prevents dictionaries, nested dataclasses, and other unsupported values
from appearing as unusable command-line flags.

LLMTunerConfig.from_groups grafts the separately parsed scheduler, checkpoint,
dataloader, metrics, and profiler groups into their owning configs. EMA,
validation, and the nested activation-checkpoint/compile configs remain
programmatic-only and retain their defaults when using this CLI.
"""

from __future__ import annotations

import dataclasses
import typing
from copy import copy

from .checkpoint_config import CheckpointConfig
from .data_config import DataloaderConfig
from .lr_scheduler_config import LRSchedulerConfig
from .model_config import ModelConfig
from .observability_config import MetricsConfig, ProfilerConfig
from .optimizer_config import OptimizerConfig
from .parallel_config import ParallelConfig
from .training_config import TrainingConfig

# The groups the CLI parses, in the order ``LLMTunerConfig.from_groups`` takes
# them. Parser order is significant: ``parse_args_into_dataclasses`` returns one
# instance per class in this exact order, and ``trainer/cli.py`` zips the two
# together. Kept here, next to the view, so the parser's group set has one
# definition and no caller can quietly drop or reorder a group.
PARSER_GROUPS: tuple[type, ...] = (
    ModelConfig,
    ParallelConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    TrainingConfig,
    CheckpointConfig,
    DataloaderConfig,
    MetricsConfig,
    ProfilerConfig,
)

# The groups that need a view, and which of their fields the CLI cannot carry.
# A new non-scalar field either joins this table or becomes a flag that rejects
# every value it is given; ``cli_view`` fails loudly if a name here goes stale.
PROGRAMMATIC_ONLY: dict[type, frozenset[str]] = {
    ModelConfig: frozenset({"arch_overrides"}),
    OptimizerConfig: frozenset({"param_groups", "lr_scheduler_config"}),
    CheckpointConfig: frozenset({"purge_exempt"}),
    ParallelConfig: frozenset({"fsdp_apply_modules", "module_fqns_per_model_part"}),
    TrainingConfig: frozenset(
        {
            "ema_config",
            "validation_config",
            # Only checkpoint/data/metrics/profiler have separate parser groups.
            # Other nested configs remain programmatic-only.
            "checkpoint_config",
            "dataloader_config",
            "metrics_config",
            "profiler_config",
            "compile_config",
            "selective_ac",
            "memory_budget_ac",
            "region_ac",
        }
    ),
}


def cli_view(cls: type) -> type:
    """``cls`` with its programmatic-only fields hidden from the parser."""
    hidden = PROGRAMMATIC_ONLY.get(cls)
    if not hidden:
        return cls
    unknown = hidden - set(cls.__dataclass_fields__)
    if unknown:
        raise RuntimeError(
            f"{cls.__name__} has no fields {sorted(unknown)}: "
            "config.cli_config.PROGRAMMATIC_ONLY is stale"
        )
    hints = typing.get_type_hints(cls)
    namespace: dict[str, typing.Any] = {
        "__annotations__": {name: hints[name] for name in hidden}
    }
    for name in hidden:
        option = copy(cls.__dataclass_fields__[name])
        option.init = False
        namespace[name] = option
    return dataclasses.dataclass(type(cls.__name__, (cls,), namespace))
