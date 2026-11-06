"""Optimizer and parameter-group configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_layout import BucketConfig, ComputeLayout
from llmtuner.config.utils import require_integer_at_least
from llmtuner.errors import ConfigError


@dataclass
class ParamGroupConfig:
    """One parameter group and the optimizer that owns it."""

    pattern: str
    """Regex matched against parameter FQNs, e.g. ``r".*\\.bias$"``, ``r".*"``."""

    optimizer_name: str
    """Optimizer: ``Adam``, ``AdamW``, ``AdamWSwap``, or ``DistMuon``."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    "Per-group optimizer arguments; omitted values use optimizer defaults, not "
    "enclosing config values. Unknown options are rejected during construction; "
    "initial_lr is allowed as scheduler metadata."


def validated_betas(value: list[float]) -> tuple[float, ...]:
    try:
        betas = tuple(value)
    except TypeError as error:
        raise ConfigError(
            "optimizer.betas must contain two numeric entries; "
            "pass both: --betas 0.9 0.95."
        ) from error
    if len(betas) != 2:
        raise ConfigError(
            f"optimizer.betas must have exactly 2 entries (beta1, beta2), "
            f"got {len(betas)}: {betas}. Pass both: --betas 0.9 0.95."
        )
    if not all(
        isinstance(beta, int | float)
        and not isinstance(beta, bool)
        and 0.0 <= beta < 1.0
        for beta in betas
    ):
        raise ConfigError(f"optimizer.betas must each be in [0, 1), got {betas}")
    return betas


@dataclass
class OptimizerConfig:
    """Default AdamW options and explicit optimizer parameter groups."""

    optimizer_type: Literal["Adam", "AdamW", "AdamWSwap"] = field(
        default="AdamW", metadata={"help": "Optimizer class"}
    )
    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay"})
    # Annotated as the CLI's input shape; __post_init__ normalizes to the
    # tuple the optimizer actually consumes (root's flat view reads that one).
    betas: list[float] = field(
        default_factory=lambda: [0.9, 0.999],
        metadata={
            "help": "AdamW (beta1, beta2). Pass as two values: "
            "--betas 0.9 0.95. The default is torch's; torchtitan's reference "
            "LLM recipe uses (0.9, 0.95), which is a NUMERIC change."
        },
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "AdamW epsilon (denominator floor)."},
    )
    implementation: Literal["fused", "foreach", "for-loop"] = field(
        default="fused",
        metadata={
            "help": "Optimizer kernel: fused, foreach, or for-loop. Fused requires a "
            "supported device."
        },
    )
    state_residency: Literal["device", "npu_virtual"] = field(
        default="device",
        metadata={
            "help": "Adam moment storage: device or Host-backed NPU virtual memory. "
            "npu_virtual requires native DCP checkpoints and is incompatible with "
            "FSDP CPU offload."
        },
    )
    swap_max_device_bytes: int = field(
        default=64 * 1024**2,
        metadata={
            "help": "AdamWSwap total staging payload limit across lookahead slots, "
            "including offloaded parameter/gradient copies; excludes update "
            "temporaries and allocator cache. Large parameters are sliced."
        },
    )
    swap_prefetch: bool = field(
        default=False, metadata={"help": "Overlap one lookahead H2D slice with update."}
    )
    swap_memory_fraction: float | None = field(
        default=None,
        metadata={
            "help": "Fraction of free accelerator memory for staging; "
            "also capped by swap_max_device_bytes."
        },
    )
    swap_max_host_bytes: int = field(
        default=0,
        metadata={
            "help": "AdamWSwap owned Host moment payload limit per optimizer "
            "(0: unlimited); excludes RSS and allocator cache."
        },
    )
    param_groups: list[ParamGroupConfig] = field(
        default_factory=list,
        metadata={
            "help": "Per-parameter-group optimizers. Empty (the default) means "
            "one catch-all group built from the flat scalars above. No CLI flag "
            "-- nested dataclass lists cannot be parsed; set it from code."
        },
    )
    compute_sharding_by_fqn: dict[str, ComputeLayout] = field(
        default_factory=dict,
        metadata={
            "help": "DistMuon per-parameter compute layouts. Required for every "
            "DistMuon parameter and rebuilt with matching values before restore."
        },
    )
    bucket_configs: list[BucketConfig] = field(
        default_factory=list,
        metadata={"help": "DistMuon packed redistribution buckets."},
    )
    lr_scheduler_config: LRSchedulerConfig = field(
        default_factory=LRSchedulerConfig,
        metadata={"help": "Learning-rate schedule (see components/optimizer)."},
    )

    def __post_init__(self) -> None:
        if not isinstance(self.swap_prefetch, bool):
            raise ConfigError("optimizer.swap_prefetch must be bool")
        fraction = self.swap_memory_fraction
        if fraction is not None and (
            isinstance(fraction, bool)
            or not isinstance(fraction, int | float)
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ConfigError("optimizer.swap_memory_fraction must be in (0, 1]")
        require_integer_at_least(
            self, "swap_max_device_bytes", minimum=8, group="optimizer"
        )
        require_integer_at_least(
            self, "swap_max_host_bytes", minimum=0, group="optimizer"
        )
        if self.optimizer_type not in ("Adam", "AdamW", "AdamWSwap"):
            raise ConfigError(
                f"optimizer.optimizer_type must be 'Adam', 'AdamW' or 'AdamWSwap', "
                f"got {self.optimizer_type!r}"
            )
        if self.implementation not in ("fused", "foreach", "for-loop"):
            raise ConfigError(
                f"Unknown optimizer implementation {self.implementation!r}; "
                "expected one of 'fused', 'foreach', 'for-loop'."
            )
        if self.state_residency not in ("device", "npu_virtual"):
            raise ConfigError(
                "optimizer.state_residency must be 'device' or 'npu_virtual', "
                f"got {self.state_residency!r}"
            )
        # ``list`` is what the parser can build from two CLI values, but the
        # optimizer wants a tuple and the field must not be mutable: an
        # in-place edit to a shared list would leak across instances.
        # Normalizing here makes that unreachable, and the length
        # check is what turns a typo like ``--betas 0.9`` into an error rather
        # than a one-element betas that torch rejects deep in a step.
        self.betas = validated_betas(self.betas)

        for name in ("learning_rate", "weight_decay", "eps"):
            value = getattr(self, name)
            positive = name == "eps"
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                bound = "> 0" if positive else ">= 0"
                raise ConfigError(
                    f"optimizer.{name} must be finite and {bound}, got {value}"
                )

        # The degenerate grouping: no explicit param_groups means one catch-all
        # group over every trainable parameter, built from the flat scalars.
        # Synthesized here rather than in the container so there is exactly one
        # description of the default, and so a caller that reads
        # ``cfg.param_groups`` sees the groups a run actually uses.
        if not self.param_groups:
            self.param_groups = [
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name=self.optimizer_type,
                    optimizer_kwargs={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "betas": self.betas,
                        "eps": self.eps,
                    },
                )
            ]
        if self.uses_optimizer_swap and (
            self.state_residency != "device" or self.implementation == "foreach"
        ):
            raise ConfigError(
                "AdamWSwap requires state_residency='device' and "
                "implementation='for-loop' or 'fused'; "
                "it is independent of npu_virtual."
            )

    @property
    def uses_optimizer_swap(self) -> bool:
        """Whether any configured parameter group selects explicit state exchange."""
        return any(group.optimizer_name == "AdamWSwap" for group in self.param_groups)

    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        """The schedule's config."""
        return self.lr_scheduler_config
