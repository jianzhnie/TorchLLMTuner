"""Optimizer and parameter-group configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.errors import ConfigError


@dataclass
class ParamGroupConfig:
    """One parameter group and the optimizer that owns it.

    A list of these, in order, is what ``OptimizerConfig.param_groups`` means:
    each parameter is claimed by the first entry whose ``pattern`` matches its
    FQN, so a catch-all ``.*`` belongs last.

    ``optimizer_name`` and ``optimizer_kwargs`` define the group's optimizer
    completely -- there is no inheritance from ``OptimizerConfig``'s flat
    scalars, which supply the *default* catch-all group and nothing else. A
    group that omits an option uses the selected optimizer class's default,
    so set ``lr`` explicitly when the run requires a particular learning rate.

    There is no CLI flag for this: ``HfArgumentParser`` builds one flag per
    dataclass *field*, and a list of nested dataclasses has no flag spelling.
    Groups are set from code, by constructing an ``OptimizerConfig`` with
    ``param_groups=[...]``. See that class's ``__post_init__`` for how the flat
    scalars become a group when this list is left empty.
    """

    pattern: str
    """Regex matched against parameter FQNs, e.g. ``r".*\\.bias$"``, ``r".*"``."""

    optimizer_name: str
    """Optimizer class for this group's parameters: ``"Adam"`` or ``"AdamW"``."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the optimizer constructor; omitted options use
    its defaults. Nothing is filled in from the enclosing config.
    Entries override the run-wide implementation kwargs, so a group can ask for
    ``fused=False`` where the run default is fused."""


@dataclass
class OptimizerConfig:
    """Default AdamW options and explicit Adam/AdamW parameter groups.

    The schedule lives here rather than in ``TrainingConfig`` because it
    scales this group's learning rate: a lr with no schedule is the degenerate
    case of one, and splitting them would let the two be set independently.
    """

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
            "help": "Optimizer kernel: 'fused' | 'foreach' | 'for-loop'. "
            "'fused' needs a device torch ships a fused Adam kernel for "
            "(CUDA/XPU); on any other device the request is honoured as the "
            "for-loop kernel, which computes the same update. torchtitan's "
            "default."
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
    lr_scheduler_config: LRSchedulerConfig = field(
        default_factory=LRSchedulerConfig,
        metadata={"help": "Learning-rate schedule (see components/optimizer)."},
    )

    def __post_init__(self) -> None:
        if self.implementation not in ("fused", "foreach", "for-loop"):
            raise ConfigError(
                f"Unknown optimizer implementation {self.implementation!r}; "
                "expected one of 'fused', 'foreach', 'for-loop'."
            )
        # ``list`` is what the parser can build from two CLI values, but the
        # optimizer wants a tuple and the field must not be mutable: an
        # in-place edit to a shared list would leak across instances.
        # Normalizing here makes that unreachable, and the length
        # check is what turns a typo like ``--betas 0.9`` into an error rather
        # than a one-element betas that torch rejects deep in a step.
        betas = tuple(self.betas)
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
        self.betas = betas

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
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "betas": self.betas,
                        "eps": self.eps,
                    },
                )
            ]

    @property
    def lr_scheduler(self) -> LRSchedulerConfig:
        """The schedule's config.

        A property backed by ``lr_scheduler_config`` so the flat
        ``cfg.lr_scheduler`` spelling works at the call site, while the parser
        still sees a real field to generate flags from. See
        ``TrainingConfig.checkpoint`` for the same shape.
        """
        return self.lr_scheduler_config
