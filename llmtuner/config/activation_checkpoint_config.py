"""Activation-checkpointing configs: the mode's accepted values and the per-policy
settings grafted onto ``TrainingConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmtuner.errors import ConfigError

VALID_AC_MODES: tuple[str, ...] = (
    "none",
    "full",
    "selective",
    "memory_budget",
    "region",
)
"Accepted activation-checkpointing modes."


def _validate_patterns(value: list[str], *, field_name: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(pattern, str) or not pattern.strip() for pattern in value
    ):
        raise ConfigError(f"{field_name} must be a list of nonempty strings")


@dataclass(kw_only=True)
class SelectiveACConfig:
    """Settings for ``activation_checkpoint_mode='selective'``."""

    force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=list,
        metadata={
            "help": "FQN substrings selecting nn.Linear weight shapes for "
            "unconditional recomputation."
        },
    )
    preserve_rng_state: bool = field(
        default=True,
        metadata={
            "help": "Stash and restore the RNG state around each checkpointed "
            "region so the backward-time recompute redraws the same random "
            "values, at some speed cost. Set false only if the checkpointed "
            "region is known to hold no random state."
        },
    )
    determinism_check: str = field(
        default="default",
        metadata={
            "help": "Determinism function torch's checkpoint uses to compare "
            "the recompute against the original. 'default' checks tensors that "
            "have no data-invariant structure; 'none' disables."
        },
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Must stay false: PyTorch checkpoint debug tracing is "
            "incompatible with selective checkpointing's custom context_fn."
        },
    )

    def __post_init__(self) -> None:
        _validate_patterns(
            self.force_recompute_mm_shapes_by_fqns,
            field_name="selective_ac.force_recompute_mm_shapes_by_fqns",
        )
        for name in ("preserve_rng_state", "debug"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ConfigError(f"selective_ac.{name} must be bool, got {value!r}")
        if self.debug:
            raise ConfigError(
                "selective_ac.debug must stay False: PyTorch checkpoint "
                "debug=True is incompatible with the custom context_fn "
                "required by selective activation checkpointing."
            )
        if self.determinism_check not in ("default", "none"):
            raise ConfigError(
                "selective_ac.determinism_check must be 'default' or 'none', "
                f"got {self.determinism_check!r}"
            )


@dataclass(kw_only=True)
class MemoryBudgetACConfig:
    """Settings for ``activation_checkpoint_mode='memory_budget'``."""

    memory_budget: float = field(
        default=0.5,
        metadata={
            "help": "How much the compile partitioner trades compute for "
            "memory: 0.0 is the activation memory of full checkpointing over "
            "the compiled region, 1.0 the default runtime-optimized strategy. "
            "Must be in [0, 1]."
        },
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_budget, bool)
            or not isinstance(self.memory_budget, int | float)
            or not 0 <= self.memory_budget <= 1
        ):
            raise ConfigError(
                "memory_budget_ac.memory_budget must be between 0 and 1 "
                "(numeric, excluding bool), "
                f"got {self.memory_budget!r}"
            )


@dataclass(kw_only=True)
class RegionACConfig:
    """Settings for ``activation_checkpoint_mode='region'``."""

    save_regions: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Block-relative glob patterns for regions whose outputs are "
            "retained, such as attention projections."
        },
    )
    recompute_regions: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Block-relative glob patterns forcing recomputation; override "
            "matching save_regions."
        },
    )
    determinism_check: str = field(
        default="default",
        metadata={
            "help": "The check torch_remat runs to compare the recompute "
            "against the original forward. 'default' checks the tensors that "
            "have no data-invariant structure; 'none' disables it, which is "
            "torch_remat's own default."
        },
    )
    preserve_rng_state: bool = field(
        default=False,
        metadata={
            "help": "Must stay false: torch_remat does not preserve torch's "
            "RNG state, and a region whose callable draws random numbers needs "
            "an explicit RecomputeStateHook instead."
        },
    )

    def __post_init__(self) -> None:
        for name in ("save_regions", "recompute_regions"):
            _validate_patterns(getattr(self, name), field_name=f"region_ac.{name}")
        if not isinstance(self.preserve_rng_state, bool):
            raise ConfigError(
                "region_ac.preserve_rng_state must be bool, "
                f"got {self.preserve_rng_state!r}"
            )
        if self.determinism_check not in ("default", "none"):
            raise ConfigError(
                "region_ac.determinism_check must be 'default' or 'none', got "
                f"{self.determinism_check!r}"
            )
        if self.preserve_rng_state:
            raise ConfigError(
                "region_ac.preserve_rng_state must stay False: region "
                "activation checkpointing does not support it -- "
                "torch_remat.checkpoint refuses True because a generator drawn "
                "inside a skipped save region would desync the recompute, and "
                "boundary-only stashing would hide that rather than fix it. "
                "Register a RecomputeStateHook for the random state your "
                "retained regions use, or leave it false."
            )
