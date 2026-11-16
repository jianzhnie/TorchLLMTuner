"""The assembly stage list: the order contract, as data.

``parallelize_model``'s correctness is its call order, and this
table is the single source of ``STAGE_ORDER``. Kept engine-free (a dataclass and
tuples, no ``apply_*`` imports) so the contract is importable anywhere --
``llmtuner/config``, tests, docs tooling -- without paying for the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "STAGE_ORDER",
    "STAGES",
    "Stage",
    "StageName",
]

StageName = Literal["tp", "ep", "cp", "ac", "activation_swap", "compile", "fsdp"]


# The order IS the contract (see parallelize.py's docstring).
@dataclass(frozen=True)
class Stage:
    """One assembly stage: name and why it sits here."""

    name: StageName
    why_here: str


STAGES: tuple[Stage, ...] = (
    Stage(
        "tp",
        "Sharding wrappers first: TP cuts the projections before anything "
        "wraps the layers.",
    ),
    Stage(
        "ep",
        "After TP: expert replacement consumes TP's dense sharding and owns "
        "the routed experts on the sparse EP mesh.",
    ),
    Stage(
        "cp",
        "With the other sharding wrappers: attention consumes this rank's "
        "sequence shard before checkpointing and compilation.",
    ),
    Stage(
        "ac",
        "After the sharding wrappers (it must enclose the TP/CP-modified "
        "layer), before compile and FSDP -- torchtitan's order in "
        "``parallelize_llama``.",
    ),
    Stage(
        "activation_swap",
        "Outside AC to preserve the recompute lifecycle; caller-owned "
        "checkpoint inputs stay resident. Before FSDP and incompatible "
        "with compile.",
    ),
    Stage(
        "compile",
        "After AC, before FSDP; runs only when the caller passes compile=True.",
    ),
    Stage(
        "fsdp",
        "Last, so FSDP's hooks sit outermost.",
    ),
)

STAGE_ORDER: tuple[StageName, ...] = tuple(stage.name for stage in STAGES)
