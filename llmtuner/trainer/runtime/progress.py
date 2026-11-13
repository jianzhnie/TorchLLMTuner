"""Checkpointable counters owned by one training run.

The checkpoint service retains this state instead of the Trainer and its model,
loader and resource graph. Counter names match the existing native format.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainingProgress:
    """Completed optimizer steps and logical tokens consumed by the run."""

    step: int = 0
    ntokens_seen: int = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate both counters before changing either live value."""
        step = state_dict["step"]
        ntokens_seen = state_dict["ntokens_seen"]
        for name, value in (("step", step), ("ntokens_seen", ntokens_seen)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"trainer {name} must be a nonnegative integer")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
