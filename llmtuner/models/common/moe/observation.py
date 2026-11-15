"""Model-owned boundaries for optional dispatch observations."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class DispatchObserver(Protocol):
    def measure(self, device: torch.device) -> AbstractContextManager[None]: ...


@dataclass(eq=False, slots=True)
class DispatchObservation:
    """Declare a dispatch boundary that the model actually enters in forward."""

    observer: DispatchObserver | None = None

    def measure(self, device: torch.device) -> AbstractContextManager[None]:
        return nullcontext() if self.observer is None else self.observer.measure(device)
