"""Checkpoint RNG streams with the experimental pipeline topology."""

from __future__ import annotations

import torch

from llmtuner.components.checkpointer.core.random_state import (
    RNGCheckpointState as BaseRNGCheckpointState,
)

from .topology import ParallelTopology, ParallelTopologySnapshot


class RNGCheckpointState(BaseRNGCheckpointState):
    """Keep pipeline degrees in the rank-local checkpoint metadata."""

    def __init__(
        self,
        device: torch.device,
        *,
        topology: ParallelTopology | ParallelTopologySnapshot | None = None,
    ) -> None:
        super().__init__(device)
        self.topology = (
            None if topology is None else ParallelTopologySnapshot.capture(topology)
        )
        if self.topology is not None and self.topology.world_size != self.world_size:
            raise ValueError("RNG topology world size differs from the runtime")
