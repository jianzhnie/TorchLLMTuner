"""Rank-local default RNG streams in trusted distributed checkpoints.

Private loader generators are not included here. PyTorch 2.14's regular DTensor
tracker uses the selected device's default RNG; it has no independent RNG copy.
A rank-qualified opaque payload prevents DCP deduplicating distinct streams.
"""

from __future__ import annotations

import pickle
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful

from ...accelerator.distributed import get_rank, get_world_size

if TYPE_CHECKING:
    from ...parallel.topology import ParallelTopology


class RNGCheckpointState(Stateful):
    """Save Python, NumPy, CPU and this rank's selected device generators."""

    def __init__(
        self, device: torch.device, *, topology: ParallelTopology | None = None
    ) -> None:
        self.device = device
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.key = f"rank_{self.rank}"
        self.parallel_degrees = (
            None
            if topology is None
            else {
                name: getattr(topology, name)
                for name in (
                    "dp_replicate",
                    "dp_shard",
                    "cp",
                    "tp",
                    "pp",
                    "ep",
                    "cp_kv",
                    "cp_ulysses",
                )
            }
        )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "version": 1,
            "world_size": self.world_size,
            "device_type": self.device.type,
            "parallel_degrees": self.parallel_degrees,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "device": (
                None
                if self.device.type == "cpu"
                else torch.get_device_module(self.device).get_rng_state(self.device)
            ),
        }
        return {self.key: pickle.dumps(state, protocol=4)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError("RNG checkpoint state must be a dictionary")
        if not state_dict:
            return
        state = pickle.loads(state_dict[self.key])
        if not isinstance(state, dict):
            raise ValueError("RNG checkpoint payload must be a dictionary")
        for field in ("version", "world_size"):
            value = state.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"RNG checkpoint {field} must be a positive integer")
        if state["version"] != 1:
            raise ValueError("Unsupported RNG checkpoint version")
        if state["world_size"] != self.world_size:
            raise ValueError(
                "RNG checkpoint world size changed; exclude rng_state from loading "
                "to resume without exact random-stream continuity"
            )
        if state["device_type"] != self.device.type:
            raise ValueError(
                "RNG checkpoint device type changed; exclude rng_state from loading"
            )
        saved_degrees = state.get("parallel_degrees")
        valid_degrees = saved_degrees is None or (
            isinstance(saved_degrees, dict)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in saved_degrees.values()
            )
        )
        if not valid_degrees or saved_degrees != self.parallel_degrees:
            raise ValueError(
                "RNG checkpoint parallel topology changed or is missing; exclude "
                "rng_state from loading to resume without exact stream continuity"
            )
        # Validate host states without changing any live generator.
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator().set_state(state["cpu"])
        if self.device.type != "cpu":
            module = torch.get_device_module(self.device)
            module.set_rng_state(state["device"], self.device)
        torch.set_rng_state(state["cpu"])
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
