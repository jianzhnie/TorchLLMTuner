"""Reuse production FQN/layout adapters for synchronous DCP recovery."""

import json
import pickle
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.core.model_state import ModelCheckpointState
from llmtuner.components.optimizer.optimizer import OptimizersContainer


class TrainingState:
    """Own the completed-step cursor and rank-local RNG, not model tensors."""

    def __init__(self, device: torch.device, signature: dict[str, Any]) -> None:
        self.device = device
        self.signature = json.dumps(signature, sort_keys=True, default=str)
        self.next_step = 0

    def state_dict(self) -> dict[str, Any]:
        local = {
            "python": pickle.dumps(random.getstate()),
            "cpu": torch.get_rng_state(),
        }
        if self.device.type != "cpu":
            local["device"] = getattr(torch, self.device.type).get_rng_state(
                self.device
            )
        states = [local]
        if dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, local)
        return {
            "next_step": self.next_step,
            "signature": self.signature,
            "rng": {str(rank): state for rank, state in enumerate(states)},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with synchronize_local_errors("Experiment checkpoint compatibility"):
            if state["signature"] != self.signature:
                raise ValueError(
                    "Checkpoint configuration/topology differs from this run"
                )
        rank = dist.get_rank() if dist.is_initialized() else 0
        local = state["rng"][str(rank)]
        self.next_step = int(state["next_step"])
        random.setstate(pickle.loads(local["python"]))
        torch.set_rng_state(local["cpu"].cpu())
        if self.device.type != "cpu":
            getattr(torch, self.device.type).set_rng_state(
                local["device"].cpu(), self.device
            )


class CheckpointManager:
    """Synchronous native DCP; targets are built before DCP plans a load.

    Model and optimizer retain their separate lifecycle and TP/EP adapters.
    Checkpoints are trusted local artifacts; Python RNG uses pickle bytes.
    This experiment deliberately rejects configuration/topology changes.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: OptimizersContainer,
        training: TrainingState,
    ) -> None:
        self.model = ModelCheckpointState(model)
        self.optimizer = optimizer
        self.training = training

    def _state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "training": self.training,
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        dcp.save(self._state(), checkpoint_id=str(path))
        self.model.release_cache()

    def load(self, path: Path) -> int:
        # Validate topology/data settings before touching model or moments.
        dcp.load({"training": self.training}, checkpoint_id=str(path))
        dcp.load(
            {"model": self.model, "optimizer": self.optimizer},
            checkpoint_id=str(path),
        )
        return self.training.next_step
