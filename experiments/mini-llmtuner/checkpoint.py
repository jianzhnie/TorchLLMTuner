"""Checkpoint conversion boundary, independent from model and trainer."""

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


class CheckpointAdapter:
    """Converts runtime state to a portable prototype checkpoint."""

    def export(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        signature: str,
    ) -> dict[str, Any]:
        model_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "signature": signature,
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
        }

    def restore(
        self,
        payload: dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        signature: str,
    ) -> int:
        if payload["signature"] != signature:
            raise ValueError("Checkpoint configuration/topology differs from this run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng"])
        random.setstate(payload["python_rng"])
        return int(payload["step"])


class CheckpointManager:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        signature: str,
        adapter: CheckpointAdapter | None = None,
        *,
        scheduler=None,
        data=None,
    ) -> None:
        self.model, self.optimizer, self.signature = model, optimizer, signature
        self.adapter = adapter or CheckpointAdapter()
        self.step = 0
        self.scheduler, self.data = scheduler, data

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = self.adapter.export(
            self.model, self.optimizer, self.step, self.signature
        )
        payload["version"] = 2
        for name in ("scheduler", "data"):
            component = getattr(self, name)
            if component is not None:
                payload[name] = component.state_dict()
        fd, temporary = tempfile.mkstemp(dir=path, prefix="state-", suffix=".pt")
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path / "state.pt")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path: Path) -> int:
        payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        if payload.get("version") != 2:
            raise ValueError("Unsupported checkpoint version; expected version 2")
        if payload["signature"] != self.signature:
            raise ValueError("Checkpoint configuration/topology differs from this run")
        for name in ("scheduler", "data"):
            if getattr(self, name) is not None and name not in payload:
                raise ValueError(f"Missing checkpoint component: {name}")
        self.step = self.adapter.restore(
            payload, self.model, self.optimizer, self.signature
        )
        for name in ("scheduler", "data"):
            component = getattr(self, name)
            if component is not None:
                component.load_state_dict(payload[name])
        return self.step
