"""Stable contracts shared by models, parallel adapters and trainers."""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class ModelProtocol(Protocol):
    def preprocess_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: ...
    def parallelize(self, plan: Any, *, world_size: int) -> Any: ...
    def forward(
        self, input_ids: torch.Tensor, *, position_ids: torch.Tensor
    ) -> torch.Tensor: ...


class CheckpointAdapterProtocol(Protocol):
    def export(self, model, optimizer, step: int, signature: str) -> dict: ...
    def restore(self, payload: dict, model, optimizer, signature: str) -> int: ...
