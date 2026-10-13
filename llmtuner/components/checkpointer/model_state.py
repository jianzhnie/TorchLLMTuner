"""Checkpoint state view over model parts, with stable tensor storage.

Adapted from torchtitan ``components/checkpointer/base.py``. Flatten pipeline
parts and gather/slice EP expert state without depending on manager policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from ...parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    load_expert_state,
)

__all__ = ["ModelCheckpointState"]


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` are backed by the same storage.

    For ``DTensor`` the local shard is compared via ``_local_tensor`` rather
    than ``to_local()``, which is autograd-aware. The dispatcher-level alias
    check also supports wrapper subclasses without directly accessible storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return torch._C._is_alias_of(a, b)


class ModelCheckpointState(Stateful):
    """A ``Stateful`` view over one module or a list of them.

    Serves two purposes:

    1. **Flattening.** Combines the state dicts of several modules (individual
       chunks under pipeline parallelism) into one flat view, so the
       checkpointing code interacts with them through a single interface.
    2. **Stable-storage caching.** Caches the flattened state dict and, on every
       ``state_dict()`` call, returns tensors backed by the *same* storage.
       Async DCP staging may cache pinned host buffers keyed by the source
       storage, so keeping storage stable lets it reuse those buffers across
       saves -- this is the fast checkpoint path. Parameter tensors already
       satisfy this, since the cached view shares the parameter's storage.
       Tensors produced by module ``state_dict`` hooks (one that splits a fused
       parameter, say) may be freshly allocated each call, so they are refreshed
       in place: their storage stays put while their values track the parameters.

    Notes:
        - ``load_state_dict`` updates the underlying modules and refreshes the
          cache.
        - The module tree must not be structurally modified (keys changing,
          tensor references replaced) after wrapping, or the cache goes stale.
    """

    def __init__(self, model: nn.Module | Sequence[nn.Module]) -> None:
        parts = (model,) if isinstance(model, nn.Module) else tuple(model)
        if any(not isinstance(part, nn.Module) for part in parts):
            raise TypeError("Every checkpoint model part must be an nn.Module")
        # Own the collection: caller-side list edits must not invalidate the
        # EP shard map or change the set of modules being checkpointed.
        self.model = list(parts)
        # EP-sharded expert weights are rank-heterogeneous under one FQN;
        # they ride through this wrapper as all-gathered full tensors and are
        # sliced back per rank on load (parallel/expert_parallel/checkpointing.py).
        self._ep_shards = expert_shard_map(self.model)
        self.cached_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        flat = {k: v for model in self.model for k, v in model.state_dict().items()}
        return gather_expert_state(flat, self._ep_shards, self.model)

    def state_dict(self) -> dict[str, Any]:
        # Recompute so hook-produced tensors reflect the current parameters,
        # then merge into the cache without changing storage objects.
        current_state = self._get_state_dict()
        if current_state.keys() != self.cached_state_dict.keys():
            added = sorted(current_state.keys() - self.cached_state_dict.keys())
            removed = sorted(self.cached_state_dict.keys() - current_state.keys())
            raise RuntimeError(
                "Model checkpoint keys changed after wrapping; construct a new "
                f"ModelCheckpointState. Added: {added}; removed: {removed}"
            )
        for key, value in current_state.items():
            cached = self.cached_state_dict.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or not isinstance(cached, torch.Tensor)
                or cached.shape != value.shape
                or cached.dtype != value.dtype
                or cached.device != value.device
            ):
                self.cached_state_dict[key] = value
            elif not _shares_storage(cached, value):
                cached.copy_(value)
        return self.cached_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Each PP part consumes its own keys from the combined model state.
        # Other parts' keys are ignored; checkpoint state categories are separate.
        state_dict = load_expert_state(state_dict, self._ep_shards, self.model)
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()
