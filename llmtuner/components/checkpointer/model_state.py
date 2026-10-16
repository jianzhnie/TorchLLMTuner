"""Checkpoint state view over model parts, with stable tensor storage.

Adapted from torchtitan ``components/checkpointer/base.py``. Flatten pipeline
parts and gather/slice EP expert state without depending on manager policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from ...accelerator.collectives import synchronize_local_errors
from ...parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    load_expert_state,
)
from ...parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = ["ModelCheckpointState"]


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
        - A hook that changes the exported view over aliased live storage forces
          one cache allocation to avoid mutating model tensors during refresh.
    """

    def __init__(
        self,
        model: nn.Module | Sequence[nn.Module],
        *,
        initialization_group: ProcessGroup | None = None,
    ) -> None:
        # Managers opt into a world boundary before any EP collection. Direct
        # state wrappers can be rank-local and retain that default contract.
        with (
            synchronize_local_errors(
                "Checkpoint model preparation", group=initialization_group
            )
            if initialization_group is not None
            else nullcontext()
        ):
            parts = (model,) if isinstance(model, nn.Module) else tuple(model)
            if any(not isinstance(part, nn.Module) for part in parts):
                raise TypeError("Every checkpoint model part must be an nn.Module")
            # Own the collection: caller-side list edits must not invalidate the
            # EP shard map or change the set of modules being checkpointed.
            from torch._dynamo.eval_frame import OptimizedModule

            # Compilation changes execution, not checkpoint ownership or names.
            self.model = [
                part._orig_mod if isinstance(part, OptimizedModule) else part
                for part in parts
            ]
            # EP-sharded expert weights are rank-heterogeneous under one FQN;
            # they ride through this wrapper as all-gathered full tensors and are
            # sliced back per rank on load (parallel/expert_parallel/checkpointing.py).
            self._ep_shards = expert_shard_map(self.model)
        self.cached_state_dict = self._get_state_dict(
            initialization_group=initialization_group
        )

    def _get_state_dict(
        self, *, initialization_group: ProcessGroup | None = None
    ) -> dict[str, Any]:
        # Hooks may redistribute DTensors: finish their own communication before
        # entering the local merge boundary, and never wrap them as local work.
        part_states = [model.state_dict() for model in self.model]
        flat: dict[str, Any] = {}
        with (
            synchronize_local_errors(
                "Checkpoint model state merge", group=initialization_group
            )
            if initialization_group is not None
            else nullcontext()
        ):
            for part_index, part_state in enumerate(part_states):
                duplicates = flat.keys() & part_state.keys()
                if duplicates:
                    raise ValueError(
                        f"Model checkpoint keys overlap at part {part_index}: "
                        f"{sorted(duplicates)}. Model parts must use distinct FQNs."
                    )
                flat.update(part_state)
        return gather_tensor_parallel_state(
            gather_expert_state(
                flat, shard_map=self._ep_shards, model_parts=self.model
            ),
            model_parts=self.model,
        )

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
            else:
                # Inspect local storage directly: DTensor.to_local() is
                # autograd-aware and checkpoint refresh needs only alias metadata.
                cached_local = (
                    cached._local_tensor if isinstance(cached, DTensor) else cached
                )
                value_local = (
                    value._local_tensor if isinstance(value, DTensor) else value
                )
                if torch._C._is_alias_of(cached_local, value_local):
                    same_view = (
                        cached_local.shape == value_local.shape
                        and cached_local.stride() == value_local.stride()
                        and cached_local.storage_offset()
                        == value_local.storage_offset()
                    )
                    if not same_view:
                        # A hook changed the exported view over live model storage.
                        # Copying into the old view would mutate the model itself;
                        # detach the cache once, then reuse its storage on later saves.
                        self.cached_state_dict[key] = value.detach().clone()
                else:
                    cached.copy_(value)
        return self.cached_state_dict

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate full checkpoint keys and shapes without copying or communicating."""
        # Validate the combined view before any part copies weights. strict=False
        # below permits other PP parts and auxiliary checkpoint categories, but
        # must not permit missing local model state or late shape failures.
        missing = self.cached_state_dict.keys() - state_dict.keys()
        if missing:
            raise ValueError(f"Model checkpoint is missing keys: {sorted(missing)}")
        for key, expected in self.cached_state_dict.items():
            if not isinstance(expected, torch.Tensor):
                continue
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Model checkpoint {key!r} must be a tensor")
            # PyTorch accepts old one-element checkpoints for scalar state.
            legacy_scalar = expected.ndim == 0 and value.shape == torch.Size([1])
            if value.shape != expected.shape and not legacy_scalar:
                raise ValueError(
                    f"Model checkpoint {key!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        # Each PP part consumes its own keys from the combined model state.
        # Other parts' keys are ignored; checkpoint state categories are separate.
        state_dict = load_expert_state(
            load_tensor_parallel_state(state_dict, model_parts=self.model),
            shard_map=self._ep_shards,
            model_parts=self.model,
        )
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()
