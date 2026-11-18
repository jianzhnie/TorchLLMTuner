"""Checkpoint state view over model parts, with stable tensor storage.

Adapted from torchtitan ``components/checkpointer/base.py``. Combine local model
parts and gather/slice EP expert state without depending on manager policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from ....accelerator.collectives import synchronize_local_errors
from ....models.registered_state import (
    named_extra_state_owners,
    named_persistent_buffers,
)
from ....parallel.assembly import (
    ModelPartLayout,
    capture_model_part_layouts,
    validate_model_part_layouts,
)
from ....parallel.expert_parallel.checkpointing import (
    expert_shard_map,
)
from .chunked_tensor import ChunkedCheckpointTensor, chunk_layout
from .layout_conversion import (
    export_parallel_state,
    resolve_restore_layout,
    restore_parallel_state,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = ["ModelCheckpointState"]


def _local_error_boundary(
    phase: str, group: ProcessGroup | None
) -> AbstractContextManager[None]:
    """Coordinate local validation only when the caller supplied a group."""
    if group is None:
        return nullcontext()
    return synchronize_local_errors(phase, group=group)


@dataclass(frozen=True)
class TensorMetadata:
    shape: torch.Size
    dtype: torch.dtype
    chunks: Any = None


class ModelCheckpointState(Stateful):
    """A ``Stateful`` view over one module or a list of them.

    Serves two purposes:

    1. **Flattening.** Combines the state dicts of several modules (individual
       local model parts) into one flat view, so the
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
        - ``load_state_dict`` releases the cache before writing the underlying
          modules, including when layout conversion or a load hook later fails.
          The next save starts a new stable-storage cache generation; failed
          loads are not rolled back.
        - The module tree must not be structurally modified (keys changing,
          tensor references replaced) after wrapping, or the cache goes stale.
          Parameter bindings are captured through ``ModelPartLayout`` and
          persistent buffers and extra-state owners are captured locally. All
          are checked before export or restore; rebuild the wrapper after changes.
        - A hook that changes the exported view over aliased live storage forces
          one cache allocation to avoid mutating model tensors during refresh.
        - ``runtime_layouts`` shares final assembly ownership when supplied.
          Otherwise a standalone wrapper captures local bindings itself.
    """

    def __init__(
        self,
        model: nn.Module | Sequence[nn.Module],
        *,
        initialization_group: ProcessGroup | None = None,
        sharded_tensor_parallel: bool = False,
        sharded_expert_parallel: bool = False,
        runtime_layouts: tuple[ModelPartLayout, ...] = (),
    ) -> None:
        self.sharded_tensor_parallel = sharded_tensor_parallel
        self.sharded_expert_parallel = sharded_expert_parallel
        self._validation_group = initialization_group
        # Managers opt into a world boundary before any EP collection. Direct
        # state wrappers can be rank-local and retain that default contract.
        with _local_error_boundary(
            "Checkpoint model preparation", initialization_group
        ):
            parts = (model,) if isinstance(model, nn.Module) else tuple(model)
            if any(not isinstance(part, nn.Module) for part in parts):
                raise TypeError("Every checkpoint model part must be an nn.Module")
            # Own the collection: caller-side list edits must not invalidate the
            # EP shard map or change the set of modules being checkpointed.
            # Compilation changes execution, not checkpoint ownership or names.
            self.model = [
                part._orig_mod if isinstance(part, OptimizedModule) else part
                for part in parts
            ]
            self._layout_models = parts if runtime_layouts else tuple(self.model)
            self._model_layouts = tuple(runtime_layouts) or capture_model_part_layouts(
                self._layout_models
            )
            validate_model_part_layouts(self._layout_models, self._model_layouts)
            self._layout_snapshots = tuple(
                layout.snapshot() for layout in self._model_layouts
            )
            self._bound_buffers = tuple(
                dict(named_persistent_buffers(part)) for part in self.model
            )
            self._bound_extra_state_owners = tuple(
                dict(named_extra_state_owners(part)) for part in self.model
            )
            # One identity may be registered as a parameter in one part and
            # as a buffer in another. Track both in the same ownership domain.
            owners: dict[int, int] = {}
            for kind, snapshots in (
                (
                    "parameter",
                    tuple(item.parameters for item in self._layout_snapshots),
                ),
                ("persistent buffer", self._bound_buffers),
                ("extra-state", self._bound_extra_state_owners),
            ):
                for index, bindings in enumerate(snapshots):
                    for value in bindings.values():
                        previous = owners.setdefault(id(value), index)
                        if previous != index:
                            raise ValueError(
                                f"Checkpoint {kind} ownership overlaps across model "
                                f"parts {previous} and {index}"
                            )
            # EP-sharded expert weights are rank-heterogeneous under one FQN;
            # they ride through this wrapper as all-gathered full tensors and are
            # sliced back per rank on load (parallel/expert_parallel/checkpointing.py).
            self._ep_shards = expert_shard_map(self.model)
        self.cached_state_dict = self.get_state_dict(
            initialization_group=initialization_group
        )
        self.remember_metadata()

    def _validate_model_bindings(self) -> None:
        """Reject replaced state tensors before cached views or EP maps are used."""
        with _local_error_boundary(
            "Checkpoint model binding validation", self._validation_group
        ):
            self._validate_local_model_bindings()

    def _validate_local_model_bindings(self) -> None:
        validate_model_part_layouts(self._layout_models, self._model_layouts)
        for layout, bound in zip(
            self._model_layouts,
            self._layout_snapshots,
            strict=True,
        ):
            bound.validate(layout, consumer="Checkpoint")
        for description, discover, snapshots in (
            ("buffer bindings", named_persistent_buffers, self._bound_buffers),
            (
                "extra-state ownership",
                named_extra_state_owners,
                self._bound_extra_state_owners,
            ),
        ):
            for index, (part, bound) in enumerate(
                zip(self.model, snapshots, strict=True)
            ):
                current = dict(discover(part))
                if current.keys() != bound.keys() or any(
                    current[name] is not value for name, value in bound.items()
                ):
                    raise RuntimeError(
                        f"Model checkpoint {description} changed at part {index}; "
                        "construct a new ModelCheckpointState"
                    )

    def remember_metadata(self) -> None:
        # Keep validation independent of exported storage: full TP/EP exports
        # can be much larger than the rank-local model, and must not stay alive
        # merely to validate another load after the cache has been released.
        self._metadata = {
            key: (
                TensorMetadata(
                    value.shape,
                    value.dtype,
                    chunk_layout(value.chunks)
                    if isinstance(value, ChunkedCheckpointTensor)
                    else None,
                )
                if isinstance(value, torch.Tensor)
                else None
            )
            for key, value in self.cached_state_dict.items()
        }

    def release_cache(self) -> None:
        # Do not clear in place: the caller's loaded state may be this dictionary.
        # Rebuilding here would gather a second full model while the input model
        # and optimizer checkpoint tensors are still live.
        self.cached_state_dict = {}

    def get_state_dict(
        self, *, initialization_group: ProcessGroup | None = None
    ) -> dict[str, Any]:
        flat = self.local_state_dict(initialization_group=initialization_group)
        return export_parallel_state(
            flat,
            model_parts=self.model,
            expert_shards=self._ep_shards,
            sharded_tensor_parallel=self.sharded_tensor_parallel,
            sharded_expert_parallel=self.sharded_expert_parallel,
        )

    def state_dict(self) -> dict[str, Any]:
        # Recompute so hook-produced tensors reflect the current parameters,
        # then merge into the cache without changing storage objects.
        current_state = self.get_state_dict()
        with _local_error_boundary(
            "Checkpoint model cache validation", self._validation_group
        ):
            if current_state.keys() != self._metadata.keys():
                added = sorted(current_state.keys() - self._metadata.keys())
                removed = sorted(self._metadata.keys() - current_state.keys())
                raise RuntimeError(
                    "Model checkpoint keys changed after wrapping; construct a new "
                    f"ModelCheckpointState. Added: {added}; removed: {removed}"
                )
        for key, value in current_state.items():
            cached = self.cached_state_dict.get(key)
            if isinstance(value, ChunkedCheckpointTensor) and isinstance(
                cached, ChunkedCheckpointTensor
            ):
                self.cached_state_dict[key] = cached.refresh_from(value)
                continue
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
        self.remember_metadata()
        return self.cached_state_dict

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate full checkpoint keys and shapes without copying or communicating."""
        # Validate the combined view before any part copies weights. strict=False
        # below permits other model parts and auxiliary checkpoint categories, but
        # must not permit missing local model state or late shape failures.
        self._validate_local_model_bindings()
        missing = self._metadata.keys() - state_dict.keys()
        if missing:
            raise ValueError(f"Model checkpoint is missing keys: {sorted(missing)}")
        for key, expected in self._metadata.items():
            if expected is None:
                continue
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Model checkpoint {key!r} must be a tensor")
            if value.shape != expected.shape:
                raise ValueError(
                    f"Model checkpoint {key!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )
            if expected.chunks is not None and isinstance(
                value, ChunkedCheckpointTensor
            ):
                if value.dtype != expected.dtype:
                    raise ValueError(
                        f"Model checkpoint {key!r} dtype differs from destination"
                    )
                expected_layout = expected.chunks
                actual_layout = chunk_layout(value.chunks)
                if actual_layout != expected_layout:
                    raise ValueError(
                        f"Model checkpoint {key!r} chunk layout differs "
                        "from destination"
                    )

    def local_state_dict(
        self, *, initialization_group: ProcessGroup | None = None
    ) -> dict[str, Any]:
        """Export local parts with the same collision policy for save and restore."""
        self._validate_model_bindings()
        # Hooks may redistribute DTensors: finish their own communication before
        # entering the local merge boundary, and never wrap them as local work.
        export_error: BaseException | None = None
        part_states = []
        try:
            part_states = [model.state_dict() for model in self.model]
        except BaseException as error:
            export_error = error
        flat: dict[str, Any] = {}
        error_group = (
            self._validation_group
            if initialization_group is None
            else initialization_group
        )
        with _local_error_boundary("Checkpoint model state merge", error_group):
            if export_error is not None:
                raise export_error
            for part_index, part_state in enumerate(part_states):
                duplicates = flat.keys() & part_state.keys()
                if duplicates:
                    raise ValueError(
                        f"Model checkpoint keys overlap at part {part_index}: "
                        f"{sorted(duplicates)}. Model parts must use distinct FQNs."
                    )
                flat.update(part_state)
        return flat

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # validate_state_dict includes local binding checks and intentionally
        # performs no collectives. Coordinate its full result before any
        # TP/EP conversion starts, including direct Stateful restore callers.
        with _local_error_boundary(
            "Checkpoint model restore validation", self._validation_group
        ):
            self.validate_state_dict(state_dict)
            restore_layout = resolve_restore_layout(
                state_dict,
                sharded_tensor_parallel=self.sharded_tensor_parallel,
                sharded_expert_parallel=self.sharded_expert_parallel,
            )
        # Sharded layout conversion can copy directly into live destinations,
        # before module load hooks run. Retire cached exports before either can
        # partially mutate state and fail. Replacing the dictionary preserves
        # caller-owned input even when it is the previous cache itself.
        self.release_cache()
        destination = (
            self.local_state_dict() if restore_layout.requires_local_state else None
        )
        local_state = restore_parallel_state(
            state_dict,
            local_state=destination,
            model_parts=self.model,
            expert_shards=self._ep_shards,
            layout=restore_layout,
        )
        # Each local model part consumes its keys from the combined model state.
        # Other parts' keys are ignored; checkpoint state categories are separate.
        # Load hooks may communicate, so run them outside the local error
        # boundary. Agree on their outcome before restoring optimizer state.
        restore_error: BaseException | None = None
        try:
            for model in self.model:
                model.load_state_dict(local_state, strict=False)
        except BaseException as error:
            restore_error = error
        with _local_error_boundary(
            "Checkpoint model restore completion", self._validation_group
        ):
            if restore_error is not None:
                raise restore_error
