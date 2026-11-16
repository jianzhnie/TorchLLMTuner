"""External safetensors key mapping and storage selection.

Explicit models use their own canonical FQNs and declared parameter aliases.
Native DCP and optimizer recovery remain owned by the existing checkpointer.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from ...accelerator.collectives import synchronize_local_errors
from ...models.registry import resolve_state_format
from ...parallel.assembly import ModelPartLayout
from .core.checkpoint_keys import SAFETENSORS_INDEX
from .core.chunked_tensor import ChunkedCheckpointTensor
from .hf.hf_expert_storage import (
    GroupedExpertHuggingFaceStorageReader,
)
from .hf.hf_storage import IndexedHuggingFaceStorageReader, read_hf_weight_map


def clone_local_state_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Copy owned checkpoint storage without redistributing DTensor layouts.

    DTensor clone dispatch may reduce Partial placements. Export keys can belong
    to only some ranks, so storage cloning must not introduce a collective.
    Explicit shape/stride also preserve uneven-shard metadata without checks.
    """
    if isinstance(tensor, DTensor):
        return DTensor.from_local(
            tensor.to_local().clone(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
            run_check=False,
            shape=tensor.shape,
            stride=tensor.stride(),
        )
    return tensor.clone()


class ModelStateDictAdapter:
    """Map declared model names and coordinate one checkpoint's key coverage.

    ``error_group`` includes every owner contributing to the external state,
    across ranks. It is borrowed for conversion errors and key gathering;
    storage-backend IO remains independently configured. None uses the default
    process group for standalone callers.
    """

    # to_hf prepares reader destinations and coordinates key coverage itself.
    # from_hf remains local for formats without collective tensor conversion.
    coordinates_hf_preparation = True

    def __init__(
        self,
        model_config: Any,
        hf_assets_path: str | None,
        *,
        expert_parallel: bool = False,
        sharded_expert_parallel: bool = True,
        error_group: dist.ProcessGroup | None = None,
        runtime_layouts: tuple[ModelPartLayout, ...] = (),
    ) -> None:
        for name, enabled in (
            ("expert_parallel", expert_parallel),
            ("sharded_expert_parallel", sharded_expert_parallel),
        ):
            if type(enabled) is not bool:
                raise TypeError(f"{name} must be boolean")
        # Includes all owners whose external keys form one checkpoint.
        # The adapter borrows this group; the entry point owns its lifetime.
        self.error_group = error_group
        self._runtime_layouts = tuple(runtime_layouts)
        parameter_names: dict[str, str] = {}
        for layout in self._runtime_layouts:
            if not isinstance(layout, ModelPartLayout):
                raise TypeError("Checkpoint layouts must be ModelPartLayout")
            for path, canonical in layout.state_parameter_names.items():
                previous = parameter_names.setdefault(path, canonical)
                if previous != canonical:
                    raise ValueError(f"Runtime parameter alias conflict: {path!r}")
        self._layout_snapshots = tuple(
            layout.snapshot() for layout in self._runtime_layouts
        )
        self._runtime_parameter_names = MappingProxyType(parameter_names)
        # Composite HF models store the trained decoder's weight ties here.
        self.model_config = getattr(model_config, "text_config", model_config)
        self.state_format = resolve_state_format(
            self.model_config,
            expert_parallel=expert_parallel,
            sharded_expert_parallel=sharded_expert_parallel,
        )
        self.hf_assets_path = hf_assets_path
        self._hf_keys: set[str] | None = None
        self.fqn_to_index_mapping: dict[str, int] | None = None
        self.hf_export_fqn_to_index_mapping: dict[str, int] | None = None

        if hf_assets_path is None:
            return
        index_path = os.path.join(hf_assets_path, SAFETENSORS_INDEX)
        indexed = os.path.isfile(index_path)
        if indexed:
            weight_map = read_hf_weight_map(index_path)
            self._hf_keys = set(weight_map)
            # Preserve filename grouping with the writer's contiguous 1..N IDs.
            shard_indices = {
                filename: shard_index
                for shard_index, filename in enumerate(
                    sorted(set(weight_map.values())), 1
                )
            }
            self.fqn_to_index_mapping = {
                key: shard_indices[filename] for key, filename in weight_map.items()
            }
            if self.state_format.separate_export_indices:
                self.hf_export_fqn_to_index_mapping = dict(self.fqn_to_index_mapping)
        else:
            # Read only the header: single-file checkpoints omit tied heads
            # too, and need the same key coverage policy as indexed exports.
            tensor_path = os.path.join(hf_assets_path, "model.safetensors")
            if not os.path.isfile(tensor_path):
                return
            with safe_open(tensor_path, framework="pt") as tensors:
                self._hf_keys = set(tensors.keys())
        self._hf_keys, self.fqn_to_index_mapping = self.state_format.normalize_keys(
            self._hf_keys, self.fqn_to_index_mapping
        )
        if indexed and not self.state_format.separate_export_indices:
            self.hf_export_fqn_to_index_mapping = self.fqn_to_index_mapping

    @property
    def coordinates_state_conversion(self) -> bool:
        """Whether format conversion owns its rank-coordinated tensor boundary."""
        return self.state_format.coordinates_conversion

    def _validate_runtime_layouts(self) -> None:
        for layout, snapshot in zip(
            self._runtime_layouts, self._layout_snapshots, strict=True
        ):
            snapshot.validate(layout, consumer="HF checkpoint adapter")

    def _map_state_keys(
        self, state: dict[str, Any], *, to_hf: bool
    ) -> dict[str, Any]:
        """Map canonical runtime names and external names without losing aliases."""
        mapped = {}
        for key, value in state.items():
            if to_hf:
                key = self._runtime_parameter_names.get(key, key)
            target = self.state_format.map_key(key, to_hf=to_hf)
            if not to_hf:
                target = self._runtime_parameter_names.get(target, target)
            if target in mapped:
                raise ValueError(f"HF checkpoint key mapping collision: {target!r}")
            mapped[target] = value
        return mapped

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # Agree before model-owned conversion can enter tensor collectives.
        with synchronize_local_errors(
            "HF export layout validation", group=self.error_group
        ):
            self._validate_runtime_layouts()
        with (
            nullcontext()
            if self.state_format.coordinates_conversion
            else synchronize_local_errors(
                "HF destination format conversion", group=self.error_group
            )
        ):
            state_dict = self.state_format.convert_state(
                state_dict, to_hf=True, error_group=self.error_group
            )
        # Only dictionary mapping is local. Model-owned tensor conversion above and
        # the global key coverage gather below stay outside this boundary.
        with synchronize_local_errors("HF export key mapping", group=self.error_group):
            hf_state = self._map_state_keys(state_dict, to_hf=True)
            tied_embeddings = getattr(self.model_config, "tie_word_embeddings", False)
        if self._hf_keys is not None:
            requested = set(hf_state)
            if dist.is_available() and dist.is_initialized():
                # Validate all owners' key coverage while returning only this
                # rank's tensor dictionary.
                keys_by_rank: list[set[str] | None] = [None] * dist.get_world_size(
                    self.error_group
                )
                dist.all_gather_object(keys_by_rank, requested, group=self.error_group)
                requested = set().union(*keys_by_rank)
            missing = self._hf_keys - requested
            unexpected = requested - self._hf_keys
            # A tied lm_head is legitimately omitted from a HF checkpoint.
            if (
                tied_embeddings
                and "lm_head.weight" not in self._hf_keys
            ):
                unexpected.discard("lm_head.weight")
                hf_state.pop("lm_head.weight", None)
            if missing or unexpected:
                raise ValueError(
                    "HF checkpoint keys do not exactly cover the llmtuner model: "
                    f"missing={sorted(missing)[:8]}, "
                    f"unexpected={sorted(unexpected)[:8]}"
                )
        if tied_embeddings:
            # All ranks with tied configuration participate, even without either
            # key. DTensor clone preserves placements locally.
            with synchronize_local_errors(
                "HF export tied weights", group=self.error_group
            ):
                if (
                    "model.embed_tokens.weight" in hf_state
                    and "lm_head.weight" in hf_state
                ):
                    # Safetensors cannot serialize aliases within one shard.
                    # HF restores parameter sharing from its configuration.
                    hf_state["lm_head.weight"] = clone_local_state_tensor(
                        hf_state["lm_head.weight"]
                    )
        return hf_state

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # Local formats retain standalone single-rank key mapping. Collective
        # formats must agree on mapping failures before entering tensor work.
        with (
            synchronize_local_errors("HF import key mapping", group=self.error_group)
            if self.state_format.coordinates_conversion
            else nullcontext()
        ):
            self._validate_runtime_layouts()
            hf_state_dict = dict(hf_state_dict)
            if (
                "lm_head.weight" not in hf_state_dict
                and "model.embed_tokens.weight" in hf_state_dict
                and getattr(self.model_config, "tie_word_embeddings", False)
            ):
                hf_state_dict["lm_head.weight"] = hf_state_dict[
                    "model.embed_tokens.weight"
                ]
            state = self._map_state_keys(hf_state_dict, to_hf=False)
        return self.state_format.convert_state(
            state, to_hf=False, error_group=self.error_group
        )

    def to_hf_export(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Produce standard HF expert keys rather than virtual reader templates."""
        state = self.to_hf(state_dict)
        if not self.state_format.split_expert_stacks:
            return state
        result = {}
        pending = []
        with synchronize_local_errors(
            self.state_format.export_phase, group=self.error_group
        ):
            for key, value in state.items():
                expert_keys = self.state_format.export_keys(
                    key, value.shape if isinstance(value, torch.Tensor) else None
                )
                targets = (key,) if expert_keys is None else expert_keys
                for target in targets:
                    if target in result:
                        raise ValueError(f"HF export key collision: {target}")
                    result[target] = value if expert_keys is None else None
                if expert_keys is not None:
                    pending.append((targets, value))
        for targets, value in pending:
            if isinstance(value, ChunkedCheckpointTensor):
                for target in targets:
                    del result[target]
                chunks_by_target: dict[
                    str, list[tuple[torch.Size, torch.Tensor]]
                ] = {}
                for offset, local in value.chunks:
                    for index in range(local.shape[0]):
                        target = targets[offset[0] + index]
                        chunks_by_target.setdefault(target, []).append(
                            (offset[1:], local[index])
                        )
                for target, chunks in chunks_by_target.items():
                    result[target] = ChunkedCheckpointTensor(value.shape[1:], chunks)
                continue
            for target, weight in zip(targets, value.unbind(0), strict=True):
                result[target] = weight
        return result

    def get_hf_storage_reader(
        self,
        path: str,
        *,
        thread_count: int | None = None,
    ) -> IndexedHuggingFaceStorageReader:
        if thread_count is None:
            # Two workers overlap shard reads without heavy storage pressure.
            thread_count = 2
        layout = self.state_format.reader_layout()
        if layout is not None:
            return GroupedExpertHuggingFaceStorageReader(
                path,
                layout=layout,
                thread_count=thread_count,
            )
        return IndexedHuggingFaceStorageReader(path, thread_count=thread_count)
