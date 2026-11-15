"""Model-owned external state rules, independent of checkpoint storage.

Formats describe keys, expert storage and any tensor conversion. Checkpointers
own IO, distributed layout collection and destination state. The default format
preserves a model whose native keys already match its external checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass(frozen=True, slots=True)
class SourceLayoutMigration:
    """Pure rules for full CPU source-state migration before live restoration.

    Callbacks inspect keys or convert isolated state dictionaries. They must
    not communicate or mutate live model/optimizer state; storage and placement
    remain owned by the checkpoint loader.
    """

    name: str
    requires_migration: Callable[[Iterable[str]], bool]
    convert_model: Callable[[dict[str, Any]], dict[str, Any]]
    convert_optimizer: Callable[[dict[str, Any]], dict[str, Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Source layout migration requires a diagnostic name")
        if not all(
            callable(callback)
            for callback in (
                self.requires_migration,
                self.convert_model,
                self.convert_optimizer,
            )
        ):
            raise TypeError("Source layout migration rules must be callable")


@dataclass(frozen=True, slots=True)
class ExpertTensorSlice:
    """One external matrix's slot in an [expert, projection rows, input] stack."""

    target: str
    expert_index: int
    projection_index: int
    projection_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Expert slice target must be a nonempty key")
        if (
            type(self.expert_index) is not int
            or self.expert_index < 0
            or type(self.projection_count) is not int
            or self.projection_count <= 0
            or type(self.projection_index) is not int
            or not 0 <= self.projection_index < self.projection_count
        ):
            raise ValueError("Expert slice indices disagree with projection geometry")


@dataclass(frozen=True, slots=True)
class ExpertStorageLayout:
    """Describe individual external experts presented as packed read targets."""

    num_experts: int
    map_source_key: Callable[[str], ExpertTensorSlice | None]
    validate_source_key: Callable[[str], None]

    def __post_init__(self) -> None:
        if type(self.num_experts) is not int or self.num_experts <= 0:
            raise ValueError("Expert storage layout requires a positive expert count")
        if not callable(self.map_source_key) or not callable(self.validate_source_key):
            raise TypeError("Expert storage layout requires model-owned key rules")


class ModelStateFormat:
    """Small train/export contract; identity unless a model declares conversion."""

    grouped_experts = False
    coordinates_conversion = False
    split_expert_stacks = False
    separate_export_indices = False
    export_phase = "HF export expert mapping"

    def source_layout_migration(self) -> SourceLayoutMigration | None:
        """Optionally declare full-source native checkpoint layout conversion."""
        return None

    def supports_ep_initial_load(
        self, *, tp_degree: int, sharded_tensor_parallel: bool
    ) -> bool:
        """Declare whether external weights can populate converted EP storage.

        A format without an explicit expert mapping cannot initialize native
        EP parameters. This describes format/layout support, not IO or memory
        capacity; the checkpointer still validates destinations when loading.
        """
        return False

    def normalize_keys(
        self, keys: Iterable[str], indices: Mapping[str, int] | None
    ) -> tuple[set[str], dict[str, int] | None]:
        return set(keys), None if indices is None else dict(indices)

    def map_key(self, key: str, *, to_hf: bool) -> str:
        return key

    def convert_state(
        self,
        state: dict[str, Any],
        *,
        to_hf: bool,
        error_group: ProcessGroup | None = None,
    ) -> dict[str, Any]:
        """Convert format; collective-capable formats use the supplied peers."""
        return state

    def export_keys(
        self, key: str, shape: Sequence[int] | None
    ) -> tuple[str, ...] | None:
        """Return split expert destinations, or None to retain a tensor as-is."""
        return None

    def reader_layout(self) -> ExpertStorageLayout | None:
        return None
