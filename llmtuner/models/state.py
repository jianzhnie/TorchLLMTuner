"""Model-owned external state rules, independent of checkpoint storage.

Formats describe keys, expert storage and any tensor conversion. Checkpointers
own IO, distributed layout collection and destination state. The default format
preserves a model whose native keys already match its external checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass(frozen=True, slots=True)
class ExpertStorageLayout:
    """Describe individual external experts presented as packed read targets."""

    num_experts: int
    native_ep_layout: bool


class ModelStateFormat:
    """Small train/export contract; identity unless a model declares conversion."""

    grouped_experts = False
    coordinates_conversion = False
    split_expert_stacks = False
    separate_export_indices = False
    export_phase = "HF export expert mapping"

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
