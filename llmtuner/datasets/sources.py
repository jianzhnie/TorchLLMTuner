"""Storage adapters for Grain datasets.

Vendored from torchtitan ``components/data/sources.py``. Each source is a plain
dataclass describing *what* to read; :func:`build_source` -- a free function, not
a method -- is what reads it. That split is the same one every other config in
llmtuner uses (``llmtuner/config/``: configs are descriptions, not builders), and
it is what lets a dataset catalog hold sources at module scope without touching
the filesystem or the Hub: constructing one resolves nothing.

Random-access sources leave DP sharding to composition.build_dataset, after
shuffle. Streaming sources use split_dataset_by_node before window shuffle.
Built random-access objects and Grain streams pass through unchanged. Unsupported
source objects raise TypeError at construction.

"""

from __future__ import annotations

import glob
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import datasets
import grain.python as grain
from datasets.distributed import split_dataset_by_node

from .contracts import DatasetIterationPolicy

__all__ = [
    "DatasetSource",
    "HuggingFaceRandomAccessSource",
    "HuggingFaceStreamingSource",
    "IndexedJsonlSource",
    "RandomAccessDataSource",
    "build_source",
]


@runtime_checkable
class RandomAccessDataSource(Protocol):
    """Finite data source addressable by integer index."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


@dataclass(kw_only=True)
class IndexedJsonlSource:
    """Provides random access to JSONL rows through compact byte offsets.

    Accepts either a single pattern or a list of them; several patterns resolve
    to one index over the union of the matched files.
    """

    patterns: tuple[str, ...]
    """File globs to index. Every pattern must match at least one file."""

    def __post_init__(self) -> None:
        if isinstance(self.patterns, str):
            self.patterns = (self.patterns,)
        else:
            self.patterns = tuple(self.patterns)


class IndexedJsonlDataSource:
    """The offset index :func:`build_source` builds for an ``IndexedJsonlSource``.

    Separate from the source so the index is built once per ``build_source()``
    call and not carried around in a dataclass that is compared and hashed as
    configuration.
    """

    def __init__(self, patterns: tuple[str, ...]) -> None:
        self._paths = file_patterns_to_paths(patterns)
        self._path_ids = array("I")
        self._byte_offsets = array("Q")
        # TODO(data-jsonl-sidecar): Startup rescans every JSONL file per rank and
        # worker. Build one validated offset index that all processes can
        # memory-map.
        for path_id, path in enumerate(self._paths):
            with open(path, "rb") as file:
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    if line.strip():
                        self._path_ids.append(path_id)
                        self._byte_offsets.append(offset)

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path = self._paths[self._path_ids[index]]
        with open(path, "rb") as file:
            file.seek(self._byte_offsets[index])
            row = json.loads(file.readline())
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row {index} in {path!r} must be a JSON object; "
                f"got {type(row).__name__}"
            )
        return row


def reject_duplicated_hf_fields(load_dataset_kwargs: dict[str, Any]) -> None:
    """Refuse kwargs that shadow the sources' first-class Hugging Face fields."""
    duplicated = {"path", "split", "name", "revision", "streaming"} & (
        load_dataset_kwargs.keys()
    )
    if duplicated:
        raise ValueError(
            f"first-class Hugging Face fields repeated in kwargs: {sorted(duplicated)}"
        )


@dataclass(kw_only=True)
class _HuggingFaceSource:
    """Shared inert Hugging Face loading options and explicit-field validation."""

    path: str
    split: str
    name: str | None = None
    revision: str | None = None
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reject_duplicated_hf_fields(self.load_dataset_kwargs)


@dataclass(kw_only=True)
class HuggingFaceRandomAccessSource(_HuggingFaceSource):
    """Provides random access to a materialized Hugging Face dataset."""


@dataclass(kw_only=True)
class HuggingFaceStreamingSource(_HuggingFaceSource):
    """Provides a DP-sharded Hugging Face stream with cursor checkpointing.

    The dataset is *not* loaded here. A dataset catalog holds these at module
    scope, so loading in ``__init__`` would make importing a catalog reach the
    Hub; :func:`build_source` is what resolves and splits the stream.
    """


DatasetSource: TypeAlias = (
    IndexedJsonlSource
    | HuggingFaceRandomAccessSource
    | HuggingFaceStreamingSource
    | RandomAccessDataSource
    | grain.IterDataset
)
"""An inert source description or an already readable dataset source."""


def build_source(
    source: DatasetSource,
    *,
    iteration_policy: DatasetIterationPolicy,
) -> RandomAccessDataSource | grain.IterDataset:
    """Resolve ``source`` into something Grain can read.

    A source slot takes either a description -- one of the three classes above --
    or something already built. The second form is the escape hatch for a caller
    that has its own random-access object or a Grain stream in hand: there is no
    ``build()`` to override any more, so without it every source would have to be
    one of the three, and the tests that hold raw rows in memory would have to
    write themselves a JSONL file first.

    The policy argument is passed by every caller but read only by the streaming
    source. The others ignore it deliberately: they are random-access, and the DP
    slice is taken later, after the shuffle, by
    :func:`~llmtuner.datasets.composition.build_dataset`. Sharding here as well would
    drop rows.
    """
    if isinstance(source, IndexedJsonlSource):
        return IndexedJsonlDataSource(source.patterns)
    if isinstance(source, HuggingFaceRandomAccessSource | HuggingFaceStreamingSource):
        # Descriptions are mutable: recheck options at the point of use so
        # edits after construction cannot bypass the explicit-field contract.
        reject_duplicated_hf_fields(source.load_dataset_kwargs)
        streaming = isinstance(source, HuggingFaceStreamingSource)
        dataset = datasets.load_dataset(
            source.path,
            name=source.name,
            split=source.split,
            revision=source.revision,
            streaming=streaming,
            **source.load_dataset_kwargs,
        )
        if not streaming:
            if not isinstance(dataset, datasets.Dataset):
                raise TypeError(
                    "random-access Hugging Face source requires one Dataset; "
                    f"got {type(dataset).__qualname__}"
                )
            return dataset
        if not isinstance(dataset, datasets.IterableDataset):
            raise TypeError(
                "streaming Hugging Face source requires one IterableDataset; "
                f"got {type(dataset).__qualname__}"
            )
        if not hasattr(dataset, "state_dict") or not hasattr(
            dataset, "load_state_dict"
        ):
            raise TypeError(
                "Hugging Face streaming source does not support exact resume"
            )
        return HuggingFaceStreamingIterDataset(
            # Split before the shuffle window, not after: every rank must draw
            # from its own stream, or two ranks would shuffle the same rows
            # into different orders and train on the same data twice.
            split_dataset_by_node(
                dataset,
                rank=iteration_policy.dp_rank,
                world_size=iteration_policy.dp_world_size,
            ),
            repeat=iteration_policy.repeat,
            shuffle=iteration_policy.shuffle,
        )
    if isinstance(source, str | bytes):
        raise TypeError(
            "dataset source cannot be a string or bytes; use IndexedJsonlSource "
            "or a Hugging Face source description for dataset paths"
        )
    if isinstance(source, RandomAccessDataSource) or isinstance(
        source, grain.IterDataset
    ):
        return source
    raise TypeError(f"unhandled source type {type(source).__qualname__}")


class HuggingFaceStreamingIterDataset(grain.IterDataset):
    """The built node :func:`build_source` produces for a
    ``HuggingFaceStreamingSource``.

    Separate from the source so the source can stay an inert dataclass. This
    node has only one parent-adjacent job -- hand out a cursor iterator -- and
    the cursor is what carries the checkpointable read position.
    """

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
    ) -> None:
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        super().__init__()

    def __iter__(self) -> grain.DatasetIterator:
        return HuggingFaceCursorIterator(
            self._dataset,
            repeat=self._repeat,
            shuffle=self._shuffle,
        )


def file_patterns_to_paths(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve patterns in order, sorting matches within each pattern.

    Every pattern must match at least one file. Duplicate resolved paths are
    rejected so one file cannot be indexed twice.
    """
    if not patterns:
        raise ValueError("JSONL source requires at least one file pattern")
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        for match in matches:
            if not Path(match).is_file():
                raise ValueError(f"JSONL pattern matched a non-file path: {match!r}")
        paths.extend(str(Path(match).resolve()) for match in matches)
    if len(paths) != len(set(paths)):
        raise ValueError("patterns resolve to the same file more than once")
    return tuple(paths)


class HuggingFaceCursorIterator(grain.DatasetIterator):
    """Exposes a Hugging Face streaming cursor to Grain checkpoint recursion."""

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        self._epoch = 0
        self._initial_state = dataset.state_dict()
        self._iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        self._assert_not_closed()
        try:
            return next(self._iterator)
        except StopIteration:
            if not self._repeat:
                raise
            epoch = self._epoch + 1
            if self._shuffle:
                self._dataset.set_epoch(epoch)
            self._dataset.load_state_dict(self._initial_state)
            self._iterator = iter(self._dataset)
            self._epoch = epoch
            return next(self._iterator)

    def get_state(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "hf": self._dataset.state_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("streaming checkpoint state must be a dictionary")
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("streaming checkpoint epoch must be a nonnegative integer")
        if "hf" not in state:
            raise ValueError("streaming checkpoint state is missing the 'hf' field")
        hf_state = state["hf"]
        if not isinstance(hf_state, dict):
            raise ValueError("streaming checkpoint 'hf' field must be a dictionary")
        if self._shuffle:
            self._dataset.set_epoch(epoch)
        self._dataset.load_state_dict(hf_state)
        self._iterator = iter(self._dataset)
        self._epoch = epoch
