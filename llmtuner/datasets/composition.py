"""Build Grain graphs from composable dataset recipes.

Adapted from torchtitan ``components/data/dataset.py``. SingleDataset describes
one source and its row transforms; DatasetMix interleaves weighted children;
DatasetConcat joins finite map-style children. Builders receive runtime values
through DatasetBuildContext and ordering/sharding through DatasetIterationPolicy.

Map graphs apply pre-filters, processing, and post-filters before global shuffle,
DP sharding, and repeat. Grain map filters preserve indexed slots and mark
rejected rows as None, so disjoint index slices do not guarantee equal numbers
of usable rows. Streaming sources handle ownership in build_source and shuffle
raw rows before processing. Packing and loaders consume the resulting graphs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, TypeAlias, TypeVar, cast

import grain.python as grain

from .contracts import (
    DatasetBuildContext,
    DatasetIterationPolicy,
    GrainDataset,
    SampleProcessorFactory,
)
from .sources import DatasetSource, RandomAccessDataSource, build_source

__all__ = [
    "DatasetConcat",
    "DatasetMix",
    "DatasetRecipe",
    "SingleDataset",
    "WeightedDataset",
    "as_iter_dataset",
    "build_dataset",
    "is_not_none",
]

_DatasetT = TypeVar("_DatasetT", grain.MapDataset, grain.IterDataset)


@dataclass(frozen=True, kw_only=True, slots=True)
class SingleDataset:
    """One dataset built from a source and row-level transforms.

    `pre_filters` run before `processor`; `post_filters` run afterward.
    This node owns the leaf shuffle, repeat, and effective-DP sharding.
    """

    source: DatasetSource
    pre_filters: tuple[Callable[[Any], bool], ...] = ()
    processor: SampleProcessorFactory | None = None
    """Factory called once per build as ``processor(context=context)``.

    Use a processor class directly or a partial binding processor options.
    The build context always comes from this graph's caller.
    """
    post_filters: tuple[Callable[[Any], bool], ...] = ()


def is_not_none(sample: Any) -> bool:
    """``post_filters`` entry for a processor that drops unusable rows.

    Every processor in this package signals "drop this row" by returning
    ``None`` (too long, missing field, unrenderable media), so the filter that
    acts on that signal is the same one everywhere. Named rather than a lambda
    so the recipes that use it read as one contract, and so the predicate stays
    picklable.
    """
    return sample is not None


def as_iter_dataset(
    dataset: GrainDataset, *, context: DatasetBuildContext
) -> grain.IterDataset:
    """The graph as an ``IterDataset``, converting only if it is map-style.

    Streaming transforms (``ConcatThenSplit``, ``FlatMap``, window shuffle) and
    ``IterDataset.mix`` need an iter graph; a map-style one has to be read into
    one first, and ``read_options`` decides how many rows are read concurrently.
    Passing an iter graph through unchanged is what makes this usable on either
    kind of child.
    """
    if isinstance(dataset, grain.MapDataset):
        return dataset.to_iter_dataset(read_options=context.read_options)
    return dataset


def build_dataset(
    node: DatasetRecipe,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    """Build one node of a Grain dataset graph from its description.

    Dispatches on the concrete node class. Add new node kinds to both this
    dispatcher and DatasetRecipe so builders share the same accepted types.
    """
    match node:
        case SingleDataset():
            builder = build_single_dataset
        case DatasetMix():
            builder = build_dataset_mix
        case DatasetConcat():
            builder = build_dataset_concat
        case _:
            raise TypeError(f"unhandled dataset type {type(node).__qualname__}")
    return builder(node, context=context, iteration_policy=iteration_policy)


def build_single_dataset(
    node: SingleDataset,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    source = build_source(
        node.source,
        iteration_policy=iteration_policy,
    )
    if isinstance(source, RandomAccessDataSource):
        dataset: GrainDataset = grain.MapDataset.source(source)
    elif isinstance(source, grain.IterDataset):
        dataset = source
    else:
        raise TypeError("source must be a RandomAccessDataSource or grain.IterDataset")

    if isinstance(dataset, grain.MapDataset):
        dataset = apply_sample_transforms(
            node, dataset, context=context, seed=iteration_policy.seed
        )
        return apply_iteration_policy(dataset, iteration_policy)
    # Shuffle raw stream rows.
    if iteration_policy.shuffle:
        dataset = grain.experimental.WindowShuffleIterDataset(
            dataset,
            window_size=iteration_policy.streaming_shuffle_buffer_size,
            seed=iteration_policy.seed,
        )

    # Streaming processors use a per-rank seed; map processors share the
    # global indexed seed so DP shards see one consistent transformation.
    return apply_sample_transforms(
        node,
        dataset,
        context=context,
        seed=iteration_policy.seed + iteration_policy.dp_rank,
    )


def apply_sample_transforms(
    node: SingleDataset,
    dataset: _DatasetT,
    *,
    context: DatasetBuildContext,
    seed: int,
) -> _DatasetT:
    """Apply the shared row contract without changing dataset ordering."""
    for filter_fn in node.pre_filters:
        dataset = dataset.filter(filter_fn)
    if node.processor is not None:
        dataset = dataset.random_map(node.processor(context=context), seed=seed)
    for filter_fn in node.post_filters:
        dataset = dataset.filter(filter_fn)
    return dataset


def apply_iteration_policy(
    dataset: grain.MapDataset,
    iteration_policy: DatasetIterationPolicy,
) -> grain.MapDataset:
    """Apply the map-side iteration policy: shuffle, shard, repeat.

    The order is the contract, so it lives in one place rather than at each
    caller (a single-dataset graph and a concatenation of them):

    * ``shuffle`` is global, so every rank computes the same permutation (a
      per-rank shuffle would give each rank a different dataset);
    * then each DP rank takes a contiguous, disjoint slice of that permutation;
    * then ``repeat``. Grain preserves the epoch through sliced map indices, so
      the upstream shuffle re-runs with seed + epoch on each repeat instead of
      re-slicing the same order.
    """
    if iteration_policy.shuffle:
        dataset = dataset.shuffle(seed=iteration_policy.seed)
    dp_world_size = iteration_policy.dp_world_size
    dp_rank = iteration_policy.dp_rank
    if len(dataset) < dp_world_size:
        raise ValueError(
            f"dataset has {len(dataset)} rows, fewer than dp_world_size={dp_world_size}"
        )
    shard_size, remainder = divmod(len(dataset), dp_world_size)
    shard_start = dp_rank * shard_size + min(dp_rank, remainder)
    shard_stop = shard_start + shard_size
    if dp_rank < remainder:
        shard_stop += 1
    dataset = dataset[shard_start:shard_stop]
    if iteration_policy.repeat:
        dataset = dataset.repeat()
    return dataset


@dataclass(frozen=True, kw_only=True, slots=True)
class WeightedDataset:
    """A dataset and its relative selection weight."""

    dataset: DatasetRecipe
    weight: float = 1.0


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetMix:
    """Interleave weighted children.

    `MapDataset.filter` leaves rejected indices as `None`, so the all-map path
    weights attempted indices rather than accepted samples. Otherwise, weights
    select elements emitted by each iterable child. The child defines the
    element: mixing `TextSequence` children weights documents; mixing packed
    fixed-length children weights physical tokens.

    With `repeat=True` the mix is infinite. With `repeat=False` the mix stops
    at the first exhausted child, so larger children are not fully covered.
    """

    datasets: tuple[WeightedDataset, ...]


def build_dataset_mix(
    node: DatasetMix,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
) -> GrainDataset:
    if not node.datasets or any(
        not isinstance(item.weight, Real)
        or isinstance(item.weight, bool)
        or not math.isfinite(item.weight)
        or item.weight <= 0
        for item in node.datasets
    ):
        raise ValueError("DatasetMix requires finite, positive-weight datasets")
    # Grain scales weights by 100 / min(weights). Normalize first so a
    # common tiny or huge scale cannot overflow that intermediate value.
    weights = [float(item.weight) for item in node.datasets]
    minimum = min(weights)
    weights = [weight / minimum for weight in weights]
    if any(not math.isfinite(weight * 100) for weight in weights):
        raise ValueError("DatasetMix weight ratios exceed Grain's numeric range")
    children = [
        build_dataset(
            item.dataset,
            context=context,
            # Inserting or reordering a child reseeds every later child.
            iteration_policy=replace(
                iteration_policy,
                seed=iteration_policy.seed + index,
            ),
        )
        for index, item in enumerate(node.datasets)
    ]
    # TODO(data-token-weighted-mix): Support source weights by token count,
    # not only emitted examples or packed rows. Checkpoint running estimates.
    if all(isinstance(child, grain.MapDataset) for child in children):
        return grain.MapDataset.mix(
            cast(list[grain.MapDataset], children),
            weights=weights,
        )
    children = [as_iter_dataset(child, context=context) for child in children]
    return grain.IterDataset.mix(children, weights=weights)


@dataclass(frozen=True, kw_only=True, slots=True)
class DatasetConcat:
    """Concatenates finite children before global shuffle and DP sharding."""

    datasets: tuple[DatasetRecipe, ...]


DatasetRecipe: TypeAlias = SingleDataset | DatasetMix | DatasetConcat
"""A source/transform recipe; build_dataset turns it into a GrainDataset."""


def build_dataset_concat(
    node: DatasetConcat,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
) -> grain.MapDataset:
    child_iteration_policy = replace(
        iteration_policy,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
    )
    children = [
        build_dataset(
            dataset,
            context=context,
            iteration_policy=child_iteration_policy,
        )
        for dataset in node.datasets
    ]

    if not children or not all(
        isinstance(child, grain.MapDataset) for child in children
    ):
        raise TypeError("DatasetConcat requires map-style children")

    dataset = grain.MapDataset.concatenate(cast(list[grain.MapDataset], children))
    return apply_iteration_policy(dataset, iteration_policy)
