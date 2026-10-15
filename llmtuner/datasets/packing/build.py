"""The packing entry points and the row-length policy they share.

``build_concat_then_split_packing`` and ``build_first_fit_packing`` turn a
dataset recipe into a Grain graph that emits fixed-length rows; both fill every
token feature to ``context.num_tokens_per_batch``, and ``text_packing_lengths`` is the
one place that length is spelled out.
"""

from __future__ import annotations

from functools import partial

import grain.python as grain

from ...components.loss import IGNORE_INDEX
from ..composition import (
    DatasetRecipe,
    as_iter_dataset,
    build_dataset,
)
from ..contracts import (
    DatasetBuildContext,
    DatasetIterationPolicy,
    require_positive_integer,
)
from .conversions import (
    packing_output_is_full,
    packing_output_to_text_sequence,
    text_sequence_to_packing_input,
)
from .iterators import (
    DocumentAwareConcatThenSplitIterDataset,
    SplitTextSequenceDocuments,
)


def text_packing_lengths(*, num_tokens_per_row: int) -> dict[str, int]:
    """The per-feature row length every packing node fills to.

    All four token features are packed to the same length; the dict is what the
    Grain packing iterators take, and building it in one place is what keeps the
    two packers' idea of a row identical.
    """
    return {
        feature: num_tokens_per_row
        for feature in ("input_ids", "labels", "positions", "padding_mask")
    }


def build_concat_then_split_packing(
    recipe: DatasetRecipe,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
) -> grain.IterDataset:
    """Concatenate documents, chunking them into fixed-length rows."""
    dataset_graph = build_dataset(
        recipe,
        context=context,
        iteration_policy=iteration_policy,
    )
    if context.max_num_documents is not None:
        dataset_graph = as_iter_dataset(dataset_graph, context=context)
        return DocumentAwareConcatThenSplitIterDataset(
            dataset_graph,
            max_num_documents_per_row=context.max_num_documents,
            max_context_length=context.max_context_length,
            num_tokens_per_row=context.num_tokens_per_batch,
        )
    dataset_graph = dataset_graph.map(
        partial(
            text_sequence_to_packing_input,
            max_context_length=context.max_context_length,
        )
    )
    dataset_graph = as_iter_dataset(dataset_graph, context=context)
    dataset_graph = grain.experimental.ConcatThenSplitIterDataset(
        dataset_graph,
        length_struct=text_packing_lengths(
            num_tokens_per_row=context.num_tokens_per_batch
        ),
    )
    # Finite validation must include the last document tokens even when they
    # do not fill a row. Conversion masks Grain's padding before collation.
    if iteration_policy.repeat:
        dataset_graph = dataset_graph.filter(packing_output_is_full)
    return dataset_graph.map(
        partial(
            packing_output_to_text_sequence,
            max_context_length=context.max_context_length,
        )
    )


def build_first_fit_packing(
    recipe: DatasetRecipe,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
    num_packing_bins: int = 8,
) -> grain.IterDataset:
    """Pack document chunks no longer than the context window.

    ``num_packing_bins`` is how many candidate rows are kept open; more bins can
    reduce padding, but buffer more samples.
    """
    require_positive_integer("num_packing_bins", num_packing_bins)

    dataset_graph = build_dataset(
        recipe,
        context=context,
        iteration_policy=iteration_policy,
    )
    dataset_graph = as_iter_dataset(dataset_graph, context=context)
    dataset_graph = grain.experimental.FlatMapIterDataset(
        dataset_graph,
        SplitTextSequenceDocuments(
            max_context_length=context.max_context_length,
        ),
    )
    dataset_graph = dataset_graph.map(
        partial(
            text_sequence_to_packing_input,
            max_context_length=context.max_context_length,
        )
    )
    # TODO(data-global-pack-plan): Consider packing before DP sharding so
    # ranks receive similarly filled rows.
    dataset_graph = grain.experimental.FirstFitPackIterDataset(
        dataset_graph,
        length_struct=text_packing_lengths(
            num_tokens_per_row=context.num_tokens_per_batch
        ),
        padding_struct={
            "input_ids": 0,
            "labels": IGNORE_INDEX,
            "positions": 0,
            "padding_mask": True,
        },
        num_packing_bins=num_packing_bins,
        meta_features=("labels", "positions"),
        seed=iteration_policy.seed,
        shuffle_bins=iteration_policy.shuffle,
        max_sequences_per_bin=context.max_num_documents,
    )
    return dataset_graph.map(
        partial(
            packing_output_to_text_sequence,
            max_context_length=context.max_context_length,
        )
    )
