"""Packing multimodal documents into fixed-length training rows."""

from __future__ import annotations

from functools import partial
from typing import Any

import grain.python as grain
import numpy as np
import torch

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


def build_multimodal_packing(
    recipe: DatasetRecipe,
    *,
    context: DatasetBuildContext,
    iteration_policy: DatasetIterationPolicy,
    num_packing_bins: int = 8,
) -> grain.IterDataset[dict[str, Any]]:
    """Pack whole multimodal documents into fixed-length rows.

    Lives here rather than in ``packing/build.py`` beside the text recipes: this
    node packs media arrays, and naming it from the text module would make the
    text-only path import all of this.

    ``num_packing_bins`` is how many candidate rows are kept open; more bins can
    reduce padding, but retain more media.
    """
    require_positive_integer("num_packing_bins", num_packing_bins)

    dataset_graph = build_dataset(
        recipe,
        context=context,
        iteration_policy=iteration_policy,
    )
    dataset_graph = dataset_graph.filter(
        lambda sample: len(sample["input_ids"]) <= context.max_context_length
    )
    dataset_graph = dataset_graph.map(multimodal_sample_to_packing_input)
    dataset_graph = as_iter_dataset(dataset_graph, context=context)
    # TODO(data-global-pack-plan): Consider packing before DP sharding so
    # ranks receive similar text and media work.
    dataset_graph = grain.experimental.FirstFitPackIterDataset(
        dataset_graph,
        length_struct={
            "input_ids": context.num_tokens_per_batch,
            "labels": context.num_tokens_per_batch,
            "positions": context.num_tokens_per_batch,
            "padding_mask": context.num_tokens_per_batch,
        },
        padding_struct={
            "input_ids": context.tokenizer.pad_id,
            "labels": IGNORE_INDEX,
            "positions": 0,
            "padding_mask": True,
        },
        num_packing_bins=num_packing_bins,
        max_sequences_per_bin=context.max_num_documents,
        meta_features=(
            "labels",
            "positions",
            "pixel_values",
            "pixel_values_videos",
        ),
        seed=iteration_policy.seed,
        shuffle_bins=iteration_policy.shuffle,
    )
    return dataset_graph.map(
        partial(
            packing_output_to_multimodal_sample,
            max_context_length=context.max_context_length,
        )
    )


def multimodal_sample_to_packing_input(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert Torch token fields to the arrays expected by Grain packing."""
    return {
        "input_ids": np.asarray(sample["input_ids"]),
        "labels": np.asarray(sample["labels"]),
        "positions": np.asarray(sample["positions"]),
        "padding_mask": np.asarray(
            sample.get("padding_mask", np.zeros(len(sample["input_ids"]), dtype=bool)),
            dtype=bool,
        ),
        "pixel_values": sample.get("pixel_values", []),
        "pixel_values_videos": sample.get("pixel_values_videos", []),
    }


def packing_output_to_multimodal_sample(
    packing_output: dict[str, Any],
    *,
    max_context_length: int,
) -> dict[str, Any]:
    """Mask packing padding, restore tensors, and flatten document media."""
    require_positive_integer("max_context_length", max_context_length)
    packing_padding = np.asarray(packing_output["input_ids_segment_ids"]) == 0
    padding_mask = packing_padding.copy()
    if "padding_mask" in packing_output:
        padding_mask |= np.asarray(packing_output["padding_mask"], dtype=bool)
    labels = np.asarray(packing_output["labels"]).copy()
    labels[padding_mask] = IGNORE_INDEX
    positions = np.asarray(packing_output["positions"]).copy()
    if np.any(packing_padding):
        first_padding_token = int(np.flatnonzero(packing_padding)[0])
        positions[first_padding_token:] = (
            np.arange(len(positions) - first_padding_token) % max_context_length
        )
    return {
        "input_ids": torch.from_numpy(packing_output["input_ids"]),
        "labels": torch.from_numpy(labels),
        "positions": torch.from_numpy(positions),
        "padding_mask": torch.from_numpy(padding_mask),
        "pixel_values": [
            image
            for document_images in packing_output["pixel_values"]
            for image in document_images
        ],
        "pixel_values_videos": [
            video
            for document_videos in packing_output["pixel_values_videos"]
            for video in document_videos
        ],
    }
