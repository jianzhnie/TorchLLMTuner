"""Packing multimodal documents into fixed-length training rows."""

from __future__ import annotations

from functools import partial
from typing import Any

import grain.python as grain
import numpy as np
import torch

from ...components.loss import IGNORE_INDEX
from ..contracts import DatasetBuildContext, DatasetIterationPolicy, require_positive
from ..graph import (
    DatasetConcat,
    DatasetMix,
    SingleDataset,
    as_iter_dataset,
    build_dataset,
)


def build_multimodal_packing(
    dataset: SingleDataset | DatasetMix | DatasetConcat,
    *,
    context: DatasetBuildContext,
    dataset_iteration_policy: DatasetIterationPolicy,
    num_packing_bins: int = 8,
) -> grain.IterDataset[dict[str, Any]]:
    """Pack whole multimodal documents into fixed-length rows.

    Lives here rather than in ``packing/build.py`` beside the text recipes: this
    node packs media arrays, and naming it from the text module would make the
    text-only path import all of this.

    ``num_packing_bins`` is how many candidate rows are kept open; more bins can
    reduce padding, but retain more media.
    """
    require_positive("num_packing_bins", num_packing_bins)

    dataset_graph = build_dataset(
        dataset,
        context=context,
        dataset_iteration_policy=dataset_iteration_policy,
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
        },
        padding_struct={
            "input_ids": context.tokenizer.pad_id,
            "labels": IGNORE_INDEX,
            "positions": 0,
        },
        num_packing_bins=num_packing_bins,
        meta_features=(
            "labels",
            "positions",
            "pixel_values",
            "pixel_values_videos",
        ),
        seed=dataset_iteration_policy.seed,
        shuffle_bins=dataset_iteration_policy.shuffle,
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
        "pixel_values": sample.get("pixel_values", []),
        "pixel_values_videos": sample.get("pixel_values_videos", []),
    }


def packing_output_to_multimodal_sample(
    packing_output: dict[str, Any],
    *,
    max_context_length: int,
) -> dict[str, Any]:
    """Restore Torch token fields and flatten per-document media lists."""
    padding_mask = np.asarray(packing_output["input_ids_segment_ids"]) == 0
    positions = np.asarray(packing_output["positions"]).copy()
    if np.any(padding_mask):
        first_padding_token = int(np.flatnonzero(padding_mask)[0])
        positions[first_padding_token:] = (
            np.arange(len(positions) - first_padding_token) % max_context_length
        )
    return {
        "input_ids": torch.from_numpy(packing_output["input_ids"]),
        "labels": torch.from_numpy(packing_output["labels"]),
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
