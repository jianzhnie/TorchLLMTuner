"""Multimodal dataset processing for VLM training.

Workflow overview::

    HuggingFace Dataset (streaming)
            |
            v
    +-------------------------------------------------------+
    |  Sample Processor  (MultimodalProcessor)              |
    |                                                       |
    |  1. Parse raw sample (dataset-specific format)        |
    |     e.g. OBELICS interleaved text/images,             |
    |          CC12M text-image pairs                       |
    |                                                       |
    |  2. Process vision: decode image/video bytes,         |
    |     resize to multiples of (patch_size * merge_size), |
    |     normalize with image_mean/std                     |
    |     -> pixel_values: list[Tensor(T,H,W,C)]            |
    |                                                       |
    |  3. Process text: insert vision placeholder tokens    |
    |     <|vision_start|><|image_pad|>...<|vision_end|>    |
    |     into text, then tokenize                          |
    |     -> input_ids: Tensor(num_tokens,)                 |
    |     -> labels: next-token targets, with vision tokens |
    |       masked to ignore_id (-100)                      |
    +-------------------------------------------------------+
            |
            v  (packing: whole documents, first-fit into bins)
    +-------------------------------------------------------+
    |  Sample Packer                                        |
    |  Bin-pack short samples into num_tokens_per_batch-    |
    |  token rows to reduce padding waste                   |
    +-------------------------------------------------------+
            |
            v  GrainDataLoader batches rows (num_tokens_per_batch tokens)
    +-------------------------------------------------------+
    |  Collator  (MultimodalCollator)                       |
    |                                                       |
    |  1. collate_images: for each image Tensor(T,H,W,C),   |
    |     reshape into patches, then concatenate            |
    |     -> pixel_values: (total_patches, patch_dim)       |
    |     -> grid_thw: (N, 3) per-image [T, H', W'] dims    |
    |     (same for videos)                                 |
    |                                                       |
    |  2. collate_text: concatenate whole samples and pad   |
    |     the tail to the token batch                       |
    |     -> input_ids: (num_tokens_per_batch,)             |
    |     -> labels: (num_tokens_per_batch,)                |
    +-------------------------------------------------------+
            |
            v
    Model receives: {input_ids, pixel_values, grid_thw,
                     pixel_values_videos, grid_thw_videos,
                     special_tokens: dict[str, int]}, labels

Vendored from torchtitan ``hf_datasets/multimodal/mm_datasets.py``. The
per-dataset sample function is a constructor argument, which is why
``MULTIMODAL_DATASETS`` binds it with ``functools.partial`` rather than wrapping each
one in its own config subclass.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

from ...components.loss import IGNORE_INDEX
from ...components.tokenizer import MultimodalTokenizer
from ...utils.logging import get_logger
from ..composition import SingleDataset, is_not_none
from ..contracts import DatasetBuildContext, SampleProcessor
from ..sources import HuggingFaceStreamingSource
from .image import calculate_vision_tokens, process_image, resize_to_pixel_budget
from .placeholders import insert_vision_placeholders

logger = get_logger(__name__)

__all__ = [
    "MULTIMODAL_DATASETS",
    "MultimodalProcessor",
]


def process_multimodal_sample(
    texts: list[str | None],
    images: list[bytes | None],
    tokenizer: MultimodalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    resize_fn: Callable[..., tuple[int, int, int, int]],
    max_patches: int,
    max_patches_per_side: int,
    **kwargs,
) -> dict[str, Any] | None:
    """Common processing logic for multimodal samples.

    Args:
        texts: List of strings with None indicating image positions
        images: List of image bytes with None for text positions
        tokenizer: Tokenizer for text processing
        patch_size: Size of image patches
        temporal_patch_size: Temporal patch size (frames per temporal patch)
        spatial_merge_size: merge 2D image patches to reduce LLM's sequence length.
            - if 1 (default): no merge, effectively NoOp
            - if 2: 2x2=4 image patches will be reduced to 1 LLM visual token
        min_pixels: Lower spatial-pixel budget per frame
        max_pixels: Upper spatial-pixel budget per frame
        image_mean: Per-channel mean for normalization
        image_std: Per-channel std for normalization
        resize_fn: Resize strategy; see ``process_image``
        max_patches: Pre-padding patch budget (``resize_to_navit_patch_grid``)
        max_patches_per_side: Pre-padding per-side patch limit
            (``resize_to_navit_patch_grid``)

    Returns:
        Dict with:
            - input_ids: Tensor of token IDs
            - labels: Tensor of label IDs, vision placeholders masked out
            - positions: Tensor of per-token positions
            - pixel_values: List of processed image tensors
        or None if the sample is unusable (no texts, undecodable images,
        or fewer than two tokens).

    Example:
        Interleaved format:
        texts = [text1, None, text2, None, text3]
        images = [None, img1, None, img2, None]

        Image-text pair format as a special case of interleaved:
        texts = [None, text]
        images = [image, None]
    """
    if not texts or len(texts) != len(images):
        return None

    processed_images = []
    num_image_tokens = []

    for idx, img in enumerate(images):
        if img is not None:
            # Resize (to multiples of patch_size x merge_size) and normalize images
            processed_img = process_image(
                img,
                patch_size=patch_size,
                merge_size=spatial_merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                image_mean=image_mean,
                image_std=image_std,
                resize_fn=resize_fn,
                max_patches=max_patches,
                max_patches_per_side=max_patches_per_side,
            )
            if processed_img is not None:
                num_tokens, _, _ = calculate_vision_tokens(
                    num_frames=1,
                    height=processed_img.shape[1],
                    width=processed_img.shape[2],
                    patch_size=patch_size,
                    spatial_merge_size=spatial_merge_size,
                    temporal_patch_size=temporal_patch_size,
                )
                processed_images.append(processed_img)
                num_image_tokens.append(num_tokens)
                # Keep the accepted image at this aligned position as a placeholder.
                texts[idx] = None

    if len(processed_images) != len([_ for _ in images if _ is not None]):
        logger.warning("Cannot process all images for sample. Dropping")
        return None

    # Replace image placeholders (None) with image token sequences
    processed_text = insert_vision_placeholders(
        texts,
        num_image_tokens,
        vision_start_token=tokenizer.vision_start_token,
        vision_token=tokenizer.image_token,
        vision_end_token=tokenizer.vision_end_token,
        eos_token=tokenizer.eos_token,
    )

    tokens = tokenizer.encode(processed_text)
    if len(tokens) < 2:
        return None

    input_ids = torch.tensor(tokens[:-1])
    labels = torch.tensor(tokens[1:])

    special_token_ids = torch.tensor(
        [
            tokenizer.vision_start_id,
            tokenizer.vision_end_id,
            tokenizer.image_id,
            tokenizer.video_id,
        ]
    )
    labels = torch.where(torch.isin(labels, special_token_ids), IGNORE_INDEX, labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(len(input_ids)),
        "pixel_values": processed_images,
    }


def process_obelics_sample(
    sample: dict[str, Any],
    tokenizer: MultimodalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return process_multimodal_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        **kwargs,
    )


def process_cc12_wd_sample(
    sample: dict[str, Any],
    tokenizer: MultimodalTokenizer,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    **kwargs,
) -> dict[str, Any] | None:
    """Process a sample from the CC12-WD dataset (text-image pairs)."""
    text = sample.get("txt", "")
    image = sample.get("jpg", None)

    # Deliberate divergence: upstream falls back to a text-only row here, which
    # leaves the ``None`` slot with no token count for ``insert_vision_placeholders``
    # to expand -- llmtuner lets that raise instead. A ``cc12m-wds`` row with no
    # ``jpg`` is malformed, not an image that failed to decode, and the sample
    # processor drops the latter explicitly; see
    # ``tests/unit_tests/cpu/datasets/test_multimodal_data.py``.
    texts = [None, text]
    images = [image, None]

    return process_multimodal_sample(
        texts=texts,
        images=images,
        tokenizer=tokenizer,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        **kwargs,
    )


class MultimodalProcessor(SampleProcessor):
    """Adapts a multimodal sample function to Grain's map contract."""

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        sample_processor: Callable[..., dict[str, Any] | None],
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        min_pixels: int = 65_536,
        max_pixels: int = 16_777_216,
        image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
        resize_fn: Callable[..., tuple[int, int, int, int]] = resize_to_pixel_budget,
        max_patches: int = 4096,
        max_patches_per_side: int = 512,
        video_dir: str = "",
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
    ) -> None:
        self._sample_processor = sample_processor
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._spatial_merge_size = spatial_merge_size
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._image_mean = image_mean
        self._image_std = image_std
        self._resize_fn = resize_fn
        self._max_patches = max_patches
        self._max_patches_per_side = max_patches_per_side
        self._video_dir = video_dir
        self._video_fps = video_fps
        self._video_min_frames = video_min_frames
        self._video_max_frames = video_max_frames

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        del rng
        processed = self._sample_processor(
            sample=sample,
            tokenizer=self._tokenizer,
            patch_size=self._patch_size,
            temporal_patch_size=self._temporal_patch_size,
            spatial_merge_size=self._spatial_merge_size,
            min_pixels=self._min_pixels,
            max_pixels=self._max_pixels,
            image_mean=self._image_mean,
            image_std=self._image_std,
            resize_fn=self._resize_fn,
            max_patches=self._max_patches,
            max_patches_per_side=self._max_patches_per_side,
            video_dir=self._video_dir,
            video_fps=self._video_fps,
            video_min_frames=self._video_min_frames,
            video_max_frames=self._video_max_frames,
        )
        if (
            processed is not None
            and processed["input_ids"].shape[0] > self._max_context_length
        ):
            # Routine per-sample filtering, same as ChatProcessor's overflow
            # drop: debug, not a warning, or every long sample spams the log.
            logger.debug(
                f"Sample length {processed['input_ids'].shape[0]} > training "
                f"max_context_length={self._max_context_length}. Skip"
            )
            return None
        return processed


MULTIMODAL_DATASETS: dict[str, SingleDataset] = {
    "obelics": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="HuggingFaceM4/OBELICS",
            split="train",
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_obelics_sample,
        ),
        post_filters=(is_not_none,),
    ),
    "cc12m": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="pixparse/cc12m-wds",
            split="train",
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_cc12_wd_sample,
        ),
        post_filters=(is_not_none,),
    ),
    "cc12m-test": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="tests/assets/cc12m_test",
            split="train",
            load_dataset_kwargs={
                "data_files": {"train": "*.tar"},
            },
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_cc12_wd_sample,
        ),
        post_filters=(is_not_none,),
    ),
}
