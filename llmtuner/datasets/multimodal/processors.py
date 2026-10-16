"""Decode multimodal samples into token targets and processed images.

Dataset adapters normalize OBELICS interleaved rows and CC12M image-text pairs.
MultimodalProcessor binds vision options and implements Grain's sample contract.
Packing groups whole documents; MultimodalCollator later assembles patch tensors.

Adapted from torchtitan ``hf_datasets/multimodal/mm_datasets.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

from ...components.loss import IGNORE_INDEX
from ...components.tokenizer.multimodal import MultimodalTokenizer
from ...utils.logging import get_logger
from ..contracts import DatasetBuildContext, SampleProcessor, require_positive_integer
from .image import calculate_vision_tokens, process_image, resize_to_pixel_budget
from .placeholders import insert_vision_placeholders

logger = get_logger(__name__)

__all__ = [
    "MultimodalProcessor",
]


def process_multimodal_sample(
    texts: list[str | None],
    images: list[bytes | None],
    *,
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
    **kwargs: Any,
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

    texts = list(texts)
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
    *,
    tokenizer: MultimodalTokenizer,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Process a sample from the OBELICS dataset (interleaved text and images)."""
    return process_multimodal_sample(
        texts=sample.get("texts", []),
        images=sample.get("images", []),
        tokenizer=tokenizer,
        **kwargs,
    )


def process_cc12m_sample(
    sample: dict[str, Any],
    *,
    tokenizer: MultimodalTokenizer,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Process a CC12M WebDataset row with ``txt`` and ``jpg`` fields."""
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
        **kwargs,
    )


class MultimodalProcessor(SampleProcessor):
    """Adapts a multimodal sample function to Grain's map contract."""

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        process_sample: Callable[..., dict[str, Any] | None],
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
        for name, value in (
            ("patch_size", patch_size),
            ("temporal_patch_size", temporal_patch_size),
            ("spatial_merge_size", spatial_merge_size),
        ):
            require_positive_integer(name, value)
        self._max_context_length = context.max_context_length
        self._process_sample = partial(
            process_sample,
            tokenizer=context.tokenizer,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
            resize_fn=resize_fn,
            max_patches=max_patches,
            max_patches_per_side=max_patches_per_side,
            video_dir=video_dir,
            video_fps=video_fps,
            video_min_frames=video_min_frames,
            video_max_frames=video_max_frames,
        )

    def __call__(
        self,
        sample: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        del rng
        processed = self._process_sample(sample=sample)
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
