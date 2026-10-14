"""Multimodal collator for VLM datasets.

Vendored from torchtitan ``hf_datasets/multimodal/mm_collator.py``. The
collator's knobs -- patch geometry, patch order, the media budget -- are plain
keyword arguments, and the build context supplies the tokenizer and the
token-batch size.

``build_mrope_positions`` is the one option that can produce a silently wrong
tensor rather than an error: MRoPE coordinates are laid out in block patch
order, so a raster order would leave every vision token's positional
coordinates pointing at a different patch. It raises instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

from ...components.loss import IGNORE_INDEX
from ...components.tokenizer.multimodal import MultimodalTokenizer
from ..collators import Collator
from ..contracts import DatasetBuildContext, TrainerBatch, require_positive_integer
from .image import vision_to_patches
from .mrope import build_mrope_positions

__all__ = ["MultimodalCollator"]


class MultimodalCollator(Collator):
    """Multimodal collator for VLM training.

    Handles both image and text data, converting images to patches
    and preparing text for model input.
    """

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        max_images_per_batch: int = 128,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        build_mrope_positions: bool = False,
        patch_order: Literal["block", "raster"] = "block",
    ) -> None:
        for name, value in (
            ("patch_size", patch_size),
            ("temporal_patch_size", temporal_patch_size),
            ("spatial_merge_size", spatial_merge_size),
        ):
            require_positive_integer(name, value)
        if (
            not isinstance(max_images_per_batch, int)
            or isinstance(max_images_per_batch, bool)
            or max_images_per_batch < 0
        ):
            raise ValueError("max_images_per_batch must be a nonnegative integer")
        if patch_order not in ("block", "raster"):
            raise ValueError("patch_order must be 'block' or 'raster'")
        self._num_tokens_per_batch = context.num_tokens_per_batch
        self._max_context_length = context.max_context_length
        self.max_images_per_batch = max_images_per_batch
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.tokenizer = cast(MultimodalTokenizer, context.tokenizer)
        self.build_mrope_positions = build_mrope_positions
        self.patch_order = patch_order

    def collate_media(
        self, media: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process image/video tensors into packed patches and grid dimensions.

        Args:
            media: Non-empty sequence of image/video tensors, each of shape
                (T, H, W, C)

        Returns:
            pixel_values: Packed patches (num_patches, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) with [T, H_patches, W_patches]

        ``grid_thw.prod(-1)`` gives each item's length in the patch sequence.
        """
        results = [
            vision_to_patches(
                item,
                self.patch_size,
                self.temporal_patch_size,
                self.spatial_merge_size,
                patch_order=self.patch_order,
            )
            for item in media
        ]
        all_patches = [r[0] for r in results]
        grid_thw_list = [r[1] for r in results]

        packed_patches = torch.cat(all_patches, dim=0)
        grid_thw = torch.stack(grid_thw_list, dim=0)  # (num_images, 3)

        return packed_patches, grid_thw

    def collate_text(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate whole samples and pad only the token-batch tail."""
        if not batch:
            raise ValueError("multimodal batch must contain at least one sample")
        for index, sample in enumerate(batch):
            input_shape = sample["input_ids"].shape
            for field in ("input_ids", "labels", "positions", "padding_mask"):
                if field == "padding_mask" and field not in sample:
                    continue
                value = sample[field]
                if value.ndim != 1 or value.shape != input_shape:
                    raise ValueError(
                        f"multimodal sample {index}: {field} must be a 1D tensor "
                        f"matching input_ids shape {tuple(input_shape)}"
                    )
                if field == "padding_mask" and value.dtype != torch.bool:
                    raise ValueError(
                        f"multimodal sample {index}: padding_mask must have bool dtype"
                    )
        input_ids = torch.cat([sample["input_ids"] for sample in batch])
        labels = torch.cat([sample["labels"] for sample in batch])
        positions = torch.cat([sample["positions"] for sample in batch])
        padding_mask = torch.cat(
            [
                (
                    sample["padding_mask"]
                    if "padding_mask" in sample
                    else torch.zeros(sample["input_ids"].shape[0], dtype=torch.bool)
                )
                for sample in batch
            ]
        )
        labels.masked_fill_(padding_mask, IGNORE_INDEX)
        pad_len = self._num_tokens_per_batch - input_ids.shape[0]
        if pad_len < 0:
            raise ValueError("multimodal rows exceed the configured token batch")
        if pad_len:
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, pad_len),
                value=self.tokenizer.pad_id,
            )
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            padding_positions = (
                torch.arange(pad_len, dtype=positions.dtype) % self._max_context_length
            )
            positions = torch.cat([positions, padding_positions])
            padding_mask = torch.nn.functional.pad(
                padding_mask, (0, pad_len), value=True
            )

        return input_ids, labels, positions, padding_mask

    def __call__(self, batch: Sequence[dict[str, Any]]) -> TrainerBatch:
        """Collate batch with patch-based approach."""
        all_images = [
            image for sample in batch for image in sample.get("pixel_values", [])
        ]
        all_videos = [
            video for sample in batch for video in sample.get("pixel_values_videos", [])
        ]
        # Each image costs one entry; videos cost one per temporal patch group.
        total_images = len(all_images) + sum(
            (video.shape[0] + self.temporal_patch_size - 1) // self.temporal_patch_size
            for video in all_videos
        )
        if total_images > self.max_images_per_batch:
            raise ValueError(
                f"multimodal batch has {total_images} vision entries, exceeding "
                f"max_images_per_batch={self.max_images_per_batch}"
            )

        # Collate image and video patches.
        patches, grids = self.collate_media(all_images) if all_images else (None, None)
        video_patches, video_grids = (
            self.collate_media(all_videos) if all_videos else (None, None)
        )

        # Pad text.
        input_ids, labels, positions, padding_mask = self.collate_text(batch)
        input_dict = {
            "input": input_ids,
            "labels": labels,
            "positions": positions,
            "padding_mask": padding_mask,
            "pixel_values": patches,
            "grid_thw": grids,
            "pixel_values_videos": video_patches,
            "grid_thw_videos": video_grids,
            "special_tokens": {
                f"{name}_id": getattr(self.tokenizer, f"{name}_id")
                for name in self.tokenizer.TOKEN_FIELDS
            },
            "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
        }

        if self.build_mrope_positions and (
            grids is not None or video_grids is not None
        ):
            special_tokens = input_dict["special_tokens"]
            input_dict["mrope_positions"] = build_mrope_positions(
                input_ids,
                grids,
                video_grids,
                positions,
                spatial_merge_size=self.spatial_merge_size,
                patch_order=self.patch_order,
                image_token_id=special_tokens["image_id"],
                video_token_id=special_tokens["video_id"],
            )

        return input_dict
