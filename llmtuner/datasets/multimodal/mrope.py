"""Three-axis rotary positions for packed multimodal token sequences."""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["build_mrope_positions"]


def text_positions(text_len: int, offset: int) -> torch.Tensor:
    """Sequential positions for a text run, identical on all 3 MRoPE axes."""
    return torch.arange(text_len).view(1, -1).expand(3, -1) + offset


def vision_grid_positions(
    t: int, h: int, w: int, cache: dict[tuple[int, int, int], torch.Tensor]
) -> torch.Tensor:
    """The (3, t*h*w) 3D grid coordinates for a vision run, cached by shape."""
    key = (t, h, w)
    if key not in cache:
        hw = h * w
        t_index = torch.arange(t).view(-1, 1).expand(-1, hw).flatten()
        h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
        w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
        cache[key] = torch.stack([t_index, h_index, w_index])
    return cache[key]


def build_mrope_positions(
    tokens: torch.Tensor,
    grid_thw: torch.Tensor | None,
    grid_thw_videos: torch.Tensor | None,
    positions: torch.Tensor | None,
    *,
    spatial_merge_size: int,
    patch_order: Literal["block", "raster"],
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """Build 3D (temporal, height, width) MRoPE position IDs per token.

    Returns ``(num_tokens, 3)`` with temporal/height/width coordinates in
    the final dimension. Runs here on CPU data workers, off the GPU path.

    Args:
        tokens: (num_tokens,) token IDs.
        grid_thw: (num_images, 3) image grid dims, or None.
        grid_thw_videos: (num_videos, 3) video grid dims, or None.
        positions: (num_tokens,) per-token positions; document
            boundaries are detected where positions reset.
        spatial_merge_size: Spatial patch merge factor used by the vision encoder.
        patch_order: Patch layout; MRoPE requires block order.
        image_token_id: Placeholder token ID marking image positions.
        video_token_id: Placeholder token ID marking video positions.

    Returns:
        (num_tokens, 3) MRoPE position IDs.
    """
    if (
        not isinstance(spatial_merge_size, int)
        or isinstance(spatial_merge_size, bool)
        or spatial_merge_size < 1
    ):
        raise ValueError(
            f"spatial_merge_size must be positive integer, got {spatial_merge_size!r}"
        )
    if tokens.ndim != 1:
        raise ValueError("MRoPE tokens must be one-dimensional")
    if positions is not None and positions.shape != tokens.shape:
        raise ValueError("MRoPE positions must match the one-dimensional token shape")
    for name, grid in (("grid_thw", grid_thw), ("grid_thw_videos", grid_thw_videos)):
        if grid is None:
            continue
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"MRoPE {name} must have shape (num_media, 3)")
        if grid.dtype not in (torch.int32, torch.int64) or bool((grid <= 0).any()):
            raise ValueError(f"MRoPE {name} must contain positive integer dimensions")
    # MRoPE position IDs are laid out in block order; a raster patch order
    # would desync them from the patch sequence.
    if patch_order != "block":
        raise ValueError(f"MRoPE requires patch_order='block', got {patch_order!r}.")

    if tokens.numel() == 0:
        if any(
            grid is not None and grid.numel() for grid in (grid_thw, grid_thw_videos)
        ):
            raise ValueError("MRoPE media/grid mismatch: empty tokens have media grids")
        return tokens.new_empty((0, 3))

    # Transformers splits Qwen3.5 video grids into per-frame entries because
    # timestamps split its video tokens into matching modality runs. Our data
    # pipeline emits one contiguous placeholder run per video instead, so each
    # run must consume the original [T, H, W] grid as a single 3D region.

    tokens = tokens.unsqueeze(0)
    positions = positions.unsqueeze(0) if positions is not None else None
    batch_size, seq_len = tokens.shape
    mrope_positions = torch.zeros(
        batch_size, seq_len, 3, dtype=tokens.dtype, device=tokens.device
    )

    if positions is not None:
        # Every document starts at 0. A decrease check misses 0 -> 0
        # boundaries after padding or a single-token document.
        resets = positions[:, 1:] == 0  # (batch, seq_len-1)
    # First token of each consecutive vision region (image or video).
    vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
    prev_vision = torch.cat(
        [torch.zeros_like(vision_mask[:, :1]), vision_mask[:, :-1]], dim=1
    )
    prev_tokens = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
    # A transition image -> video is two media runs even though both sides
    # satisfy ``vision_mask``. Treating it as one would consume only the
    # image grid and shift every following grid by one.
    batch_vision_starts = vision_mask & (
        ~prev_vision | (tokens != prev_tokens)
    )  # (batch, seq_len)
    if positions is not None:
        # Identical placeholders on opposite sides of a document boundary
        # belong to distinct media and must consume separate grids.
        batch_vision_starts[:, 1:] |= vision_mask[:, 1:] & resets

    num_image_runs = int((batch_vision_starts & (tokens == image_token_id)).sum())
    num_video_runs = int((batch_vision_starts & (tokens == video_token_id)).sum())
    num_image_grids = 0 if grid_thw is None else int(grid_thw.shape[0])
    num_video_grids = 0 if grid_thw_videos is None else int(grid_thw_videos.shape[0])
    if num_image_runs != num_image_grids or num_video_runs != num_video_grids:
        raise ValueError(
            "MRoPE media/grid mismatch: "
            f"found {num_image_runs} image placeholder run(s) and "
            f"{num_video_runs} video placeholder run(s), but received "
            f"{num_image_grids} image grid(s) and {num_video_grids} video grid(s)"
        )
    grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    image_index, video_index = 0, 0
    # With sample packing, each sample may contain multiple documents.
    for sample_i in range(batch_size):
        llm_pos_ids_list: list[torch.Tensor] = []

        if positions is not None:
            reset_indices = torch.where(resets[sample_i])[0] + 1
            doc_starts = [0] + reset_indices.tolist()
            doc_ranges = [
                (
                    doc_starts[d],
                    doc_starts[d + 1] if d + 1 < len(doc_starts) else seq_len,
                )
                for d in range(len(doc_starts))
            ]
        else:
            doc_ranges = [(0, seq_len)]

        sample_tokens = tokens[sample_i]
        sample_vision_starts = torch.where(batch_vision_starts[sample_i])[0].tolist()
        vision_start_index = 0

        for doc_start, doc_end in doc_ranges:
            doc_pos_ids_list: list[torch.Tensor] = []

            doc_vision_starts: list[int] = []
            while (
                vision_start_index < len(sample_vision_starts)
                and sample_vision_starts[vision_start_index] < doc_end
            ):
                doc_vision_starts.append(sample_vision_starts[vision_start_index])
                vision_start_index += 1

            pair_cursor = doc_start
            for vision_start in doc_vision_starts:
                if sample_tokens[vision_start] == image_token_id:
                    t, h, w = grid_thw[image_index]
                    image_index += 1
                else:
                    t, h, w = grid_thw_videos[video_index]
                    video_index += 1

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    int(t.item()),
                    int(h.item()) // spatial_merge_size,
                    int(w.item()) // spatial_merge_size,
                )
                if (
                    int(h.item()) % spatial_merge_size
                    or int(w.item()) % spatial_merge_size
                ):
                    raise ValueError(
                        "MRoPE grid spatial dimensions must be divisible by "
                        f"spatial_merge_size={spatial_merge_size}; got "
                        f"grid [{int(t.item())}, {int(h.item())}, {int(w.item())}]"
                    )
                expected_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w
                vision_end = vision_start
                while (
                    vision_end < doc_end
                    and sample_tokens[vision_end] == sample_tokens[vision_start]
                ):
                    vision_end += 1
                actual_vision_tokens = vision_end - vision_start
                if actual_vision_tokens != expected_vision_tokens:
                    media_kind = (
                        "image"
                        if sample_tokens[vision_start] == image_token_id
                        else "video"
                    )
                    raise ValueError(
                        f"MRoPE {media_kind} placeholder run has "
                        f"{actual_vision_tokens} token(s), but its grid "
                        f"requires {expected_vision_tokens}"
                    )
                text_len = vision_start - pair_cursor

                pos_id_offset = (
                    int(doc_pos_ids_list[-1].max()) + 1 if doc_pos_ids_list else 0
                )
                # [text tokens] -- sequential positions, identical on all 3 axes.
                doc_pos_ids_list.append(text_positions(text_len, pos_id_offset))
                # [vision tokens] -- 3D grid positions (T, H, W).
                doc_pos_ids_list.append(
                    vision_grid_positions(
                        llm_grid_t, llm_grid_h, llm_grid_w, grid_cache
                    )
                    + text_len
                    + pos_id_offset
                )
                pair_cursor = vision_end

            # Trailing [text tokens] after the last text/vision pair.
            if pair_cursor < doc_end:
                pos_id_offset = (
                    int(doc_pos_ids_list[-1].max()) + 1 if doc_pos_ids_list else 0
                )
                doc_pos_ids_list.append(
                    text_positions(doc_end - pair_cursor, pos_id_offset)
                )

            llm_pos_ids_list.extend(doc_pos_ids_list)

        # llm_pos_ids_list is (3, segment_len); concat -> (3, seq), then transpose
        mrope_positions[sample_i] = torch.cat(llm_pos_ids_list, dim=1).T

    return mrope_positions.squeeze(0)
