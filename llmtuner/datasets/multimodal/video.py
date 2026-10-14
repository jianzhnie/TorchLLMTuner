"""Video processing utilities for multimodal datasets.

Vendored from torchtitan ``hf_datasets/multimodal/utils/video.py``. ``av`` is
imported inside ``load_video`` rather than at module scope so the rest of the
multimodal layer imports without a video backend installed; that was already
true upstream and is what keeps a text-only or image-only run from needing it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torchvision.transforms.v2.functional as TVF

from ...utils.logging import get_logger
from ..contracts import require_positive_integer
from .image import smart_resize

logger = get_logger(__name__)

__all__ = ["load_video", "process_video"]


def load_video(
    path: str,
    *,
    fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 768,
) -> torch.Tensor | None:
    """Load and sample frames from a video file using PyAV.

    Iterates through the video stream and only converts the sampled frames
    to numpy, avoiding full decode of every frame.

    Args:
        path: Path to the video file.
        fps: Target frames per second for sampling.
        min_frames: Minimum number of frames to sample.
        max_frames: Maximum number of frames to sample.

    Returns:
        Tensor of shape (T, H, W, C) in uint8, channel-last format,
        or None if loading fails. Missing PyAV raises ImportError with an
        installation hint.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    require_positive_integer("min_frames", min_frames)
    require_positive_integer("max_frames", max_frames)
    if not 1 <= min_frames <= max_frames:
        raise ValueError("frame bounds must satisfy 1 <= min_frames <= max_frames")
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "Video loading requires the optional PyAV backend; "
            "install it with `pip install av`."
        ) from error

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]

            video_fps = float(stream.average_rate or stream.guessed_rate or 24)
            total_frames = stream.frames
            if total_frames == 0 and stream.duration:
                total_frames = int(
                    float(stream.duration * stream.time_base) * video_fps
                )
            all_frames = None
            if total_frames == 0:
                # No metadata available -- decode all frames to count them
                container.seek(0)
                all_frames = [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
                if not all_frames:
                    return None
                total_frames = len(all_frames)

            duration = total_frames / video_fps
            nframes = int(duration * fps)
            nframes = max(min_frames, min(nframes, max_frames))
            nframes = min(nframes, total_frames)

            indices = np.linspace(0, total_frames - 1, nframes).astype(int).tolist()
            if all_frames is not None:
                return torch.from_numpy(np.stack([all_frames[i] for i in indices]))
            selected_indices = set(indices)

            # Must decode sequentially -- inter-frame codecs (H.264/H.265) have
            # dependencies between frames. Only selected frames are converted to
            # numpy to avoid the RGB conversion cost.
            frames = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i in selected_indices:
                    frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            return None

        return torch.from_numpy(np.stack(frames))  # (T, H, W, C)

    except Exception as e:
        logger.warning(f"Error loading video {path}: {e}")
        return None


def process_video(
    video: torch.Tensor,
    *,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Resize and normalize video frames for VLM training.

    Uses torchvision v2 APIs with uint8 resize for faster SIMD paths.

    Args:
        video: Raw video tensor of shape (T, H, W, C) in uint8.
        patch_size: Spatial patch size.
        merge_size: Spatial merge size.
        min_pixels: Minimum spatial pixels per frame (H * W budget).
        max_pixels: Maximum spatial pixels per frame (H * W budget).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.

    Returns:
        Normalized tensor of shape (T, H', W', C) in float32.
    """
    if video.ndim != 4 or any(size < 1 for size in video.shape):
        raise ValueError("video must have nonempty (T, H, W, C) dimensions")
    if video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise ValueError("video must contain uint8 RGB pixels")
    if patch_size < 1 or merge_size < 1:
        raise ValueError("patch_size and merge_size must be positive")
    _, height, width, _ = video.shape
    factor = patch_size * merge_size

    target_h, target_w = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize on uint8 for faster SIMD paths (AVX2/NEON)
    # (T, H, W, C) -> (T, C, H, W) for torchvision
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W) uint8
    video = TVF.resize(
        video,
        [target_h, target_w],
        interpolation=TVF.InterpolationMode.BICUBIC,
        antialias=True,
    )

    # uint8 -> float32 [0, 1] -> normalize -> channel-last
    video = TVF.to_dtype(video, torch.float32, scale=True)
    video = TVF.normalize(video, list(image_mean), list(image_std), inplace=True)
    # (T, C, H', W') -> (T, H', W', C)
    return video.permute(0, 2, 3, 1)
