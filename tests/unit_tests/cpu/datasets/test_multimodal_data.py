"""The multimodal data path: image preprocessing, placeholder alignment,
collation, and packing.

The failure this file is built around is *misalignment*. Every stage maps N
vision items to N spans of placeholder tokens, and if the two counts disagree
the copy still succeeds -- it just puts the wrong features behind the wrong
tokens, and attention trains happily on them. So each test below asserts either
an exact count or that a mismatch raises or drops.

Runs on CPU with no network: images are generated and encoded as real PNGs, and
the corpus is written to a tmp dir. No video backend is required -- ``load_video``
imports ``av`` lazily, and every video path exercised here goes through
``process_video`` with a synthetic frame stack.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("grain")


import base64
import json

import numpy as np
import pytest
import torch

from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.components.tokenizer import MultimodalTokenizer
from llmtuner.datasets import (
    IndexedJsonlSource,
    SingleDataset,
    build_dataset,
)
from llmtuner.datasets.multimodal.collator import MultimodalCollator
from llmtuner.datasets.multimodal.image import (
    calculate_vision_tokens,
    process_image,
    resize_to_navit_patch_grid,
    resize_to_pixel_budget,
    smart_resize,
    vision_to_patches,
)
from llmtuner.datasets.multimodal.mrope import build_mrope_positions
from llmtuner.datasets.multimodal.packing import (
    build_multimodal_packing,
    packing_output_to_multimodal_sample,
)
from llmtuner.datasets.multimodal.placeholders import insert_vision_placeholders
from llmtuner.datasets.multimodal.processors import (
    MultimodalProcessor,
    process_cc12_wd_sample,
    process_multimodal_sample,
)
from llmtuner.datasets.multimodal.video import load_video, process_video
from tests.data_fixtures import (  # noqa: F401
    VOCAB,
    Base64JsonlSource,
    make_context,
    make_policy,
    png_bytes,
    write_tokenizer,
)

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"

# The five tokens a MultimodalTokenizer validates against the vocabulary.
MM_TOKENS = (IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END, "[PAD]")


@pytest.fixture(scope="module")
def mm_tokenizer(tmp_path_factory) -> MultimodalTokenizer:
    """A tokenizer whose vision placeholders are real added tokens.

    They have to be in the saved vocabulary rather than added afterwards:
    ``MultimodalTokenizer`` validates all five at construction. Ids must be
    contiguous with the base vocabulary -- WordLevel refuses to save a vocab
    with holes in it.
    """
    path = write_tokenizer(
        str(tmp_path_factory.mktemp("mm_tokenizer")),
        extra_vocab={
            token: max(VOCAB.values()) + 1 + i for i, token in enumerate(MM_TOKENS)
        },
    )
    return MultimodalTokenizer(
        tokenizer_path=path,
        image_token=IMAGE_TOKEN,
        video_token=VIDEO_TOKEN,
        vision_start_token=VISION_START,
        vision_end_token=VISION_END,
        pad_token="[PAD]",
    )


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    return png_bytes(64, 64)


MM_KWARGS = dict(
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    min_pixels=32 * 32,
    max_pixels=10**8,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    resize_fn=resize_to_pixel_budget,
    max_patches=4096,
    max_patches_per_side=512,
)


# --------------------------------------------------------------------------
# Image utilities
# --------------------------------------------------------------------------


def test_smart_resize_rounds_both_dimensions_to_the_factor():
    height, width = smart_resize(
        100, 200, factor=32, min_pixels=32 * 32, max_pixels=10**8
    )
    assert height % 32 == 0 and width % 32 == 0


def test_smart_resize_rejects_an_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="Absolute aspect ratio"):
        smart_resize(10, 4000, factor=32, min_pixels=1, max_pixels=10**8)


def test_resize_to_pixel_budget_upscales_a_tiny_image_to_the_factor():
    """Both dims have to reach the factor or the patch grid comes out empty."""
    height, width, pad_h, pad_w = resize_to_pixel_budget(
        3, 5, patch_size=16, merge_size=2, min_pixels=32 * 32, max_pixels=10**8
    )
    assert height >= 32 and width >= 32
    assert (pad_h, pad_w) == (0, 0)


def test_resize_to_navit_patch_grid_pads_rather_than_rescales():
    """The Kimi-VL strategy pads up to the grid instead of resizing to it, so an
    in-budget image is not upscaled and the remainder comes back as padding."""
    height, width, pad_h, pad_w = resize_to_navit_patch_grid(
        70,
        70,
        patch_size=16,
        merge_size=2,
        max_patches=4096,
        max_patches_per_side=512,
    )
    assert (height, width) == (70, 70)
    assert pad_h == (32 - 70 % 32) % 32
    assert pad_w == (32 - 70 % 32) % 32


def test_process_image_returns_a_normalized_dummy_temporal_dim(image_bytes):
    tensor = process_image(
        image_bytes, patch_size=16, merge_size=2, min_pixels=32 * 32, max_pixels=10**8
    )
    assert tensor is not None
    assert tensor.ndim == 4 and tensor.shape[0] == 1  # (1, H, W, C)
    assert tensor.dtype == torch.float32
    # Normalized by mean=std=0.5, so a raw 0..255 range would sit far outside.
    assert float(tensor.max()) <= 1.5 and float(tensor.min()) >= -1.5


def test_process_image_returns_none_on_undecodable_bytes():
    """A bad image drops its sample rather than failing the run -- which is also
    how a dataset quietly ends up imageless, so the None is worth pinning."""
    assert process_image(b"not an image", patch_size=16, merge_size=2) is None


def test_calculate_vision_tokens_matches_the_patch_grid():
    total, per_row, rows = calculate_vision_tokens(
        num_frames=1,
        height=64,
        width=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    assert (per_row, rows) == (2, 2)
    assert total == 4


def test_vision_to_patches_counts_raw_patches_per_side():
    image = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    patches, grid = vision_to_patches(
        image, patch_size=16, temporal_patch_size=2, merge_size=2
    )
    # patch_dim is (C, pt, ph, pw) = 3 * 2 * 16 * 16; the temporal patch is part
    # of each patch vector, not a separate patch.
    assert patches.shape == (16, 3 * 2 * 16 * 16)
    assert grid.tolist() == [1, 4, 4]


def test_vision_to_patches_block_and_raster_orders_differ():
    """Why ``patch_order`` exists: in block order the patches that merge into a
    token are adjacent, in raster order they are not. Same content, new order.
    """
    image = torch.arange(1 * 4 * 8, dtype=torch.uint8).reshape(1, 4, 8, 1)
    block, _ = vision_to_patches(
        image, patch_size=1, temporal_patch_size=1, merge_size=2, patch_order="block"
    )
    raster, _ = vision_to_patches(
        image, patch_size=1, temporal_patch_size=1, merge_size=2, patch_order="raster"
    )
    assert not torch.equal(block, raster)
    assert sorted(block.flatten().tolist()) == sorted(raster.flatten().tolist())


def test_vision_to_patches_rejects_an_unknown_order():
    with pytest.raises(ValueError, match="patch_order must be"):
        vision_to_patches(torch.zeros(1, 4, 4, 3), 2, 1, 1, patch_order="spiral")


def test_vision_to_patches_pads_the_temporal_dim_by_repeating_the_last_frame():
    image = torch.zeros(3, 8, 8, 3, dtype=torch.uint8)
    _, grid = vision_to_patches(
        image, patch_size=4, temporal_patch_size=2, merge_size=1
    )
    assert grid[0].item() == 2  # 3 frames -> padded to 4 -> 2 temporal patches


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


def test_process_video_resizes_and_normalizes():
    frames = torch.arange(2 * 32 * 32 * 3, dtype=torch.uint8).reshape(2, 32, 32, 3)
    out = process_video(
        frames, patch_size=16, merge_size=2, min_pixels=32 * 32, max_pixels=10**8
    )
    assert out.shape == (2, 32, 32, 3)
    assert out.dtype == torch.float32
    assert float(out.max()) <= 1.5 and float(out.min()) >= -1.5


def test_load_video_without_a_backend_returns_none(tmp_path):
    """``av`` is imported inside the function so a missing video backend is a
    None here rather than an ImportError that takes the whole layer down."""
    assert load_video(str(tmp_path / "missing.mp4")) is None


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((32, 32, 3), torch.uint8),
        ((0, 32, 32, 3), torch.uint8),
        ((1, 0, 32, 3), torch.uint8),
        ((1, 32, 32, 1), torch.uint8),
        ((1, 32, 32, 3), torch.float32),
    ],
)
def test_process_video_rejects_invalid_pixel_layout(shape, dtype):
    with pytest.raises(ValueError, match="video must"):
        process_video(
            torch.zeros(shape, dtype=dtype),
            patch_size=16,
            merge_size=2,
            min_pixels=1024,
            max_pixels=4096,
        )


@pytest.mark.parametrize("geometry", [{"patch_size": -1}, {"merge_size": -1}])
def test_process_video_rejects_negative_patch_geometry(geometry):
    with pytest.raises(ValueError, match="must be positive"):
        process_video(
            torch.zeros(1, 32, 32, 3, dtype=torch.uint8),
            **(
                dict(patch_size=16, merge_size=2, min_pixels=1024, max_pixels=4096)
                | geometry
            ),
        )


# --------------------------------------------------------------------------
# Placeholder insertion: the alignment contract
# --------------------------------------------------------------------------


def test_insert_vision_placeholders_expands_each_slot_to_its_token_count():
    text = insert_vision_placeholders(
        ["hello", None, "world"],
        [3],
        vision_start_token=VISION_START,
        vision_token=IMAGE_TOKEN,
        vision_end_token=VISION_END,
    )
    assert text == f"hello{VISION_START}{IMAGE_TOKEN * 3}{VISION_END}world"


def test_insert_vision_placeholders_appends_eos_once():
    kwargs = dict(
        vision_start_token=VISION_START,
        vision_token=IMAGE_TOKEN,
        vision_end_token=VISION_END,
        eos_token="[EOS]",
    )
    assert insert_vision_placeholders(["abc"], [], **kwargs) == "abc[EOS]"
    assert insert_vision_placeholders(["abc[EOS]"], [], **kwargs) == "abc[EOS]"


def test_insert_vision_placeholders_rejects_a_slot_with_no_token_count():
    """A ``None`` slot with no corresponding vision token count raises.

    Not a bug: the processor only ever sets a slot to ``None`` for an image it
    accepted, and it drops the whole sample if any image failed. So a slot with
    no count means that invariant broke, and failing loudly beats emitting text
    with a hole where an image should be.
    """
    with pytest.raises(ValueError, match="one vision token count per placeholder"):
        insert_vision_placeholders(
            ["a", None],
            [],
            vision_start_token=VISION_START,
            vision_token=IMAGE_TOKEN,
            vision_end_token=VISION_END,
        )


# --------------------------------------------------------------------------
# Sample processing
# --------------------------------------------------------------------------


def test_process_multimodal_sample_rejects_mismatched_text_and_image_lists(
    mm_tokenizer,
):
    assert (
        process_multimodal_sample(["a"], [], tokenizer=mm_tokenizer, **MM_KWARGS)
        is None
    )


def test_process_multimodal_sample_masks_the_vision_tokens_in_the_labels(
    mm_tokenizer, image_bytes
):
    """Vision placeholders must not be predicted as text.

    The model reads features, not tokens, at those positions, so a label there
    trains the language head against something it never sees. The assertion is
    therefore the negative one: no placeholder survives in ``labels``.
    """
    result = process_multimodal_sample(
        texts=[None, "hello"],
        images=[image_bytes, None],
        tokenizer=mm_tokenizer,
        **MM_KWARGS,
    )
    assert result is not None

    special = torch.tensor(
        [
            mm_tokenizer.vision_start_id,
            mm_tokenizer.vision_end_id,
            mm_tokenizer.image_id,
            mm_tokenizer.video_id,
        ]
    )
    # The placeholders are all in the input, so the sample really does contain
    # them -- otherwise the check below would pass on an empty sequence.
    assert bool(torch.isin(result["input_ids"], special).any())
    assert not bool(torch.isin(result["labels"], special).any())


def test_process_multimodal_sample_aligns_every_token_field(mm_tokenizer, image_bytes):
    result = process_multimodal_sample(
        texts=[None, "hello"],
        images=[image_bytes, None],
        tokenizer=mm_tokenizer,
        **MM_KWARGS,
    )
    assert result is not None
    assert len(result["pixel_values"]) == 1
    assert result["input_ids"].shape == result["positions"].shape
    assert result["input_ids"].shape == result["labels"].shape


def test_process_multimodal_sample_preserves_source_texts(mm_tokenizer, image_bytes):
    texts = ["image caption", "hello"]
    result = process_multimodal_sample(
        texts=texts,
        images=[image_bytes, None],
        tokenizer=mm_tokenizer,
        **MM_KWARGS,
    )
    assert result is not None
    assert texts == ["image caption", "hello"]


def test_process_multimodal_sample_drops_the_sample_when_one_image_fails(
    mm_tokenizer, image_bytes
):
    """A partially processed sample is dropped, not kept with fewer images than
    placeholders -- keeping it would misalign every image after the failure."""
    assert (
        process_multimodal_sample(
            texts=[None, None, "hello"],
            images=[image_bytes, b"broken", None],
            tokenizer=mm_tokenizer,
            **MM_KWARGS,
        )
        is None
    )


def test_process_cc12_wd_sample_reads_the_pair_format(mm_tokenizer, image_bytes):
    result = process_cc12_wd_sample(
        {"txt": "hello", "jpg": image_bytes}, tokenizer=mm_tokenizer, **MM_KWARGS
    )
    assert result is not None
    assert len(result["pixel_values"]) == 1


def test_process_cc12_wd_sample_raises_when_the_image_field_is_absent(mm_tokenizer):
    """A row without a ``jpg`` key is malformed for this dataset, not an image
    that failed to decode -- so it raises rather than being silently dropped."""
    with pytest.raises(ValueError, match="one vision token count per placeholder"):
        process_cc12_wd_sample({"txt": "hello"}, tokenizer=mm_tokenizer, **MM_KWARGS)


# --------------------------------------------------------------------------
# The processor as a Grain map
# --------------------------------------------------------------------------


def _mm_processor(mm_tokenizer, *, max_context_length=256):
    return MultimodalProcessor(
        context=make_context(mm_tokenizer, max_context_length=max_context_length),
        sample_processor=process_cc12_wd_sample,
    )


def test_multimodal_processor_drops_samples_over_the_context_length(
    mm_tokenizer, image_bytes
):
    processor = _mm_processor(mm_tokenizer, max_context_length=4)
    sample = {"txt": "lorem ipsum lorem ipsum lorem ipsum", "jpg": image_bytes}
    assert processor(sample, np.random.default_rng(0)) is None


def test_multimodal_processor_passes_short_samples_through(mm_tokenizer, image_bytes):
    processor = _mm_processor(mm_tokenizer)
    result = processor({"txt": "hello", "jpg": image_bytes}, np.random.default_rng(0))
    assert result is not None
    assert "pixel_values" in result


def test_multimodal_processor_runs_over_a_jsonl_corpus(
    mm_tokenizer, image_bytes, tmp_path
):
    """The end-to-end shape of the image-text path: rows in, a tokenized sample
    with its media out. ``pixel_values`` stays per sample until the collator
    flattens it, so the row is the unit under test here.

    The context is sized to fit one padded image: a 64x64 image at patch 16
    with merge 2 contributes many placeholder tokens, and the default 32-token
    context would drop every row. That is the real constraint a caller sizes
    against, so it is stated rather than hidden behind a larger default.
    """
    path = str(tmp_path / "pairs.jsonl")
    encoded = base64.b64encode(image_bytes).decode()
    with open(path, "w") as handle:
        for i in range(8):
            handle.write(
                json.dumps({"txt": f"w{i} hello", "jpg": {"bytes": encoded}}) + "\n"
            )

    config = SingleDataset(
        source=Base64JsonlSource(patterns=(path,)),
        # A class, not a `partial` fixing a different context: row transformation
        # calls `processor(context=...)`, and a partial's bound keyword would be
        # silently overridden by that call-site keyword.
        processor=_CtxSizedMMProcessor,
        post_filters=(lambda sample: sample is not None,),
    )
    dataset = build_dataset(
        config,
        context=make_context(mm_tokenizer, num_tokens_per_batch=1024),
        iteration_policy=make_policy(shuffle=False),
    )
    row = next(iter(dataset))
    assert row is not None
    assert len(row["pixel_values"]) == 1
    assert row["input_ids"].shape[0] >= 2


class _CtxSizedMMProcessor(MultimodalProcessor):
    """``MultimodalProcessor`` with a context large enough for one image."""

    def __init__(self, *, context):
        from dataclasses import replace

        super().__init__(
            context=replace(context, max_context_length=1024),
            sample_processor=process_cc12_wd_sample,
        )


# --------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------


@pytest.fixture
def token_ids(mm_tokenizer) -> dict[str, int]:
    return {
        "image": mm_tokenizer.image_id,
        "video": mm_tokenizer.video_id,
        "vision_start": mm_tokenizer.vision_start_id,
        "vision_end": mm_tokenizer.vision_end_id,
        "pad": mm_tokenizer.pad_id,
    }


def _collator(mm_tokenizer, **kwargs):
    return MultimodalCollator(
        context=make_context(mm_tokenizer, num_tokens_per_batch=128), **kwargs
    )


def _mm_sample(num_tokens: int, token_ids: dict[str, int], *, image_tokens: int = 0):
    from llmtuner.datasets import DatasetBuildContext  # noqa: F401  (type doc)

    input_ids = torch.full((num_tokens,), VOCAB["lorem"], dtype=torch.long)
    labels = torch.full((num_tokens,), VOCAB["ipsum"], dtype=torch.long)
    if image_tokens:
        input_ids[:image_tokens] = token_ids["image"]
        labels[:image_tokens] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
        "positions": torch.arange(num_tokens),
        "pixel_values": [],
    }


def test_collator_pads_the_token_batch_with_pad_and_ignore(mm_tokenizer, token_ids):
    batch = _collator(mm_tokenizer)([_mm_sample(10, token_ids)])
    assert batch["input"].shape == (128,)
    assert batch["input"][10:].tolist() == [token_ids["pad"]] * 118
    assert batch["labels"][10:].tolist() == [IGNORE_INDEX] * 118
    assert batch["num_valid_tokens"] == 10


@pytest.mark.parametrize("field", ["input_ids", "labels", "positions", "padding_mask"])
@pytest.mark.parametrize("shape", [(9,), (2, 5)])
def test_collator_rejects_misaligned_text_fields(mm_tokenizer, token_ids, field, shape):
    sample = _mm_sample(10, token_ids)
    sample[field] = torch.zeros(
        shape, dtype=torch.bool if field == "padding_mask" else torch.long
    )
    with pytest.raises(ValueError, match="must be a 1D tensor"):
        _collator(mm_tokenizer)([sample])


def test_collator_rejects_nonboolean_padding_mask(mm_tokenizer, token_ids):
    sample = _mm_sample(10, token_ids)
    sample["padding_mask"] = torch.zeros(10, dtype=torch.long)
    with pytest.raises(ValueError, match="padding_mask must have bool dtype"):
        _collator(mm_tokenizer)([sample])


def test_collator_rejects_empty_batch(mm_tokenizer):
    with pytest.raises(ValueError, match="at least one sample"):
        _collator(mm_tokenizer)([])


def test_collator_pads_positions_within_the_context_window(mm_tokenizer, token_ids):
    batch = _collator(mm_tokenizer)([_mm_sample(10, token_ids)])
    assert int(batch["positions"][10:].max()) < 32


def test_collator_masks_existing_padding_without_mutating_sample(
    mm_tokenizer, token_ids
):
    sample = _mm_sample(10, token_ids)
    original_labels = sample["labels"].clone()
    sample["padding_mask"] = torch.arange(10) >= 7
    batch = _collator(mm_tokenizer)([sample])
    assert batch["num_valid_tokens"] == 7
    assert (batch["labels"][7:] == IGNORE_INDEX).all()
    torch.testing.assert_close(sample["labels"], original_labels)


def test_collator_concatenates_whole_samples_before_padding(mm_tokenizer, token_ids):
    batch = _collator(mm_tokenizer)(
        [_mm_sample(10, token_ids), _mm_sample(5, token_ids)]
    )
    assert batch["num_valid_tokens"] == 15
    assert batch["padding_mask"][:15].tolist() == [False] * 15
    assert batch["padding_mask"][15:].tolist() == [True] * 113


def test_multimodal_collator_rejects_rows_over_the_token_batch(mm_tokenizer, token_ids):
    with pytest.raises(ValueError, match="exceed the configured token batch"):
        _collator(mm_tokenizer)([_mm_sample(129, token_ids)])


def test_collator_carries_the_special_token_ids_into_the_batch(mm_tokenizer, token_ids):
    """The model forward reads these off the batch rather than off the
    tokenizer, so every name has to survive the trip."""
    batch = _collator(mm_tokenizer)([_mm_sample(4, token_ids)])
    assert batch["special_tokens"] == {
        "image_id": token_ids["image"],
        "video_id": token_ids["video"],
        "vision_start_id": token_ids["vision_start"],
        "vision_end_id": token_ids["vision_end"],
        "pad_id": token_ids["pad"],
    }


def test_collator_emits_no_media_when_the_batch_has_no_images(mm_tokenizer, token_ids):
    batch = _collator(mm_tokenizer)([_mm_sample(4, token_ids)])
    assert batch["pixel_values"] is None
    assert batch["grid_thw"] is None


def test_collator_packs_image_patches_with_their_grids(mm_tokenizer, image_bytes):
    sample = _mm_sample(8, _ids_of(mm_tokenizer))
    sample["pixel_values"] = [
        process_image(
            image_bytes,
            patch_size=16,
            merge_size=2,
            min_pixels=32 * 32,
            max_pixels=10**8,
        )
    ]
    batch = _collator(mm_tokenizer)([sample])
    assert batch["grid_thw"].shape == (1, 3)
    assert batch["pixel_values"].shape[0] == int(batch["grid_thw"][0].prod())


def _ids_of(mm_tokenizer) -> dict[str, int]:
    return {
        "image": mm_tokenizer.image_id,
        "video": mm_tokenizer.video_id,
        "vision_start": mm_tokenizer.vision_start_id,
        "vision_end": mm_tokenizer.vision_end_id,
        "pad": mm_tokenizer.pad_id,
    }


def test_collator_enforces_the_per_batch_media_budget(mm_tokenizer, image_bytes):
    collator = _collator(mm_tokenizer, max_images_per_batch=1)
    sample = _mm_sample(8, _ids_of(mm_tokenizer))
    sample["pixel_values"] = [
        process_image(
            image_bytes,
            patch_size=16,
            merge_size=2,
            min_pixels=32 * 32,
            max_pixels=10**8,
        )
    ]
    with pytest.raises(ValueError, match="max_images_per_batch"):
        collator([sample, sample])


@pytest.mark.parametrize("merge_size", [0, -1, True, 1.5, None])
def test_mrope_rejects_invalid_merge_size(merge_size):
    with pytest.raises(ValueError, match="spatial_merge_size"):
        build_mrope_positions(
            torch.tensor([1]),
            None,
            None,
            None,
            spatial_merge_size=merge_size,
            patch_order="block",
            image_token_id=2,
            video_token_id=3,
        )


@pytest.mark.parametrize("media_kind", ["image", "video"])
@pytest.mark.parametrize(
    "grid",
    [
        torch.tensor([1, 2, 2]),
        torch.tensor([[1, 2]]),
        torch.tensor([[1.5, 2.0, 2.0]]),
        torch.tensor([[True, True, True]]),
        torch.tensor([[0, 2, 2]]),
        torch.tensor([[-1, 2, 2]]),
    ],
)
def test_mrope_rejects_invalid_grid_geometry(media_kind, grid):
    with pytest.raises(ValueError, match="MRoPE grid_thw"):
        build_mrope_positions(
            torch.tensor([2 if media_kind == "image" else 3]),
            grid if media_kind == "image" else None,
            grid if media_kind == "video" else None,
            None,
            spatial_merge_size=2,
            patch_order="block",
            image_token_id=2,
            video_token_id=3,
        )


def test_mrope_empty_tokens_return_empty_coordinates():
    result = build_mrope_positions(
        torch.empty(0, dtype=torch.long),
        None,
        None,
        None,
        spatial_merge_size=2,
        patch_order="block",
        image_token_id=2,
        video_token_id=3,
    )
    assert result.shape == (0, 3)
    assert result.dtype == torch.long


def test_mrope_rejects_a_raster_patch_order(mm_tokenizer):
    """MRoPE coordinates index the block-ordered patch sequence, so a raster
    order would leave every coordinate pointing at the wrong patch."""
    collator = _collator(mm_tokenizer, patch_order="raster", build_mrope_positions=True)
    ids = _ids_of(mm_tokenizer)
    with pytest.raises(ValueError, match="MRoPE requires patch_order='block'"):
        build_mrope_positions(
            torch.zeros(4, dtype=torch.long),
            torch.tensor([[1, 2, 2]]),
            None,
            torch.zeros(4, dtype=torch.long),
            spatial_merge_size=collator.spatial_merge_size,
            patch_order=collator.patch_order,
            image_token_id=ids["image"],
            video_token_id=ids["video"],
        )


def test_mrope_rejects_media_grid_count_mismatches(mm_tokenizer):
    collator = _collator(mm_tokenizer, build_mrope_positions=True)
    ids = _ids_of(mm_tokenizer)
    tokens = torch.tensor([ids["image"], VOCAB["lorem"]])

    with pytest.raises(ValueError, match="media/grid mismatch"):
        build_mrope_positions(
            tokens,
            None,
            None,
            torch.arange(2),
            spatial_merge_size=collator.spatial_merge_size,
            patch_order=collator.patch_order,
            image_token_id=ids["image"],
            video_token_id=ids["video"],
        )


def test_mrope_rejects_placeholder_lengths_that_disagree_with_the_grid(mm_tokenizer):
    collator = _collator(mm_tokenizer, build_mrope_positions=True)
    ids = _ids_of(mm_tokenizer)
    # A [1, 4, 4] raw grid with spatial_merge_size=2 requires four LLM
    # placeholder tokens, but this prompt contains only three.
    tokens = torch.tensor([ids["image"]] * 3 + [VOCAB["lorem"]])

    with pytest.raises(ValueError, match="placeholder run has 3 token.*requires 4"):
        build_mrope_positions(
            tokens,
            torch.tensor([[1, 4, 4]]),
            None,
            torch.arange(4),
            spatial_merge_size=collator.spatial_merge_size,
            patch_order=collator.patch_order,
            image_token_id=ids["image"],
            video_token_id=ids["video"],
        )


def test_mrope_treats_adjacent_image_and_video_placeholders_as_two_runs(
    mm_tokenizer,
):
    collator = _collator(mm_tokenizer, build_mrope_positions=True)
    ids = _ids_of(mm_tokenizer)
    tokens = torch.tensor([ids["image"], ids["video"], VOCAB["lorem"]])

    positions = build_mrope_positions(
        tokens,
        torch.tensor([[1, 2, 2]]),
        torch.tensor([[1, 2, 2]]),
        torch.arange(3),
        spatial_merge_size=collator.spatial_merge_size,
        patch_order=collator.patch_order,
        image_token_id=ids["image"],
        video_token_id=ids["video"],
    )

    assert positions.shape == (3, 3)


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_mrope_separates_identical_placeholders_across_documents(
    mm_tokenizer, media_kind
):
    collator = _collator(mm_tokenizer, build_mrope_positions=True)
    token_id = getattr(mm_tokenizer, f"{media_kind}_id")
    grids = torch.tensor([[1, 2, 4], [1, 2, 4]])
    mrope = build_mrope_positions(
        torch.full((4,), token_id),
        grids if media_kind == "image" else None,
        grids if media_kind == "video" else None,
        torch.tensor([0, 1, 0, 1]),
        spatial_merge_size=collator.spatial_merge_size,
        patch_order=collator.patch_order,
        image_token_id=mm_tokenizer.image_id,
        video_token_id=mm_tokenizer.video_id,
    )
    expected = torch.tensor([[0, 0, 0], [0, 0, 1]] * 2)
    torch.testing.assert_close(mrope, expected)


def test_mrope_positions_restart_at_each_document(mm_tokenizer):
    """Two documents in one row are positioned independently, which is what the
    position reset encodes: document 2 does not continue document 1's count."""
    collator = _collator(mm_tokenizer, build_mrope_positions=True)
    ids = _ids_of(mm_tokenizer)
    tokens = torch.full((6,), VOCAB["lorem"], dtype=torch.long)
    tokens[3] = ids["image"]
    mrope = build_mrope_positions(
        tokens,
        torch.tensor([[1, 2, 2]]),
        None,
        torch.tensor([0, 1, 2, 0, 1, 2]),
        spatial_merge_size=collator.spatial_merge_size,
        patch_order=collator.patch_order,
        image_token_id=ids["image"],
        video_token_id=ids["video"],
    )
    assert mrope.shape == (6, 3)
    # The second document's first token sits at 0 on all three axes.
    assert mrope[3].tolist() == [0, 0, 0]


# --------------------------------------------------------------------------
# Multimodal packing
# --------------------------------------------------------------------------


def test_mm_packing_rejects_a_non_positive_bin_count(mm_tokenizer):
    with pytest.raises(ValueError, match="num_packing_bins must be positive"):
        build_multimodal_packing(
            SingleDataset(
                source=IndexedJsonlSource(patterns=("unused",)),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=0,
            context=make_context(mm_tokenizer, num_tokens_per_batch=1024),
            iteration_policy=make_policy(),
        )


def test_mm_packing_flattens_per_document_media_into_one_list(mm_tokenizer):
    """Packing merges N documents into one row, so the media lists have to
    flatten with them or the model receives a nested list it cannot index."""
    output = {
        "input_ids": np.zeros(8, dtype=np.int64),
        "labels": np.zeros(8, dtype=np.int64),
        "positions": np.arange(8),
        "input_ids_segment_ids": np.array([1, 1, 1, 1, 0, 0, 0, 0]),
        "padding_mask": np.zeros(8, dtype=np.bool_),
        "pixel_values": [["a", "b"], ["c"]],
        "pixel_values_videos": [["v"]],
    }
    packed = packing_output_to_multimodal_sample(output, max_context_length=32)
    assert packed["pixel_values"] == ["a", "b", "c"]
    assert packed["pixel_values_videos"] == ["v"]
    assert bool(packed["padding_mask"][4:].all())
    assert bool((packed["labels"][4:] == IGNORE_INDEX).all())
    assert bool((packed["labels"][:4] == 0).all())
    assert bool((output["labels"] == 0).all())
    assert int(packed["positions"][4:].max()) < 32


@pytest.mark.parametrize(
    "parts,counts",
    [
        (["text"], [2]),
        ([None], [-1]),
        ([None], [True]),
        ([None], [1.5]),
        ([None], ["2"]),
        ([None], [None]),
    ],
)
def test_vision_placeholders_reject_invalid_counts(parts, counts):
    with pytest.raises(ValueError):
        insert_vision_placeholders(
            parts,
            counts,
            vision_start_token="[start]",
            vision_token="[vision]",
            vision_end_token="[end]",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 0},
        {"width": -1},
        {"factor": 0},
        {"min_pixels": 0},
        {"min_pixels": 2048, "max_pixels": 1024},
        {"max_pixels": 100},
    ],
)
def test_smart_resize_rejects_invalid_geometry(kwargs):
    parameters = dict(height=64, width=64, factor=32, min_pixels=1, max_pixels=4096)
    with pytest.raises(ValueError):
        smart_resize(**(parameters | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"min_frames": 0},
        {"min_frames": 5, "max_frames": 4},
    ],
)
def test_video_sampling_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        load_video("unused.mp4", **kwargs)


@pytest.mark.parametrize("reported_frames", [0, 10])
def test_video_sampling_matches_with_and_without_frame_metadata(
    monkeypatch, reported_frames
):
    import sys
    from types import SimpleNamespace

    class Container:
        streams = SimpleNamespace(
            video=[
                SimpleNamespace(average_rate=5, frames=reported_frames, duration=None)
            ]
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def seek(self, offset):
            pass

        def decode(self, **kwargs):
            for index in range(10):
                yield SimpleNamespace(
                    to_ndarray=lambda format, index=index: np.full(
                        (2, 2, 3), index, dtype=np.uint8
                    )
                )

    monkeypatch.setitem(
        sys.modules, "av", SimpleNamespace(open=lambda path: Container())
    )
    result = load_video("sample.mp4", fps=2, min_frames=1, max_frames=8)
    assert result is not None
    assert result[:, 0, 0, 0].tolist() == [0, 3, 6, 9]


def test_remote_image_rejects_http_errors_before_decoding(monkeypatch):
    import requests

    from llmtuner.datasets.multimodal import image as image_module

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def raise_for_status(self):
            raise requests.HTTPError("404")

        @property
        def content(self):
            raise AssertionError("HTTP error body must not be decoded")

    monkeypatch.setattr(image_module.requests, "get", lambda *a, **k: Response())
    with pytest.raises(requests.HTTPError, match="404"):
        image_module.decode_image("https://example.invalid/image.png")
    assert calls == ["closed"]


@pytest.mark.parametrize("frames", [1, 2, 3, 5])
@pytest.mark.parametrize("merge_size", [1, 2])
def test_vision_token_count_matches_extracted_patches(frames, merge_size):
    image = torch.zeros(frames, 32, 48, 3)
    patches, grid = vision_to_patches(image, 8, 2, merge_size)
    total, columns, rows = calculate_vision_tokens(frames, 32, 48, 8, merge_size, 2)
    assert total == patches.shape[0] // merge_size**2
    assert columns == int(grid[2]) // merge_size
    assert rows == int(grid[1]) // merge_size


@pytest.mark.parametrize(
    "field",
    [
        "num_frames",
        "height",
        "width",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_vision_token_count_rejects_nonpositive_geometry(field, value):
    geometry = dict(
        num_frames=1,
        height=32,
        width=32,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    geometry[field] = value
    with pytest.raises(ValueError, match="must be positive"):
        calculate_vision_tokens(**geometry)


def test_vision_token_count_rejects_partial_merge_grid():
    with pytest.raises(ValueError, match="must be divisible"):
        calculate_vision_tokens(1, 48, 32, 16, 2, 2)
    with pytest.raises(ValueError, match="must be divisible"):
        vision_to_patches(torch.zeros(1, 48, 32, 3), 16, 2, 2)


@pytest.mark.parametrize(
    "shape", [(0, 32, 32, 3), (1, 0, 32, 3), (1, 32, 32, 0), (32, 32, 3)]
)
def test_vision_to_patches_rejects_empty_or_wrong_rank_tensors(shape):
    with pytest.raises(ValueError, match="nonempty"):
        vision_to_patches(torch.zeros(shape), 16, 2, 2)


def test_raster_patches_do_not_require_block_merge_divisibility():
    patches, grid = vision_to_patches(
        torch.zeros(1, 48, 32, 3), 16, 2, 2, patch_order="raster"
    )
    assert grid.tolist() == [1, 3, 2]
    assert patches.shape[0] == 6


@pytest.mark.parametrize(
    "field", ["patch_size", "temporal_patch_size", "spatial_merge_size"]
)
def test_multimodal_constructors_reject_zero_patch_sizes(mm_tokenizer, field):
    with pytest.raises(ValueError, match="patch sizes must be positive"):
        _collator(mm_tokenizer, **{field: 0})
    with pytest.raises(ValueError, match="patch sizes must be positive"):
        MultimodalProcessor(
            context=make_context(mm_tokenizer),
            sample_processor=process_cc12_wd_sample,
            **{field: 0},
        )


@pytest.mark.parametrize(
    "resize_fn, budgets",
    [
        (resize_to_pixel_budget, dict(min_pixels=1024, max_pixels=4096)),
        (resize_to_navit_patch_grid, dict(max_patches=16, max_patches_per_side=8)),
    ],
)
def test_resize_rejects_empty_dimensions_before_division(resize_fn, budgets):
    with pytest.raises(ValueError, match="must be positive"):
        resize_fn(0, 32, patch_size=16, merge_size=2, **budgets)


@pytest.mark.parametrize("height, width", [(32, 6144), (6144, 32), (32, 64)])
@pytest.mark.parametrize(
    "min_pixels, max_pixels", [(65536, 65536), (3000, 4096), (1100, 2048)]
)
def test_smart_resize_respects_tight_budgets_and_narrow_aspect_ratios(
    height, width, min_pixels, max_pixels
):
    h, w = smart_resize(height, width, 32, min_pixels, max_pixels)
    assert h % 32 == 0 and w % 32 == 0
    assert min_pixels <= h * w <= max_pixels


def test_smart_resize_rejects_budgets_with_no_feasible_grid():
    with pytest.raises(ValueError, match="no factor-aligned patch grid"):
        smart_resize(32, 64, 32, 1100, 1200)


def test_smart_resize_preserves_upstream_result_when_already_in_budget():
    assert smart_resize(100, 200, 32, 1024, 100000000) == (96, 192)
