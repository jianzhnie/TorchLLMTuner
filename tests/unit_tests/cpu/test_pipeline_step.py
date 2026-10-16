"""Pipeline batch splitting contracts without starting a process group."""

import pytest
import torch

from llmtuner.datasets.contracts import Batch
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.trainer.runtime.pipeline import split_pipeline_microbatches


def test_synthetic_batches_split_on_rows() -> None:
    batch = Batch(
        input_ids=torch.arange(24).reshape(4, 6),
        labels=torch.arange(24).reshape(4, 6),
    )
    chunks = split_pipeline_microbatches(batch, num_microbatches=2)
    assert len(chunks) == 2
    assert all(isinstance(chunk, Batch) for chunk in chunks)
    torch.testing.assert_close(chunks[0].input_ids, batch.input_ids[:2])
    torch.testing.assert_close(chunks[1].labels, batch.labels[2:])


def test_corpus_batches_slice_tensor_rows_and_keep_metadata() -> None:
    batch = {
        "input_ids": torch.arange(24).reshape(4, 6),
        "labels": torch.arange(24).reshape(4, 6),
        "source": "corpus",
        "weight": torch.tensor(0.5),
    }
    chunks = split_pipeline_microbatches(batch, num_microbatches=2)
    assert [chunk["source"] for chunk in chunks] == ["corpus", "corpus"]
    assert all(chunk["weight"] is batch["weight"] for chunk in chunks)
    torch.testing.assert_close(chunks[0]["input_ids"], batch["input_ids"][:2])
    torch.testing.assert_close(chunks[1]["labels"], batch["labels"][2:])


def test_packed_stream_rejects_multiple_microbatches() -> None:
    batch = {"labels": torch.arange(8), "input_ids": torch.arange(8)}
    with pytest.raises(UnsupportedCombinationError, match="packed"):
        split_pipeline_microbatches(batch, num_microbatches=2)


def test_pipeline_split_recounts_valid_tokens_without_mutating_source() -> None:
    labels = torch.tensor([[1, -100, -100], [2, 3, -100]])
    batch = {
        "input": torch.ones_like(labels),
        "labels": labels,
        "num_valid_tokens": 3,
    }
    chunks = split_pipeline_microbatches(batch, num_microbatches=2)
    assert [chunk["num_valid_tokens"] for chunk in chunks] == [1, 2]
    assert batch["num_valid_tokens"] == 3
    torch.testing.assert_close(batch["labels"], labels)


@pytest.mark.parametrize(
    "field, leading_entries",
    [("pixel_values", 6), ("grid_thw", 3), ("mrope_positions", 3)],
)
def test_multiple_pipeline_microbatches_reject_non_row_tensor_layouts(
    field, leading_entries
):
    labels = torch.arange(16).reshape(4, 4)
    batch = {
        "input": labels.clone(),
        "labels": labels,
        field: torch.arange(leading_entries * 8).reshape(leading_entries, 8),
    }
    original = {name: value.clone() for name, value in batch.items()}
    with pytest.raises(ValueError, match=rf"{field!r}.*row-aligned"):
        split_pipeline_microbatches(batch, num_microbatches=2)
    for name, value in batch.items():
        torch.testing.assert_close(value, original[name])


def test_pipeline_split_rejects_mismatched_synthetic_row_counts():
    batch = Batch(input_ids=torch.ones(3, 4), labels=torch.ones(4, 4))
    with pytest.raises(ValueError, match="'input_ids'.*row-aligned"):
        split_pipeline_microbatches(batch, num_microbatches=2)


@pytest.mark.parametrize("num_microbatches", [0, -1, True, 1.5, "2", None])
def test_pipeline_microbatch_degree_is_validated_before_batch_access(num_microbatches):
    with pytest.raises(ValueError, match="num_pp_microbatches.*positive integer"):
        split_pipeline_microbatches({}, num_microbatches=num_microbatches)


@pytest.mark.parametrize("shape", [(), (2, 2, 2)])
def test_pipeline_split_rejects_labels_without_a_supported_row_layout(shape):
    with pytest.raises(ValueError, match="Pipeline batch labels.*rank"):
        split_pipeline_microbatches({"labels": torch.ones(shape)}, num_microbatches=1)


def test_pipeline_split_requires_synthetic_batches_to_have_rows():
    batch = Batch(input_ids=torch.arange(4), labels=torch.arange(4))
    with pytest.raises(ValueError, match="Pipeline batch labels.*rank"):
        split_pipeline_microbatches(batch, num_microbatches=2)


@pytest.mark.parametrize("num_tokens", [2, 8])
def test_single_pipeline_microbatch_preserves_packed_media_and_coordinates(
    num_tokens,
) -> None:
    labels = torch.arange(num_tokens)
    labels[-1] = -100
    batch = {
        "input": torch.arange(num_tokens),
        "labels": labels,
        "positions": torch.arange(num_tokens),
        "pixel_values": torch.arange(num_tokens * 4 * 6).reshape(num_tokens * 4, 6),
        "grid_thw": torch.tensor([[1, 2, num_tokens * 2]]),
        "mrope_positions": torch.arange(3 * num_tokens).reshape(3, num_tokens),
        "num_valid_tokens": num_tokens - 1,
        "special_tokens": {"image_id": 42},
    }
    [microbatch] = split_pipeline_microbatches(batch, num_microbatches=1)
    assert microbatch is not batch
    assert microbatch.keys() == batch.keys()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(microbatch[key], value)
        else:
            assert microbatch[key] == value
    assert microbatch["num_valid_tokens"] == num_tokens - 1
