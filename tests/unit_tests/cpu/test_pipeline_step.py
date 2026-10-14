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
    }
    chunks = split_pipeline_microbatches(batch, num_microbatches=2)
    assert [chunk["source"] for chunk in chunks] == ["corpus", "corpus"]
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
