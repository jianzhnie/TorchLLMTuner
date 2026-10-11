"""Pipeline batch splitting contracts without starting a process group."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.datasets.types import Batch
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.trainer.pipeline_step import split_pipeline_microbatches


def _trainer(num_microbatches: int):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            parallel=SimpleNamespace(num_pp_microbatches=num_microbatches)
        )
    )


def test_synthetic_batches_split_on_rows() -> None:
    batch = Batch(
        input_ids=torch.arange(24).reshape(4, 6),
        labels=torch.arange(24).reshape(4, 6),
    )
    chunks = split_pipeline_microbatches(_trainer(2), batch)
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
    chunks = split_pipeline_microbatches(_trainer(2), batch)
    assert [chunk["source"] for chunk in chunks] == ["corpus", "corpus"]
    torch.testing.assert_close(chunks[0]["input_ids"], batch["input_ids"][:2])
    torch.testing.assert_close(chunks[1]["labels"], batch["labels"][2:])


def test_packed_stream_rejects_multiple_microbatches() -> None:
    batch = {"labels": torch.arange(8), "input_ids": torch.arange(8)}
    with pytest.raises(UnsupportedCombinationError, match="packed"):
        split_pipeline_microbatches(_trainer(2), batch)
