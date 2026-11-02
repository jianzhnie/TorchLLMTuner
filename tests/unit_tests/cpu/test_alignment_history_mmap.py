"""Mapped observations remain readable after replacing their history entry."""

import torch

from tests.integration_tests.alignment_history import AlignmentHistory


def test_replacing_observation_preserves_existing_mapped_reader(tmp_path):
    history = AlignmentHistory(tmp_path / "history", writable=True)
    history.append({"input": torch.arange(1024)})
    previous = history[0]
    history[0] = {"input": torch.zeros(2)}
    torch.testing.assert_close(previous["input"], torch.arange(1024))
    torch.testing.assert_close(history[0]["input"], torch.zeros(2))
