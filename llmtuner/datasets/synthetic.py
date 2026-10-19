"""Deterministic synthetic token batches and a checkpointable DP loader.

RandomTokenSource generates each global batch from its seed and absolute index.
RandomTokenDataLoader slices rows for DP ownership and checkpoints the index.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from .contracts import (
    BaseDataLoader,
    Batch,
    require_positive_integer,
    require_same_dp_degree,
    validate_dp_coordinates,
)

__all__ = [
    "Batch",
    "RandomTokenDataLoader",
    "RandomTokenSource",
]


class RandomTokenSource:
    """A synthetic random-token corpus, standing in for a real dataset.

    Deterministic by construction: batch ``step`` depends only on ``(seed, step)``,
    on every rank. That is what makes two runs comparable and, under data
    parallelism, makes each rank's slice a slice of the *same* global batch.
    """

    def __init__(
        self, *, seed: int, vocab_size: int, batch_size: int, seq_len: int
    ) -> None:
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not -(2**63) <= seed < 2**64
        ):
            raise ValueError(
                f"seed must be an integer in [-2**63, 2**64 - 1], got {seed!r}"
            )
        require_positive_integer("vocab_size", vocab_size)
        require_positive_integer("batch_size", batch_size)
        require_positive_integer("seq_len", seq_len)
        self.seed = seed
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Batch]:
        return self.iter_from(0)

    def iter_from(self, step: int) -> Iterator[Batch]:
        """Generate the deterministic stream starting at an absolute batch index."""
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(
                f"synthetic source step must be a nonnegative integer, got {step!r}"
            )
        while True:
            # A fresh generator per step, so the sequence is reproducible from
            # (seed, step) alone with no dependence on how many steps ran before.
            generator = torch.Generator(device="cpu").manual_seed(
                (self.seed * 100_000 + step) % (2**64)
            )
            input_ids = torch.randint(
                0,
                self.vocab_size,
                (self.batch_size, self.seq_len),
                generator=generator,
            )
            # Next-token prediction: labels are the input shifted by one at loss
            # time, so the same tensor serves as both.
            yield Batch(input_ids=input_ids, labels=input_ids.clone())
            step += 1


class RandomTokenDataLoader(BaseDataLoader):
    """The synthetic corpus behind the ``BaseDataLoader`` contract.

    Each rank takes a contiguous slice of every global batch, which is the
    static form of the dataset graph's ``apply_iteration_policy``. Slicing
    here rather than in the trainer keeps the two
    loaders interchangeable: by the time a batch leaves either one it is
    already this rank's shard, and the loop does not need to know which.

    ``vocab_size``, ``batch_size`` and ``seq_len`` are the shapes the loop
    would otherwise have to know to build the source itself. They are not
    derived from ``num_tokens_per_batch`` the way the Grain path's
    ``num_tokens_per_batch`` is, because the two loaders disagree on what a
    batch means: this one yields ``(B, T)`` rows of one document each, the
    Grain one a flat packed token stream.
    """

    def __init__(
        self,
        *,
        seed: int,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        validate_dp_coordinates(dp_rank=dp_rank, dp_world_size=dp_world_size)
        source = RandomTokenSource(
            seed=seed, vocab_size=vocab_size, batch_size=batch_size, seq_len=seq_len
        )
        if batch_size % dp_world_size != 0:
            raise ValueError(
                f"batch_size={batch_size} not divisible by "
                f"dp_world_size={dp_world_size}"
            )
        self._dp_world_size = dp_world_size
        rows_per_rank = batch_size // dp_world_size
        start = dp_rank * rows_per_rank
        self._rows = slice(start, start + rows_per_rank)
        self._source = source
        self._iterator = iter(self._source)
        # Absolute global batch cursor, independent of DP slicing. Each batch is
        # reproducible from (seed, step), so restoring it needs no replay.
        self._position = 0

    def __iter__(self) -> Iterator[Batch]:
        while True:
            # Read the current stream each time: restoring a cursor must also
            # take effect for an iterator already held by the training loop.
            batch = next(self._iterator)
            self._position += 1
            yield Batch(
                input_ids=batch.input_ids[self._rows],
                labels=batch.labels[self._rows],
            )

    def state_dict(self) -> dict[str, Any]:
        # See ``_position``: the position is what a resume needs.
        return {
            "dp_world_size": self._dp_world_size,
            "steps": self._position,
            "batch_layout": {
                "seed": self._source.seed,
                "vocab_size": self._source.vocab_size,
                "batch_size": self._source.batch_size,
                "seq_len": self._source.seq_len,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError(
                "RandomTokenDataLoader checkpoint state must be a dictionary"
            )
        if not state_dict:
            return
        require_same_dp_degree(state_dict.get("dp_world_size"), self._dp_world_size)
        saved_layout = state_dict.get("batch_layout")
        if (
            saved_layout is not None
            and saved_layout != self.state_dict()["batch_layout"]
        ):
            raise ValueError(
                "cannot resume after changing synthetic data configuration"
            )
        if "steps" not in state_dict:
            raise ValueError("checkpoint is missing synthetic loader steps")
        resume_at = state_dict["steps"]
        if (
            not isinstance(resume_at, int)
            or isinstance(resume_at, bool)
            or resume_at < 0
        ):
            raise ValueError(
                "synthetic loader steps must be a nonnegative integer, "
                f"got {resume_at!r}"
            )
        if resume_at < self._position:
            raise ValueError(
                f"cannot resume at batch {resume_at}: this loader has already "
                f"yielded {self._position}. A fresh loader is required."
            )
        self._iterator = self._source.iter_from(resume_at)
        self._position = resume_at
