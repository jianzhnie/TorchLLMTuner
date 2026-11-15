"""Vocabulary shard geometry shared by model embeddings and loss services."""


def vocab_shard_bounds(
    global_vocab_size: int, tp_world_size: int, tp_rank: int
) -> tuple[int, int]:
    """The ``[start, end)`` slice of the vocabulary that ``tp_rank`` owns.

    One definition, used by both the vocab-parallel embedding and the
    vocab-parallel loss. They *must* agree -- if the embedding gathers token ids
    from one shard and the loss attributes the target log-probability to another,
    every loss value is wrong while every shape still checks out.

    The shards are even-sized except possibly the last: with ``V=10, tp=3`` the
    chunks are ``[0,4)``, ``[4,8)``, ``[8,10)``. Every rank's start is clamped to
    ``V`` as well as its end, so a TP degree greater than the vocabulary yields
    empty (not negative) slices, which the callers reject explicitly.
    """
    if (
        not isinstance(global_vocab_size, int)
        or isinstance(global_vocab_size, bool)
        or global_vocab_size < 1
    ):
        raise ValueError(f"global_vocab_size must be >= 1, got {global_vocab_size}")
    if (
        not isinstance(tp_world_size, int)
        or isinstance(tp_world_size, bool)
        or tp_world_size < 1
    ):
        raise ValueError(f"tp_world_size must be >= 1, got {tp_world_size}")
    if (
        not isinstance(tp_rank, int)
        or isinstance(tp_rank, bool)
        or not 0 <= tp_rank < tp_world_size
    ):
        raise ValueError(f"tp_rank {tp_rank} is outside [0, {tp_world_size})")
    chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
    start = min(global_vocab_size, chunk_size * tp_rank)
    end = min(global_vocab_size, start + chunk_size)
    return start, max(start, end)
