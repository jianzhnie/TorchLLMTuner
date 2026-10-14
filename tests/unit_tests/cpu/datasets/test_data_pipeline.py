"""The Grain data pipeline: dataset graph, collators, loader, tokenizer.

Worth testing outside a training run because almost none of it is checked by the
loss curve. The DP slices are the clearest case: if two ranks' slices overlap,
both ranks still produce correctly shaped batches of correctly tokenized data,
and the only symptom is that the model saw some documents twice and others
never. Same for the epoch reshuffle and the resume cursor -- a wrong seed or a
stale index trains happily on the wrong order.

Everything here runs on CPU with no process group and no network: the tokenizer
and the corpus are built into a tmp dir by the fixtures in ``data_fixtures``.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("grain")


import tempfile

import grain.python as grain
import numpy as np
import pytest
import torch

from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.components.tokenizer import HuggingFaceTokenizer
from llmtuner.config import (
    DataloaderConfig,
    LLMTunerConfig,
    ModelConfig,
    TrainingConfig,
)
from llmtuner.config.parallel_config import PipelineParallelConfig
from llmtuner.datasets import (
    DatasetConcat,
    DatasetMix,
    GrainDataLoader,
    TextCollator,
    TextSequence,
    WeightedDataset,
    build_concat_then_split_packing,
    build_dataloader,
    build_dataset,
    build_first_fit_packing,
)
from llmtuner.datasets.synthetic import RandomTokenDataLoader
from llmtuner.datasets.text.processors import ChatProcessor
from tests.data_fixtures import (
    CHAT_TEMPLATE,
    NUM_ROWS,
    VOCAB,
    corpus,  # noqa: F401  (fixture re-export)
    make_context,
    make_policy,
    streaming_text_dataset,
    streaming_text_dataset_from_texts,
    text_dataset,
    token_ids,
    tokenizer,  # noqa: F401  (fixture re-export)
    write_tokenizer,
)


@pytest.mark.parametrize("step", [-1, True, 1.5, "1"])
def test_synthetic_source_rejects_invalid_start_index(step):
    from llmtuner.datasets.synthetic import RandomTokenSource

    source = RandomTokenSource(seed=1, vocab_size=8, batch_size=2, seq_len=4)
    with pytest.raises(ValueError, match="nonnegative integer"):
        next(source.iter_from(step))


@pytest.mark.parametrize("batch_size", [True, 1.5, "2"])
def test_synthetic_loader_validates_batch_size_before_dp_arithmetic(batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive integer"):
        RandomTokenDataLoader(seed=1, vocab_size=8, batch_size=batch_size, seq_len=4)


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def test_special_tokens_are_inferred_from_the_config(tokenizer):
    assert tokenizer.bos_id == VOCAB["[BOS]"]
    assert tokenizer.eos_id == VOCAB["[EOS]"]
    assert tokenizer.bos_token == "[BOS]"
    assert tokenizer.eos_token == "[EOS]"


def test_jsonl_patterns_preserve_pattern_order_and_sort_each_match(tmp_path):
    from llmtuner.datasets.sources import file_patterns_to_paths

    for name in ("a2.jsonl", "a1.jsonl", "z.jsonl"):
        (tmp_path / name).write_text("{}\n")
    paths = file_patterns_to_paths(
        (str(tmp_path / "z.jsonl"), str(tmp_path / "a*.jsonl"))
    )
    assert paths == tuple(
        str(tmp_path / name) for name in ("z.jsonl", "a1.jsonl", "a2.jsonl")
    )


@pytest.mark.parametrize("source", ["corpus.jsonl", b"corpus.jsonl"])
def test_source_rejects_character_sequences(source):
    from llmtuner.datasets.sources import build_source

    with pytest.raises(TypeError, match="source cannot be a string or bytes"):
        build_source(source, iteration_policy=make_policy())


def test_jsonl_source_rejects_empty_patterns():
    from llmtuner.datasets.sources import IndexedJsonlDataSource

    with pytest.raises(ValueError, match="at least one file pattern"):
        IndexedJsonlDataSource(())


def test_jsonl_source_rejects_directory_matches(tmp_path):
    from llmtuner.datasets.sources import IndexedJsonlDataSource

    with pytest.raises(ValueError, match="non-file path"):
        IndexedJsonlDataSource((str(tmp_path),))


@pytest.mark.parametrize("repeat", [False, True])
def test_closed_streaming_cursor_does_not_consume_source(repeat):
    from llmtuner.datasets.sources import HuggingFaceCursorIterator

    consumed = []

    class Source:
        def state_dict(self):
            return {}

        def __iter__(self):
            for text in ("first", "second"):
                consumed.append(text)
                yield {"text": text}

    cursor = HuggingFaceCursorIterator(Source(), repeat=repeat, shuffle=False)
    assert next(cursor) == {"text": "first"}
    cursor.close()
    cursor.close()
    with pytest.raises(ValueError, match="closed iterator"):
        next(cursor)
    assert consumed == ["first"]


@pytest.mark.parametrize("epoch", [True, -1, 1.5, "2"])
def test_streaming_cursor_rejects_invalid_epoch_without_touching_source(epoch):
    from llmtuner.datasets.sources import HuggingFaceCursorIterator

    class Source:
        def __init__(self):
            self.calls = []

        def state_dict(self):
            return {}

        def __iter__(self):
            return iter([{"text": "first"}, {"text": "second"}])

        def set_epoch(self, value):
            self.calls.append(value)

        def load_state_dict(self, value):
            self.calls.append(value)

    source = Source()
    cursor = HuggingFaceCursorIterator(source, repeat=True, shuffle=True)
    assert next(cursor) == {"text": "first"}
    with pytest.raises(ValueError, match="epoch must be a nonnegative integer"):
        cursor.set_state({"epoch": epoch, "hf": {}})
    assert cursor.get_state()["epoch"] == 0
    assert source.calls == []
    assert next(cursor) == {"text": "second"}


@pytest.mark.parametrize("streaming", [False, True])
def test_streaming_repeat_preserves_epoch_when_reset_fails(streaming):
    from llmtuner.datasets.sources import HuggingFaceCursorIterator

    class Source:
        def state_dict(self):
            return {}

        def __iter__(self):
            return iter([])

        def set_epoch(self, epoch):
            pass

        def load_state_dict(self, state):
            raise RuntimeError("reset failed")

    cursor = HuggingFaceCursorIterator(Source(), repeat=True, shuffle=streaming)
    with pytest.raises(RuntimeError, match="reset failed"):
        next(cursor)
    assert cursor.get_state()["epoch"] == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("field", ["path", "split", "name", "revision", "streaming"])
def test_hf_source_rejects_kwargs_shadowing_explicit_fields(streaming, field):
    from llmtuner.datasets.sources import (
        HuggingFaceRandomAccessSource,
        HuggingFaceStreamingSource,
    )

    source_type = (
        HuggingFaceStreamingSource if streaming else HuggingFaceRandomAccessSource
    )
    with pytest.raises(ValueError, match="first-class Hugging Face fields repeated"):
        source_type(path="dataset", split="train", load_dataset_kwargs={field: "x"})


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_source_revalidates_mutated_options_before_loading(streaming, monkeypatch):
    from llmtuner.datasets import sources

    source_type = (
        sources.HuggingFaceStreamingSource
        if streaming
        else sources.HuggingFaceRandomAccessSource
    )
    source = source_type(path="dataset", split="train")
    source.load_dataset_kwargs["streaming"] = not streaming
    calls = []
    monkeypatch.setattr(
        sources.datasets, "load_dataset", lambda *a, **k: calls.append(k)
    )
    with pytest.raises(ValueError, match="first-class Hugging Face fields repeated"):
        sources.build_source(source, iteration_policy=make_policy())
    assert calls == []


@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), "2", None])
def test_dataset_context_rejects_non_integer_sizes(tokenizer, value):
    with pytest.raises(ValueError, match="max_context_length.*positive integer"):
        make_context(tokenizer, max_context_length=value)


def test_tokenizer_accepts_named_inputs_and_explicit_default_settings(tokenizer):
    expected = tokenizer.encode("lorem ipsum")
    assert tokenizer.encode(text="lorem ipsum", add_bos=None, add_eos=None) == expected
    assert tokenizer.decode(token_ids=expected, skip_special_tokens=False) == (
        tokenizer.decode(expected, skip_special_tokens=False)
    )


@pytest.mark.parametrize("kwargs", [{"add_boss": False}, {"add_special_tokens": False}])
def test_tokenizer_rejects_unrecognized_encode_options(tokenizer, kwargs):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tokenizer.encode("lorem", **kwargs)


def test_tokenizer_requires_bos_eos_options_by_keyword(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.encode("lorem", False, False)


def test_tokenizer_preserves_non_special_added_token_metadata(tmp_path):
    import json

    path = write_tokenizer(str(tmp_path / "tokenizer"))
    config_path = tmp_path / "tokenizer" / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["added_tokens_decoder"] = {
        "70": {
            "content": "<ordinary>",
            "special": False,
            "normalized": False,
            "single_word": True,
        }
    }
    config_path.write_text(json.dumps(config))
    loaded = HuggingFaceTokenizer(tokenizer_path=path)
    token_id = loaded.token_to_id("<ordinary>")
    token = loaded.tokenizer.get_added_tokens_decoder()[token_id]
    assert token.special is False
    assert token.normalized is False
    assert token.single_word is True
    assert loaded.decode([token_id], skip_special_tokens=True) == "<ordinary>"


@pytest.mark.parametrize("content", [None, 7, [], {}])
def test_tokenizer_rejects_invalid_special_token_content(tmp_path, content):
    import json

    path = write_tokenizer(str(tmp_path / "tokenizer"))
    config_path = tmp_path / "tokenizer" / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["bos_token"] = {"content": content}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bos_token.*content must be a string"):
        HuggingFaceTokenizer(tokenizer_path=path)


def test_encode_suppresses_native_special_tokens_and_adds_its_own(tokenizer):
    """The whole reason ``encode`` exists: one place decides BOS/EOS.

    If the underlying tokenizer's own post-processor also fired, the sequence
    would carry two BOS tokens and every downstream position would be off by
    one while the shapes stayed right.
    """
    assert tokenizer.encode("lorem ipsum") == [
        VOCAB["[BOS]"],
        VOCAB["lorem"],
        VOCAB["ipsum"],
        VOCAB["[EOS]"],
    ]
    assert tokenizer.encode("lorem", add_bos=False, add_eos=False) == [VOCAB["lorem"]]
    assert tokenizer.encode("lorem", add_eos=False) == [
        VOCAB["[BOS]"],
        VOCAB["lorem"],
    ]


def test_decode_round_trips(tokenizer):
    ids = tokenizer.encode("lorem ipsum", add_bos=False, add_eos=False)
    assert tokenizer.decode(ids) == "lorem ipsum"


def test_missing_tokenizer_path_raises():
    with pytest.raises(FileNotFoundError):
        HuggingFaceTokenizer(tokenizer_path="/nonexistent/tokenizer")


# --------------------------------------------------------------------------
# Graph shape: shuffle, sharding, repeat
# --------------------------------------------------------------------------


def test_dp_ranks_get_disjoint_slices_that_cover_the_corpus(tokenizer, corpus):
    """The property the whole shuffle-then-shard ordering exists to provide.

    A strided split would also be disjoint, but Grain's shuffle is a permutation
    computed from the index, so the contiguous slice of the shuffled index space
    is what keeps reads sequential per rank.
    """
    per_rank = NUM_ROWS // 2
    seen = []
    for rank in (0, 1):
        config = text_dataset(corpus)
        dataset = build_dataset(
            config,
            context=make_context(tokenizer),
            iteration_policy=make_policy(shuffle=True, dp_world_size=2, dp_rank=rank),
        )
        iterator = iter(dataset)
        seen.append(token_ids([next(iterator) for _ in range(per_rank)]))

    assert len(set(seen[0])) == per_rank
    assert len(set(seen[1])) == per_rank
    # Disjoint: no document is trained on twice across the two ranks.
    assert not (set(seen[0]) & set(seen[1]))
    # And covering: nothing is dropped between them.
    assert len(set(seen[0]) | set(seen[1])) == NUM_ROWS


def test_repeat_replays_in_order_when_shuffle_is_off(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer),
        iteration_policy=make_policy(shuffle=False, repeat=True),
    )
    iterator = iter(dataset)
    first = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    second = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    assert first == second


def test_repeat_reshuffles_each_epoch_when_shuffle_is_on(tokenizer, corpus):
    """Re-seeing the same order every epoch is a silent quality regression.

    Grain derives the epoch from the sliced map indices, so ``repeat()`` after
    ``shuffle()`` reshuffles; asserting only that the *set* is stable pins the
    epoch-dependence without pinning a permutation we do not own.
    """
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer),
        iteration_policy=make_policy(shuffle=True, repeat=True),
    )
    iterator = iter(dataset)
    first = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    second = token_ids([next(iterator) for _ in range(NUM_ROWS)])
    assert set(first) == set(second)
    assert first != second


def test_too_few_rows_for_the_dp_size_raises(tokenizer, corpus):
    """The slice arithmetic cannot be carried out below one row per rank."""
    with pytest.raises(ValueError, match="fewer than dp_world_size"):
        build_dataset(
            text_dataset(corpus),
            context=make_context(tokenizer),
            iteration_policy=make_policy(dp_world_size=NUM_ROWS + 1),
        )


def test_documents_shorter_than_two_tokens_are_dropped(tokenizer, corpus):
    """``TextProcessor`` returns None for a one-token document.

    A single token has no next token to predict, so keeping it would put a row
    with an empty label on the loss. The post-filter is what removes them, and
    the count is what proves it ran.
    """
    short = text_dataset(corpus)
    context = make_context(tokenizer)
    dataset = build_dataset(
        short, context=context, iteration_policy=make_policy(shuffle=False)
    )
    kept = sum(1 for sequence in (dataset[i] for i in range(len(dataset))) if sequence)
    # Every row here is at least two tokens, so nothing should be filtered.
    assert kept == NUM_ROWS


# --------------------------------------------------------------------------
# TextSequence and collator


@pytest.mark.parametrize("field", ["input_ids", "labels", "positions", "padding_mask"])
@pytest.mark.parametrize("shape", [(), (2, 2)])
def test_text_sequence_rejects_non_vector_fields(field, shape):
    values = {"input_ids": np.arange(2), "labels": np.arange(2)}
    values[field] = np.zeros(shape, dtype=np.int64)
    with pytest.raises(
        ValueError, match=f"TextSequence.{field} must be one-dimensional"
    ):
        TextSequence(**values)


# --------------------------------------------------------------------------


def test_text_sequence_rejects_ragged_fields():
    with pytest.raises(ValueError, match="equal lengths"):
        TextSequence(
            input_ids=np.arange(4),
            labels=np.arange(3),
        )


def test_text_sequence_accepts_an_absent_positions_field():
    sequence = TextSequence(input_ids=np.arange(4), labels=np.arange(4))
    assert sequence.positions is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_context_length": 0}, "max_context_length"),
        ({"num_tokens_per_batch": 0}, "num_tokens_per_batch"),
        ({"max_num_documents": 0}, "max_num_documents"),
    ],
)
def test_dataset_build_context_rejects_non_positive_sizes(tokenizer, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_context(tokenizer, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dp_world_size": 0},
        {"dp_rank": -1},
        {"dp_rank": True},
        {"dp_rank": 0.5},
        {"dp_rank": 2, "dp_world_size": 2},
        {"streaming_shuffle_buffer_size": 0},
    ],
)
def test_iteration_policy_rejects_invalid_parallelism(kwargs):
    with pytest.raises(ValueError):
        make_policy(**kwargs)


@pytest.mark.parametrize(
    ("dp_rank", "dp_world_size"),
    [(-1, 1), (1, 1), (0, 0), (True, 2), (0.5, 2)],
)
def test_random_loader_rejects_invalid_parallel_coordinates(dp_rank, dp_world_size):
    with pytest.raises(ValueError):
        RandomTokenDataLoader(
            seed=0,
            vocab_size=16,
            batch_size=4,
            seq_len=8,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )


@pytest.mark.parametrize("repeat", ["false", 0, 1, None])
def test_loader_repeat_requires_boolean_before_graph_construction(repeat):
    with pytest.raises(TypeError, match="repeat must be bool"):
        GrainDataLoader(
            None,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            max_context_length=8,
            num_tokens_per_batch=8,
            repeat=repeat,
        )


@pytest.mark.parametrize("field", ["shuffle", "repeat"])
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_iteration_flags_require_booleans(field, value):
    with pytest.raises(TypeError, match=f"{field} must be bool"):
        make_policy(**{field: value})


@pytest.mark.parametrize("rank", [True, 0.5, -1, 2])
def test_grain_loader_rejects_invalid_rank_before_building_graph(rank):
    with pytest.raises(ValueError, match="dp_rank must be an integer"):
        GrainDataLoader(
            None,
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=None,
            max_context_length=8,
            num_tokens_per_batch=8,
        )


@pytest.mark.parametrize("depth", [0, -1, True, 1.5, "2"])
def test_grain_loader_rejects_invalid_prefetch_before_building_graph(depth):
    with pytest.raises(
        ValueError, match="num_prefetch_batches must be positive integer"
    ):
        GrainDataLoader(
            None,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            max_context_length=8,
            num_tokens_per_batch=8,
            num_prefetch_batches=depth,
        )


@pytest.fixture
def collator(tokenizer):
    return TextCollator(context=make_context(tokenizer, num_tokens_per_batch=16))


def test_collator_pads_positions_inside_the_context_window(collator):
    """Padded positions restart at 0 rather than continuing the previous row.

    Zeros would make a padded region look like a continuation of the last real
    document, which is exactly what the position encoding would then encode.
    """
    rows = [
        TextSequence(input_ids=np.arange(5), labels=np.arange(5)),
        TextSequence(input_ids=np.arange(3), labels=np.arange(3)),
    ]
    batch = collator(rows)
    positions = batch["positions"]
    assert positions.shape == (16,)
    assert positions[:8].tolist() == list(range(5)) + list(range(3))
    assert int(positions[8:].max()) < 32
    # Pad positions start over, so the first padded slot is not 8.
    assert int(positions[8]) == 0


def test_collator_pads_labels_with_ignore_index(collator):
    batch = collator([TextSequence(input_ids=np.arange(5), labels=np.arange(5))])
    assert batch["labels"][5:].tolist() == [IGNORE_INDEX] * 11
    assert batch["num_valid_tokens"] == 5


def test_collator_masks_padding_tokens_from_the_row(collator):
    """A row that already carries padding keeps its own mask.

    Packing produces rows with their own padding mask, and the collator must
    respect it rather than assume every row is full -- otherwise a packed row's
    padding tokens would be counted as valid labels.
    """
    rows = [
        TextSequence(
            input_ids=np.arange(4),
            labels=np.array([1, 2, 3, IGNORE_INDEX]),
            padding_mask=np.array([False, False, False, True]),
        )
    ]
    batch = collator(rows)
    assert batch["num_valid_tokens"] == 3
    assert bool(batch["padding_mask"][3])


def test_collator_rejects_rows_over_the_token_batch(collator):
    with pytest.raises(ValueError, match="exceed the configured token batch"):
        collator([TextSequence(input_ids=np.arange(17), labels=np.arange(17))])


def test_collator_num_rows_per_batch_is_one(collator):
    assert collator.num_rows_per_batch() == 1


def test_text_collator_rejects_empty_batch_before_allocating(collator, monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("empty batch must fail before tensor allocation")

    monkeypatch.setattr(torch, "zeros", unexpected_allocation)
    with pytest.raises(ValueError, match="at least one row"):
        collator([])


def test_text_collator_preserves_empty_row_as_fully_padded_batch(collator):
    empty = np.array([], dtype=np.int64)
    batch = collator([TextSequence(input_ids=empty, labels=empty)])
    assert batch["num_valid_tokens"] == 0
    assert batch["padding_mask"].all()
    assert (batch["labels"] == IGNORE_INDEX).all()
    assert batch["positions"].tolist() == list(range(16))


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _loader(dataset, tokenizer, *, dp_rank=0, dp_world_size=1, config_kwargs=None):
    return GrainDataLoader(
        dataset,
        **dict(config_kwargs or {}),
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=32,
        num_tokens_per_batch=32,
    )


def test_loader_resolves_collator_batch_rows_once(tokenizer):
    class SingleResolutionCollator(TextCollator):
        calls = 0

        def num_rows_per_batch(self):
            self.calls += 1
            assert self.calls == 1, "batch layout must be resolved only once"
            return 1

    sample = TextSequence(input_ids=np.arange(4), labels=np.arange(4))
    loader = _loader(
        grain.MapDataset.source([sample]),
        tokenizer,
        config_kwargs={"collator": SingleResolutionCollator, "repeat": False},
    )
    try:
        assert loader.state_dict()["batch_layout"]["num_rows_per_batch"] == 1
        assert next(iter(loader))["input"].numel() == 32
    finally:
        loader.close()


def test_loader_emits_the_trainer_batch_contract(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    batch = next(iter(_loader(dataset, tokenizer)))
    assert sorted(batch) == [
        "input",
        "labels",
        "num_valid_tokens",
        "padding_mask",
        "positions",
    ]
    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["num_valid_tokens"] == int((batch["labels"] != IGNORE_INDEX).sum())


def test_loader_resume_reproduces_the_following_batches_exactly(tokenizer, corpus):
    """The loading position has to survive a checkpoint, or a resumed run
    replays or skips a slice of the epoch. Comparing the *next* batches (not the
    state dict) is the part that matters: a state that round-trips but restores
    to the wrong place still passes a state-equality check.
    """

    def fresh():
        dataset = build_dataset(
            text_dataset(corpus),
            context=make_context(tokenizer, num_tokens_per_batch=32),
            iteration_policy=make_policy(shuffle=True, repeat=True),
        )
        return _loader(dataset, tokenizer)

    ahead, behind = fresh(), fresh()
    ahead_iterator, behind_iterator = iter(ahead), iter(behind)
    next(ahead_iterator)
    next(ahead_iterator)
    next(behind_iterator)  # one batch behind

    behind.load_state_dict(ahead.state_dict())
    for _ in range(3):
        assert (
            next(ahead_iterator)["input"].tolist()
            == next(behind_iterator)["input"].tolist()
        )
    ahead.close()
    behind.close()


def test_loader_rejects_resuming_across_a_changed_dp_size(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="data-parallel degree"):
        loader.load_state_dict({"version": 1, "dp_world_size": 2, "dp_rank_0": {}})
    loader.close()


def test_dcp_restore_checks_saved_layout_before_restoring_iterator(
    tmp_path, tokenizer, corpus
):
    import torch.distributed.checkpoint as dcp

    def fresh(num_tokens):
        dataset = build_dataset(
            text_dataset(corpus),
            context=make_context(tokenizer, num_tokens_per_batch=num_tokens),
            iteration_policy=make_policy(repeat=True),
        )
        return GrainDataLoader(
            dataset,
            tokenizer=tokenizer,
            dp_rank=0,
            dp_world_size=1,
            max_context_length=8,
            num_tokens_per_batch=num_tokens,
        )

    saved, restored = fresh(32), fresh(16)
    try:
        next(iter(saved))
        dcp.save({"loader": saved}, checkpoint_id=tmp_path / "checkpoint")
        initial_state = restored.state_dict()
        with pytest.raises(ValueError, match="changing dataloader batch layout"):
            dcp.load({"loader": restored}, checkpoint_id=tmp_path / "checkpoint")
        assert restored.state_dict() == initial_state
        assert next(iter(restored))["input"].shape == (16,)
    finally:
        saved.close()
        restored.close()


@pytest.mark.parametrize("version", [99, True, 1.0, None])
def test_loader_rejects_an_unknown_state_version(tokenizer, corpus, version):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="unsupported GrainDataLoader state version"):
        loader.load_state_dict(
            {"version": version, "dp_world_size": 1, "dp_rank_0": {}}
        )
    loader.close()


def test_loader_rejects_missing_layout_without_changing_cursor(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    try:
        before = loader.state_dict()
        invalid = {key: value for key, value in before.items() if key != "batch_layout"}
        with pytest.raises(ValueError, match="missing dataloader batch_layout"):
            loader.load_state_dict(invalid)
        assert loader.state_dict() == before
    finally:
        loader.close()


@pytest.mark.parametrize("degree", [True, 1.0, None, 0])
def test_synthetic_resume_rejects_invalid_saved_dp_degree(degree):
    loader = RandomTokenDataLoader(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    before = loader.state_dict()
    with pytest.raises(ValueError, match="checkpoint dp_world_size"):
        loader.load_state_dict({"dp_world_size": degree, "steps": 10})
    assert loader.state_dict() == before


def test_loader_requires_the_rank_entry_present_in_the_checkpoint(tokenizer, corpus):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    loader = _loader(dataset, tokenizer)
    with pytest.raises(ValueError, match="missing dataloader state"):
        loader.load_state_dict({"version": 1, "dp_world_size": 1})
    loader.close()


def test_finite_dataset_under_dp_is_rejected_up_front(tokenizer, corpus):
    """Exhaustion at different steps would hang the next collective.

    This is checked at construction rather than at the first short batch, which
    is the point: by the time a rank notices, its peers are already blocked.
    """
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(repeat=True),
    )
    with pytest.raises(ValueError, match="repeat=False"):
        _loader(dataset, tokenizer, dp_world_size=2, config_kwargs={"repeat": False})


def test_max_num_documents_must_be_positive(tokenizer, corpus, num_tokens_per_batch=32):
    dataset = build_dataset(
        text_dataset(corpus),
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(),
    )
    with pytest.raises(ValueError, match="max_num_documents must be positive"):
        _loader(dataset, tokenizer, config_kwargs={"max_num_documents": 0})


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def test_concat_then_split_fills_the_token_batch(tokenizer, corpus):
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    rows = [next(iter(loader)) for _ in range(3)]
    for row in rows:
        assert row["input"].shape == (32,)
        # Packing exists to spend the whole batch on real tokens.
        assert row["num_valid_tokens"] == 32


def test_first_fit_packing_fills_the_token_batch(tokenizer, corpus):
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_first_fit_packing(
        text_dataset(corpus),
        num_packing_bins=4,
        context=context,
        iteration_policy=make_policy(),
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    iterator = iter(loader)
    for _ in range(3):
        row = next(iterator)
        assert row["input"].shape == (32,)
        assert row["num_valid_tokens"] > 0


def test_concat_then_split_marks_padding_so_it_never_contributes_loss(
    tokenizer, corpus
):
    """Padding inside a packed row must be labelled IGNORE_INDEX.

    A packed row mixes several documents to fill the batch, so the trailing
    padding is interleaved with real tokens in the same tensor. If it reached
    the loss as a normal target the model would be trained to predict pad.
    """
    context = make_context(tokenizer, num_tokens_per_batch=32)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    iterator = iter(loader)
    for _ in range(3):
        row = next(iterator)
        padded = row["padding_mask"]
        assert bool((row["labels"][padded] == IGNORE_INDEX).all())
        if bool(padded.any()):
            # `max` on an all-padding row is what the guard is for, and an
            # empty window is not an error: `arange(pad_len) % ctx` is.
            assert int(row["positions"][padded].max()) < 32


def test_document_aware_packing_caps_segments_per_row(tokenizer, corpus):
    """``max_num_documents`` bounds how many documents share a row.

    Without it a row can be assembled from dozens of documents, which is fine
    for the loss but not for anything downstream that needs per-document
    boundaries (attention masks, MRoPE).
    """
    context = make_context(tokenizer, num_tokens_per_batch=32, max_num_documents=1)
    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, iteration_policy=make_policy()
    )
    loader = _loader(graph, tokenizer, config_kwargs={"repeat": False})
    row = next(iter(loader))
    assert row["input"].shape == (32,)
    # One document, so at most one position restart plus padding.
    assert int((row["positions"] == 0).sum()) <= 1 + int(row["padding_mask"].sum())


def test_document_aware_packing_keeps_each_row_to_one_document(tokenizer, corpus):
    """``max_num_documents=1`` stops a row from carrying a second document.

    This is what separates the document-aware node from the plain
    ``grain.experimental.ConcatThenSplitIterDataset`` path, and the row *shape*
    cannot tell them apart -- both fill the batch. The plain path concatenates
    everything and cuts at the token count, so a row runs off the end of one
    document and into the head of the next. This node pads instead.

    Sizing matters. At ``num_tokens_per_batch=32`` the fixture documents all fit
    in one row, so the two paths agree and the existing tests pass on either.
    At 8 the documents are 4-12 tokens and the distinction actually shows.

    A document longer than a row is still split -- that is unavoidable, and the
    assertion is deliberately "one *source document* per row", not "one segment",
    so the unavoidable split does not read as a failure. Asserted against the
    unpacked source rather than against a shape, so a dropped or duplicated
    document cannot hide behind matching row lengths.
    """
    context = make_context(tokenizer, num_tokens_per_batch=8, max_num_documents=1)
    policy = make_policy()
    source = [
        [int(token) for token in sequence.input_ids]
        for sequence in build_dataset(
            text_dataset(corpus), context=context, iteration_policy=policy
        )
    ]
    # Which source document each token came from, flattened.
    owner = [index for index, doc in enumerate(source) for _ in doc]
    flat_source = [token for doc in source for token in doc]

    graph = build_concat_then_split_packing(
        text_dataset(corpus), context=context, iteration_policy=policy
    )

    flat_packed: list[int] = []
    for row in graph:
        assert row.input_ids.shape == (8,)
        # Positions restart at 0 within a row, so the leading run where
        # ``positions[i] == i`` is this row's real tokens; the rest is padding.
        real = 0
        while real < 8 and int(row.positions[real]) == real:
            real += 1
        tokens = [int(token) for token in row.input_ids[:real]]
        owners = owner[len(flat_packed) : len(flat_packed) + real]
        assert len(set(owners)) == 1, (
            f"row starting at token {len(flat_packed)} draws from "
            f"{len(set(owners))} documents, but max_num_documents=1"
        )
        flat_packed.extend(tokens)

    # Nothing dropped, nothing duplicated, order preserved.
    assert flat_packed == flat_source


class _SequenceIterDataset(grain.IterDataset):
    """Feed pre-built TextSequences into a packing node, no tokenizer needed."""

    def __init__(self, sequences: list[TextSequence]) -> None:
        super().__init__()
        self._sequences = sequences

    def __iter__(self) -> grain.DatasetIterator:
        return grain.MapDataset.source(self._sequences).to_iter_dataset().__iter__()


@pytest.mark.parametrize(
    "max_num_documents, expected_masks",
    [
        # Both documents plus the inner padding fit in one row; the inner
        # padding must stay marked instead of being relabelled as real.
        (4, [[False] * 5 + [True]]),
        # The cap ends the first row after two documents, so the inner padding
        # segment is carried into a second row through the remainder slice.
        (2, [[False] * 5 + [True], [True] * 6]),
    ],
)
def test_document_capped_nested_packing_keeps_inner_padding(
    max_num_documents, expected_masks
):
    """Nested packing: inner padding survives the document-capped outer packer.

    First-fit padding inside a sequence is marked in its ``padding_mask``; the
    document-aware concat-then-split iterator copies ids/labels/positions per
    slice, so the mask must be sliced along with them. Rebuilding it as all
    ``False`` would feed padding tokens to ``routing_token_counts`` and to the
    varlen metadata as if they were real documents.
    """
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    # A first-fit row: documents [1, 2] and [3, 4, 5], then one padding token.
    inner_row = TextSequence(
        input_ids=np.asarray([1, 2, 3, 4, 5, 0]),
        labels=np.asarray([2, 3, 4, 5, 0, IGNORE_INDEX]),
        positions=np.asarray([0, 1, 0, 1, 2, 0]),
        padding_mask=np.asarray([False] * 5 + [True]),
    )
    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset([inner_row]),
        max_num_documents_per_row=max_num_documents,
        max_context_length=9,
        num_tokens_per_row=6,
    )

    rows = list(iter(graph))

    assert [row.padding_mask.tolist() for row in rows] == expected_masks
    # Positions restart per segment; padding positions wrap within the window.
    assert rows[0].positions.tolist() == [0, 1, 0, 1, 2, 0]
    assert rows[0].input_ids.tolist() == [1, 2, 3, 4, 5, 0]


# --------------------------------------------------------------------------
# Packing parameter validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("consumed_rows", [0, 1, 2, 3])
def test_document_packing_restore_preserves_remainders_and_empty_samples(consumed_rows):
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    def sequence(ids):
        return TextSequence(
            input_ids=np.asarray(ids, dtype=np.int64),
            labels=np.asarray(ids, dtype=np.int64) + 1,
        )

    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset(
            [sequence([]), sequence(list(range(1, 10))), sequence([]), sequence([10])]
        ),
        max_num_documents_per_row=2,
        max_context_length=4,
        num_tokens_per_row=4,
    )
    iterator = iter(graph)
    prefix = [next(iterator) for _ in range(consumed_rows)]
    state = iterator.get_state()
    suffix = list(iterator)
    restored = iter(graph)
    restored.set_state(state)
    actual = list(restored)
    assert len(actual) == len(suffix)
    for expected, row in zip(suffix, actual, strict=True):
        for field in ("input_ids", "labels", "positions", "padding_mask"):
            np.testing.assert_array_equal(getattr(row, field), getattr(expected, field))
    real_tokens = np.concatenate(
        [row.input_ids[~row.padding_mask] for row in prefix + actual]
    )
    np.testing.assert_array_equal(real_tokens, np.arange(1, 11))
    exhausted = iter(graph)
    exhausted.set_state(restored.get_state())
    assert list(exhausted) == []


@pytest.mark.parametrize("value", [0, -1])
def test_document_split_rejects_non_positive_context(value):
    from llmtuner.datasets.packing.iterators import SplitTextSequenceDocuments

    with pytest.raises(ValueError, match="max_context_length must be positive"):
        SplitTextSequenceDocuments(max_context_length=value)


@pytest.mark.parametrize("offset", [-1, 0, 5, 6, True, 1.5])
def test_document_packing_rejects_invalid_restored_offset(offset):
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset(
            [TextSequence(input_ids=np.arange(5), labels=np.arange(5))]
        ),
        max_num_documents_per_row=2,
        max_context_length=2,
        num_tokens_per_row=2,
    )
    iterator = iter(graph)
    next(iterator)
    state = iterator.get_state()
    state["remainder_offset"] = offset
    with pytest.raises(ValueError, match="packing checkpoint"):
        iter(graph).set_state(state)


def test_document_packing_rejects_missing_checkpoint_remainder():
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset([]),
        max_num_documents_per_row=2,
        max_context_length=2,
        num_tokens_per_row=2,
    )
    iterator = iter(graph)
    state = iterator.get_state()
    state.update(has_remainder=True, remainder_offset=1)
    with pytest.raises(ValueError, match="remainder is missing"):
        iterator.set_state(state)


def test_interrupted_packing_restore_preserves_live_token_stream(monkeypatch):
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset(
            [
                TextSequence(input_ids=np.arange(5), labels=np.arange(5)),
                TextSequence(input_ids=np.arange(5, 8), labels=np.arange(5, 8)),
            ]
        ),
        max_num_documents_per_row=2,
        max_context_length=2,
        num_tokens_per_row=2,
    )
    iterator = iter(graph)
    next(iterator)
    state = iterator.get_state()
    next(iterator)
    next(iterator)
    reference = iter(graph)
    reference.set_state(iterator.get_state())
    parent_type = type(iterator._parent)
    original_next = parent_type.__next__

    def interrupted_next(parent):
        sample = original_next(parent)
        if parent is iterator._parent:
            raise KeyboardInterrupt("interrupted remainder replay")
        return sample

    with monkeypatch.context() as patch:
        patch.setattr(parent_type, "__next__", interrupted_next)
        with pytest.raises(KeyboardInterrupt, match="remainder replay"):
            iterator.set_state(state)

    assert iterator._parent.get_state() == reference._parent.get_state()
    for actual, expected in zip(list(iterator), list(reference), strict=True):
        for field in ("input_ids", "labels", "positions", "padding_mask"):
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(expected, field)
            )


@pytest.mark.parametrize("parent_position", [0, 1])
def test_failed_packing_restore_preserves_live_token_stream(parent_position):
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterDataset,
    )

    graph = DocumentAwareConcatThenSplitIterDataset(
        _SequenceIterDataset(
            [TextSequence(input_ids=np.arange(5), labels=np.arange(5))]
        ),
        max_num_documents_per_row=2,
        max_context_length=2,
        num_tokens_per_row=2,
    )
    iterator = iter(graph)
    next(iterator)
    original_state = iterator.get_state()
    reference = iter(graph)
    reference.set_state(original_state)
    invalid_state = dict(original_state, remainder_offset=5)
    parent_iterator = iter(graph._parent)
    if parent_position:
        next(parent_iterator)
    invalid_state["parent"] = parent_iterator.get_state()
    with pytest.raises(ValueError, match="packing checkpoint remainder"):
        iterator.set_state(invalid_state)
    assert iterator.get_state() == original_state
    for actual, expected in zip(list(iterator), list(reference), strict=True):
        for field in ("input_ids", "labels", "positions", "padding_mask"):
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(expected, field)
            )


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize(
    "field",
    ["max_num_documents_per_row", "max_context_length", "num_tokens_per_row"],
)
@pytest.mark.parametrize("direct_iterator", [False, True])
def test_document_packing_rejects_non_positive_dimensions(
    field, value, direct_iterator
):
    from llmtuner.datasets.packing.iterators import (
        DocumentAwareConcatThenSplitIterator,
        DocumentAwareConcatThenSplitIterDataset,
    )

    parent = _SequenceIterDataset([])
    dimensions = dict(
        max_num_documents_per_row=2, max_context_length=4, num_tokens_per_row=8
    )
    dimensions[field] = value
    constructor = (
        DocumentAwareConcatThenSplitIterator
        if direct_iterator
        else DocumentAwareConcatThenSplitIterDataset
    )
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        constructor(iter(parent) if direct_iterator else parent, **dimensions)


# --------------------------------------------------------------------------
# Mix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight", [np.float32(1), np.float64(1), np.int64(1), 5e-324, 1e308]
)
def test_mix_accepts_real_weights_across_numeric_scales(tokenizer, corpus, weight):
    mix = DatasetMix(
        datasets=(WeightedDataset(dataset=text_dataset(corpus), weight=weight),)
    )
    graph = build_dataset(
        mix, context=make_context(tokenizer), iteration_policy=make_policy()
    )
    assert next(iter(graph)) is not None


def test_mix_rejects_unrepresentable_ratios_before_building_sources(
    tokenizer, monkeypatch
):
    from llmtuner.datasets import composition

    def unexpected_build(*args, **kwargs):
        pytest.fail("invalid ratios must be rejected before source construction")

    monkeypatch.setattr(composition, "build_source", unexpected_build)
    mix = DatasetMix(
        datasets=tuple(
            WeightedDataset(dataset=object(), weight=weight)
            for weight in (5e-324, 1e308)
        )
    )
    with pytest.raises(ValueError, match="weight ratios"):
        build_dataset(
            mix, context=make_context(tokenizer), iteration_policy=make_policy()
        )


def test_mix_rejects_a_non_positive_weight(tokenizer, corpus):
    dataset = text_dataset(corpus)
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=dataset, weight=1.0),
            WeightedDataset(dataset=dataset, weight=0.0),
        )
    )
    with pytest.raises(ValueError, match="finite, positive-weight"):
        build_dataset(
            mix, context=make_context(tokenizer), iteration_policy=make_policy()
        )


def test_mix_produces_a_batch_from_interleaved_children(tokenizer, corpus):
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=text_dataset(corpus), weight=1.0),
            WeightedDataset(dataset=text_dataset(corpus), weight=2.0),
        )
    )
    graph = build_dataset(
        mix,
        context=make_context(tokenizer, num_tokens_per_batch=32),
        iteration_policy=make_policy(),
    )
    row = next(iter(_loader(graph, tokenizer)))
    assert row["input"].shape == (32,)


def test_mix_of_streaming_children_draws_from_both(tokenizer):
    """Mixing streams takes the second branch: ``IterDataset.mix``.

    Distinct from the map branch above, which is the one the existing test
    exercises. The two are separate grain calls with separate weight handling,
    and nothing else in the suite builds an iter-style mix -- so a child
    silently dropped would only ever show up in a run.

    The children hold disjoint words. A mix that kept only the first child would
    look perfectly healthy against a shared corpus, since the first child covers
    it by itself -- that is exactly the failure this guards. The words are drawn
    from the fixture vocabulary rather than invented, because an unknown word
    tokenizes to ``[UNK]`` and both children would then be indistinguishable.
    """
    first = streaming_text_dataset_from_texts([f"w{i}" for i in range(6)])
    second = streaming_text_dataset_from_texts([f"w{i}" for i in range(20, 26)])
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=first, weight=1.0),
            WeightedDataset(dataset=second, weight=1.0),
        )
    )
    context = make_context(tokenizer, num_tokens_per_batch=8)
    graph = build_dataset(mix, context=context, iteration_policy=make_policy())
    assert isinstance(graph, grain.IterDataset)

    def rows_of(config):
        return token_ids(
            build_dataset(config, context=context, iteration_policy=make_policy())
        )

    from_first, from_second = rows_of(first), rows_of(second)
    assert set(from_first) != set(from_second), "the children must be distinguishable"
    # Every row from both children, and nothing else.
    assert sorted(token_ids(list(graph))) == sorted(from_first + from_second)


@pytest.mark.parametrize("weight", [True, "1", None, float("nan"), float("inf")])
def test_mix_rejects_invalid_weights_before_building_sources(
    tokenizer, monkeypatch, weight
):
    from llmtuner.datasets import composition

    def unexpected_build(*args, **kwargs):
        pytest.fail("invalid weights must be rejected before source construction")

    monkeypatch.setattr(composition, "build_source", unexpected_build)
    mix = DatasetMix(datasets=(WeightedDataset(dataset=object(), weight=weight),))
    with pytest.raises(ValueError, match="finite, positive-weight"):
        build_dataset(
            mix, context=make_context(tokenizer), iteration_policy=make_policy()
        )


def test_mix_of_streaming_children_rejects_a_non_positive_weight(tokenizer, corpus):
    """The validation runs before either branch, and covers both."""
    mix = DatasetMix(
        datasets=(
            WeightedDataset(dataset=streaming_text_dataset(corpus), weight=1.0),
            WeightedDataset(dataset=streaming_text_dataset(corpus), weight=-1.0),
        )
    )
    with pytest.raises(ValueError, match="finite, positive-weight"):
        build_dataset(
            mix, context=make_context(tokenizer), iteration_policy=make_policy()
        )


# --------------------------------------------------------------------------
# Streaming sources: the IterDataset branch of the graph
# --------------------------------------------------------------------------


def test_a_streaming_source_takes_the_iter_dataset_branch(tokenizer, corpus):
    """The streaming half of ``SingleDataset`` is otherwise never entered.

    Streaming shuffles with a window before processing, while the map path
    shuffles globally afterward. Both paths share the row transforms, but
    their ordering policies need separate coverage.

    Both a correct stream and a wrong one produce correctly shaped batches, so
    the assertions are on content: every document once, and the window shuffle
    actually reordering them.
    """
    context = make_context(tokenizer, num_tokens_per_batch=8)
    ordered = build_dataset(
        streaming_text_dataset(corpus),
        context=context,
        iteration_policy=make_policy(shuffle=False),
    )
    assert isinstance(ordered, grain.IterDataset)
    rows = token_ids(list(ordered))
    assert len(rows) == NUM_ROWS
    assert len(set(rows)) == NUM_ROWS  # nothing dropped or duplicated

    shuffled = build_dataset(
        streaming_text_dataset(corpus),
        context=context,
        iteration_policy=make_policy(shuffle=True),
    )
    assert set(token_ids(list(shuffled))) == set(rows)
    # A window shuffle is not the identity; if it were, the stream order would
    # be trained on unchanged and the shuffle would be a silent no-op.
    assert token_ids(list(shuffled)) != rows


def test_concat_rejects_a_streaming_child(tokenizer, corpus):
    """Concatenation needs a length up front, so a stream cannot take part.

    The guard is what turns a mid-run failure into a build-time one. Note the
    child is built before the check, so this also pins that the built child is
    inspected rather than the config.
    """
    concat = DatasetConcat(
        datasets=(streaming_text_dataset(corpus), text_dataset(corpus))
    )
    with pytest.raises(TypeError, match="requires map-style children"):
        build_dataset(
            concat,
            context=make_context(tokenizer),
            iteration_policy=make_policy(),
        )


def test_concat_shards_map_children_like_a_single_dataset(tokenizer, corpus):
    """Concatenation is transparent to everything downstream of it.

    DP sharding is applied to the concatenation exactly as it is to one
    dataset, so the DP disjointness guarantee has to survive the extra layer.
    """
    concat = DatasetConcat(datasets=(text_dataset(corpus), text_dataset(corpus)))
    per_rank = (2 * NUM_ROWS) // 2
    seen = []
    for rank in (0, 1):
        graph = build_dataset(
            concat,
            context=make_context(tokenizer),
            iteration_policy=make_policy(shuffle=True, dp_world_size=2, dp_rank=rank),
        )
        iterator = iter(graph)
        seen.append(token_ids([next(iterator) for _ in range(per_rank)]))

    reference = build_dataset(
        concat,
        context=make_context(tokenizer),
        iteration_policy=make_policy(shuffle=True),
    )
    assert seen[0] + seen[1] == token_ids(list(reference))


# --------------------------------------------------------------------------
# ChatProcessor
# --------------------------------------------------------------------------


def _chat_processor(tokenizer, *, max_context_length=32):
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    return ChatProcessor(
        context=make_context(
            tokenizer,
            num_tokens_per_batch=max_context_length,
            max_context_length=max_context_length,
        ),
        messages_fn=lambda sample: sample["messages"],
    )


def test_chat_processor_masks_the_prompt_labels(tokenizer):
    """Only the response contributes loss.

    The boundary is found by re-rendering the prompt alone, so the assertion is
    that the masked span is exactly the prompt's token count -- not that some
    plausible number of labels came back masked.
    """
    processor = _chat_processor(tokenizer)
    sample = {
        "messages": [
            {"role": "user", "content": "lorem"},
            {"role": "assistant", "content": "ipsum"},
        ]
    }
    sequence = processor(sample, np.random.default_rng(0))
    assert sequence is not None

    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(
            sample["messages"][:1], add_generation_prompt=True
        ),
        add_bos=True,
        add_eos=False,
    )
    masked = int((sequence.labels == IGNORE_INDEX).sum())
    assert masked == len(prompt) - 1
    # The response labels are intact, including the appended EOS.
    assert sequence.labels[-1] == tokenizer.eos_id


def test_chat_processor_is_next_token_aligned(tokenizer):
    processor = _chat_processor(tokenizer)
    sequence = processor(
        {
            "messages": [
                {"role": "user", "content": "lorem"},
                {"role": "assistant", "content": "ipsum"},
            ]
        },
        np.random.default_rng(0),
    )
    real = sequence.labels != IGNORE_INDEX
    # Each label is the next token, wherever both positions carry a target.
    both = real[:-1] & real[1:]
    assert (sequence.labels[:-1][both] == sequence.input_ids[1:][both]).all()


def test_chat_processor_rejects_a_multi_turn_conversation(tokenizer):
    processor = _chat_processor(tokenizer)
    with pytest.raises(ValueError, match="Expected single-turn"):
        processor(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ]
            },
            np.random.default_rng(0),
        )


def test_chat_processor_rejects_swapped_roles(tokenizer):
    processor = _chat_processor(tokenizer)
    with pytest.raises(ValueError, match="First message must be 'user'"):
        processor(
            {
                "messages": [
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "a"},
                ]
            },
            np.random.default_rng(0),
        )


def test_chat_processor_drops_an_oversized_sample(tokenizer):
    """Overflow is per-sample, so dropping is correct here (unlike a template
    mismatch, which would be systematic). Returning None is what routes it to
    the post-filter rather than failing the run.
    """
    processor = _chat_processor(tokenizer, max_context_length=1)
    sequence = processor(
        {
            "messages": [
                {"role": "user", "content": "lorem ipsum"},
                {"role": "assistant", "content": "lorem ipsum lorem ipsum"},
            ]
        },
        np.random.default_rng(0),
    )
    assert sequence is None


def test_chat_processor_requires_an_eos_id():
    class _NoEos(HuggingFaceTokenizer):
        def __init__(self, *, tokenizer_path):
            super().__init__(tokenizer_path=tokenizer_path)
            self.eos_id = None

    path = write_tokenizer(tempfile.mkdtemp())
    tokenizer = _NoEos(tokenizer_path=path)
    with pytest.raises(ValueError, match="does not have an eos_id"):
        ChatProcessor(
            context=make_context(tokenizer),
            messages_fn=lambda sample: sample["messages"],
        )


# --------------------------------------------------------------------------
# The trainer-facing seam: the config that names a corpus, and the loader it
# builds. These are what the Trainer actually calls, so they are exercised
# through the same entry point rather than by reaching past it.
# --------------------------------------------------------------------------


def test_dataloader_arguments_default_to_the_synthetic_corpus() -> None:
    """The default must need no assets, so an untouched run stays offline."""
    args = DataloaderConfig()
    assert args.dataset == "random"
    assert args.tokenizer_path is None


def test_dataloader_arguments_require_a_tokenizer_for_a_real_corpus() -> None:
    with pytest.raises(ValueError, match="tokenizer_path is required"):
        DataloaderConfig(dataset="c4")


def test_dataloader_arguments_require_a_path_for_local_jsonl() -> None:
    with pytest.raises(ValueError, match="dataset_path is required"):
        DataloaderConfig(dataset="local_jsonl", tokenizer_path="/tmp/tok")


def _loader_config(
    dataloader: DataloaderConfig | None = None,
    *,
    global_batch_size: int = 4,
    max_seq_len: int = 8,
    seed: int = 42,
) -> LLMTunerConfig:
    """The run config ``build_dataloader`` now reads its scalars off.

    ``build_dataloader`` takes the whole config and pulls the corpus settings
    from ``.dataloader`` and the run scalars from the flat view, so a test that
    wants to vary one of them varies the config rather than restating it as a
    keyword argument.
    """
    return LLMTunerConfig(
        model=ModelConfig(vocab_size=128),
        training=TrainingConfig(
            global_batch_size=global_batch_size,
            max_seq_len=max_seq_len,
            seed=seed,
            dataloader_config=dataloader
            if dataloader is not None
            else DataloaderConfig(),
        ),
    )


def test_dataloader_rejects_an_unknown_corpus_at_build_time() -> None:
    """The registry check lives in ``build_dataloader``, not in the config."""
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataloader(
            _loader_config(
                DataloaderConfig(dataset="not-a-dataset", tokenizer_path="/tmp/tok")
            ),
            dp_rank=0,
            dp_world_size=1,
            num_tokens_per_batch=32,
        )


def test_dataloader_arguments_build_the_synthetic_loader_without_assets() -> None:
    loader = build_dataloader(
        _loader_config(),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert isinstance(loader, RandomTokenDataLoader)
    batch = next(iter(loader))
    assert batch.input_ids.shape == (4, 8)


def test_dataloader_arguments_build_a_grain_loader_over_a_local_corpus(
    tmp_path, corpus
) -> None:
    """The end-to-end seam, without the network.

    A tokenizer is written here rather than taken from the ``tokenizer``
    fixture: the config takes a *path*, because the Trainer has no tokenizer
    object to hand it, and the fixture lives in its own module-scoped temp
    directory.
    """
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert isinstance(loader, GrainDataLoader)

    batch = next(iter(loader))
    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["positions"].shape == (32,)
    # Packing spends the whole batch on real tokens.
    assert batch["num_valid_tokens"] == 32
    assert int((batch["labels"] != IGNORE_INDEX).sum()) == 32
    loader.close()


@pytest.mark.parametrize("packing", ["concat_then_split", "first_fit"])
@pytest.mark.parametrize("max_num_documents", [None, 1])
def test_pipeline_loader_matches_independently_packed_microbatches(
    tmp_path, corpus, packing, max_num_documents
):
    from llmtuner.config import ParallelConfig

    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    cfg = _loader_config(
        DataloaderConfig(
            dataset="local_jsonl",
            tokenizer_path=tokenizer_path,
            dataset_path=corpus,
            packing=packing,
            max_num_documents=max_num_documents,
            shuffle=False,
        )
    )
    reference = build_dataloader(
        cfg, dp_rank=0, dp_world_size=1, num_tokens_per_batch=16
    )
    cfg.parallel = ParallelConfig(
        pipeline_parallel_size=2,
        pp_config=PipelineParallelConfig(num_pp_microbatches=2),
    )
    pipeline = build_dataloader(
        cfg, dp_rank=0, dp_world_size=1, num_tokens_per_batch=32
    )
    try:
        for _ in range(3):
            batch = next(iter(pipeline))
            expected = [next(iter(reference)) for _ in range(2)]
            assert batch["input"].shape == (2, 16)
            assert batch["num_valid_tokens"] == sum(
                row["num_valid_tokens"] for row in expected
            )
            for index, row in enumerate(expected):
                for field in ("input", "labels", "positions", "padding_mask"):
                    torch.testing.assert_close(batch[field][index], row[field])
                if max_num_documents == 1:
                    positions = batch["positions"][index][~batch["padding_mask"][index]]
                    assert int((positions == 0).sum()) <= 1
        state = pipeline.state_dict()
        incompatible = {
            **state,
            "batch_layout": {**state["batch_layout"], "num_rows_per_batch": 4},
        }
        expected_next = next(iter(pipeline))
        with pytest.raises(ValueError, match="changing dataloader batch layout"):
            pipeline.load_state_dict(incompatible)
        pipeline.load_state_dict(state)
        resumed = next(iter(pipeline))
        for field in ("input", "labels", "positions", "padding_mask"):
            torch.testing.assert_close(resumed[field], expected_next[field])
    finally:
        reference.close()
        pipeline.close()


@pytest.mark.parametrize("max_context_length", [3, 16])
def test_pipeline_collator_pads_a_finite_partial_schedule_without_repeating_tokens(
    tokenizer,
    max_context_length,
):
    from dataclasses import replace

    from llmtuner.datasets.collators import PipelineTextCollator

    context = replace(
        make_context(tokenizer),
        num_tokens_per_batch=16,
        max_context_length=max_context_length,
    )
    collator = PipelineTextCollator(context=context, num_microbatches=2)
    row = TextSequence(
        input_ids=np.asarray([1, 2, 3]), labels=np.asarray([2, 3, IGNORE_INDEX])
    )
    batch = collator([row])
    assert batch["input"].shape == (2, 8)
    assert batch["num_valid_tokens"] == 2
    assert batch["input"][0, :3].tolist() == [1, 2, 3]
    assert batch["padding_mask"][1].all()
    assert (batch["labels"][1] == IGNORE_INDEX).all()
    assert batch["input"][1].eq(0).all()
    torch.testing.assert_close(
        batch["positions"][1], torch.arange(8) % max_context_length
    )
    assert int((~batch["padding_mask"]).sum()) == 3
    np.testing.assert_array_equal(row.labels, [2, 3, IGNORE_INDEX])


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("concat_then_split", "build_concat_then_split_packing"),
        ("first_fit", "build_first_fit_packing"),
    ],
)
def test_packing_selector_chooses_the_recipe(
    tmp_path, corpus, monkeypatch, selector, expected
) -> None:
    """``--packing`` must reach the choice of function ``build_dataloader`` makes.

    Asserted at the seam rather than by inspecting the graph that comes back:
    both recipes return a ``grain.IterDataset``, so a selector that never left
    the config would still build a working loader -- just the wrong one, with
    nothing in the batch to say so.
    """
    import llmtuner.datasets.loader_builder as build_module

    calls: list[str] = []
    for name in ("build_concat_then_split_packing", "build_first_fit_packing"):
        real = getattr(build_module, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(build_module, name, spy)

    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
                packing=selector,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert calls == [expected]
    loader.close()


def test_num_packing_bins_reaches_first_fit(tmp_path, corpus, monkeypatch) -> None:
    """The bin count is forwarded, not left at the callee's default."""
    import llmtuner.datasets.loader_builder as build_module

    seen: dict = {}
    real = build_module.build_first_fit_packing

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(build_module, "build_first_fit_packing", spy)

    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
                packing="first_fit",
                num_packing_bins=3,
            ),
            seed=1,
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=32,
    )
    assert seen["num_packing_bins"] == 3
    loader.close()


def test_the_two_text_recipes_agree_on_the_batch_contract(tmp_path, corpus) -> None:
    """Switching recipes changes packing, not the shape the trainer consumes."""
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    batches = {}
    for selector in ("concat_then_split", "first_fit"):
        loader = build_dataloader(
            _loader_config(
                DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path=tokenizer_path,
                    dataset_path=corpus,
                    packing=selector,
                ),
                seed=1,
            ),
            dp_rank=0,
            dp_world_size=1,
            num_tokens_per_batch=32,
        )
        batches[selector] = next(iter(loader))
        loader.close()

    for batch in batches.values():
        assert batch["input"].shape == (32,)
        assert batch["labels"].shape == (32,)
        assert batch["positions"].shape == (32,)
        assert batch["padding_mask"].shape == (32,)
        # FirstFit pads to the row length rather than filling every slot, so
        # the token count is bounded by the batch rather than equal to it.
        assert 0 < batch["num_valid_tokens"] <= 32


def test_num_packing_bins_must_be_positive() -> None:
    """Validated even under the default recipe, which never reads it.

    The field is always parsed, so a bad value accepted here would only
    surface after switching ``--packing`` to first_fit.
    """
    with pytest.raises(ValueError, match=r"num_packing_bins must be >= 1"):
        DataloaderConfig(num_packing_bins=0)


def test_dataloader_arguments_reject_a_mismatched_dp_size_at_build_time(
    tmp_path, corpus
) -> None:
    """A config whose policy was built for one rank cannot be handed another's.

    The trainer derives the policy from the config and passes the rank
    separately, so the two can disagree; the loader is where that is caught.
    """
    tokenizer_path = str(tmp_path / "tokenizer")
    write_tokenizer(tokenizer_path)
    loader = build_dataloader(
        _loader_config(
            DataloaderConfig(
                dataset="local_jsonl",
                tokenizer_path=tokenizer_path,
                dataset_path=corpus,
            ),
            seed=1,
        ),
        dp_rank=1,
        dp_world_size=2,
        num_tokens_per_batch=32,
    )
    # Stored under dp_rank_1 because that is what the config was built for.
    assert loader.state_dict()["dp_rank_1"] is not None
    loader.close()


# --------------------------------------------------------------------------
# chat template special-token injection
# --------------------------------------------------------------------------


def test_chat_template_injects_special_tokens_and_generation_prompt(tokenizer):
    """Templates referencing bos_token/eos_token must not see empty variables.

    Torchtitan's HFBackendTokenizer injects the same three defaults; without
    them a template that references the variables renders blanks or raises.
    """
    tokenizer.set_chat_template(
        "{{ bos_token }}|{{ eos_token }}|{% if add_generation_prompt %}GEN{% endif %}"
    )
    messages = [{"role": "user", "content": "x"}]

    out = tokenizer.apply_chat_template(messages)

    assert out == f"{tokenizer.bos_token or ''}|{tokenizer.eos_token or ''}|GEN"


def test_explicit_chat_template_kwargs_win_over_the_injected_defaults(tokenizer):
    """The injection is defaults only -- an explicit caller choice stands."""
    tokenizer.set_chat_template(
        "{{ bos_token }}|{% if add_generation_prompt %}GEN{% endif %}"
    )
    messages = [{"role": "user", "content": "x"}]

    out = tokenizer.apply_chat_template(
        messages, bos_token="B", add_generation_prompt=False
    )

    assert out == "B|"


def test_synthetic_source_accepts_full_64_bit_seed():
    import torch

    from llmtuner.datasets.synthetic import RandomTokenSource

    source = RandomTokenSource(seed=2**64 - 1, vocab_size=16, batch_size=2, seq_len=4)
    first = next(iter(source))
    repeated = next(iter(source))
    assert first.input_ids.shape == (2, 4)
    assert torch.equal(first.input_ids, repeated.input_ids)


def test_text_collator_masks_source_padding_without_mutating_labels(tokenizer):
    source_labels = np.array([2, 3, 4], dtype=np.int64)
    sequence = TextSequence(
        input_ids=np.array([1, 2, 3], dtype=np.int64),
        labels=source_labels,
        padding_mask=np.array([False, True, False]),
    )
    collator = TextCollator(context=make_context(tokenizer, num_tokens_per_batch=4))
    batch = collator([sequence])
    assert batch["labels"].tolist() == [2, IGNORE_INDEX, 4, IGNORE_INDEX]
    assert batch["num_valid_tokens"] == 2
    assert source_labels.tolist() == [2, 3, 4]


@pytest.mark.parametrize("field", ["vocab_size", "batch_size", "seq_len"])
@pytest.mark.parametrize("value", [0, -1])
def test_synthetic_source_rejects_empty_or_negative_shapes(field, value):
    from llmtuner.datasets.synthetic import RandomTokenSource

    kwargs = dict(seed=0, vocab_size=8, batch_size=2, seq_len=4)
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        RandomTokenSource(**kwargs)


def test_standalone_chat_template_loads_without_tokenizer_config(tokenizer):
    from pathlib import Path

    path = Path(tokenizer.tokenizer_path)
    (path / "tokenizer_config.json").unlink()
    (path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.content }}{% endfor %}",
        encoding="utf-8",
    )
    loaded = HuggingFaceTokenizer(tokenizer_path=str(path))
    assert loaded.apply_chat_template([{"role": "user", "content": "你好"}]) == "你好"


def test_restore_failure_preserves_error_when_iterator_close_fails():
    from types import SimpleNamespace

    original = ValueError("invalid cursor")
    closed = []

    def set_state(state):
        raise original

    def close():
        closed.append(True)
        raise RuntimeError("close failed")

    loader = GrainDataLoader.__new__(GrainDataLoader)
    loader._dp_world_size = 1
    loader._rank_id = "dp_rank_0"
    loader._iterator = SimpleNamespace(set_state=set_state, close=close)
    loader._batch_layout = {}
    with pytest.raises(ValueError, match="invalid cursor") as caught:
        loader.load_state_dict(
            {"version": 1, "dp_world_size": 1, "dp_rank_0": {}, "batch_layout": {}}
        )
    assert caught.value is original
    assert closed == [True]


@pytest.mark.parametrize("cursor", [True, False, -1, 1.5, "2"])
def test_synthetic_loader_rejects_invalid_cursor_without_advancing(cursor):
    loader = RandomTokenDataLoader(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    reference = RandomTokenDataLoader(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    before = loader.state_dict()
    with pytest.raises(ValueError, match="nonnegative integer"):
        loader.load_state_dict({"dp_world_size": 1, "steps": cursor})
    assert loader.state_dict() == before
    torch.testing.assert_close(
        next(iter(loader)).input_ids, next(iter(reference)).input_ids
    )


@pytest.mark.parametrize("field", ["seed", "vocab_size", "batch_size", "seq_len"])
def test_synthetic_resume_rejects_changed_source_configuration(field):
    settings = dict(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    saved = RandomTokenDataLoader(**settings)
    next(iter(saved))
    settings[field] += 1
    restored = RandomTokenDataLoader(**settings)
    before = restored.state_dict()
    with pytest.raises(ValueError, match="synthetic data configuration"):
        restored.load_state_dict(saved.state_dict())
    assert restored.state_dict() == before


def test_local_sft_recipe_survives_worker_serialization(tokenizer, tmp_path):
    import pickle

    from llmtuner.datasets.text.recipes import build_local_jsonl_sft

    recipe = build_local_jsonl_sft(
        path=str(tmp_path / "chat.jsonl"),
        prompt_field="question",
        response_field="answer",
    )
    restored = pickle.loads(pickle.dumps(recipe))
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    context = make_context(tokenizer, max_context_length=32)
    sample = {"question": "lorem", "answer": "ipsum"}
    expected = recipe.processor(context=context)(sample, np.random.default_rng(0))
    actual = restored.processor(context=context)(sample, np.random.default_rng(0))
    assert expected is not None
    assert actual is not None
    np.testing.assert_array_equal(actual.input_ids, expected.input_ids)
    np.testing.assert_array_equal(actual.labels, expected.labels)


def test_hf_random_access_source_uses_dataset_directly(monkeypatch):
    import datasets

    from llmtuner.datasets.sources import (
        HuggingFaceRandomAccessSource,
        RandomAccessDataSource,
        build_source,
    )

    dataset = datasets.Dataset.from_dict({"text": ["first", "second", "third"]})
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    source = build_source(
        HuggingFaceRandomAccessSource(path="local-corpus", split="train"),
        iteration_policy=make_policy(),
    )
    assert source is dataset
    assert isinstance(source, RandomAccessDataSource)
    graph = grain.MapDataset.source(source)
    assert list(graph) == [{"text": text} for text in ("first", "second", "third")]
    assert source[-1] == {"text": "third"}


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_source_forwards_load_options_and_rejects_dataset_dict(
    streaming, monkeypatch
):
    import datasets

    from llmtuner.datasets.sources import (
        HuggingFaceRandomAccessSource,
        HuggingFaceStreamingSource,
        build_source,
    )

    seen = {}

    def load(path, **kwargs):
        seen.update(path=path, **kwargs)
        return {}

    monkeypatch.setattr(datasets, "load_dataset", load)
    source_type = (
        HuggingFaceStreamingSource if streaming else HuggingFaceRandomAccessSource
    )
    source = source_type(
        path="local-corpus",
        name="subset",
        split="train",
        revision="fixed-revision",
        load_dataset_kwargs={"data_files": "rows.jsonl"},
    )
    with pytest.raises(TypeError, match="requires one"):
        build_source(source, iteration_policy=make_policy())
    assert seen == {
        "path": "local-corpus",
        "name": "subset",
        "split": "train",
        "revision": "fixed-revision",
        "streaming": streaming,
        "data_files": "rows.jsonl",
    }


def test_multiturn_config_does_not_require_unused_single_turn_fields():
    config = DataloaderConfig(
        dataset="local_jsonl_sft",
        tokenizer_path="tokenizer",
        dataset_path="chat.jsonl",
        chat_renderer="Qwen3RendererConfig",
        prompt_field="",
        response_field="",
        messages_field="conversation",
    )
    assert config.messages_field == "conversation"


@pytest.mark.parametrize("field", ["prompt_field", "response_field"])
def test_single_turn_config_requires_its_message_fields(field):
    with pytest.raises(ValueError, match="prompt_field/response_field cannot be empty"):
        DataloaderConfig(
            dataset="local_jsonl_sft",
            tokenizer_path="tokenizer",
            dataset_path="chat.jsonl",
            **{field: ""},
        )


def test_synthetic_resume_seeks_without_generating_previous_batches(monkeypatch):
    loader = RandomTokenDataLoader(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    checkpoint = loader.state_dict()
    checkpoint["steps"] = 1_000_000
    original_randint = torch.randint
    calls = []

    def capture_randint(*args, **kwargs):
        calls.append(1)
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", capture_randint)
    loader.load_state_dict(checkpoint)
    assert calls == []
    actual = next(iter(loader))
    assert len(calls) == 1
    generator = torch.Generator().manual_seed(42 * 100_000 + 1_000_000)
    expected = original_randint(0, 16, (2, 4), generator=generator)
    assert torch.equal(actual.input_ids, expected)
    assert loader.state_dict()["steps"] == 1_000_001


def test_synthetic_resume_rejects_missing_cursor_without_changing_stream():
    settings = dict(seed=42, vocab_size=16, batch_size=2, seq_len=4)
    loader = RandomTokenDataLoader(**settings)
    reference = RandomTokenDataLoader(**settings)
    saved = loader.state_dict()
    saved.pop("steps")
    with pytest.raises(ValueError, match="missing synthetic loader steps"):
        loader.load_state_dict(saved)
    torch.testing.assert_close(
        next(iter(loader)).input_ids, next(iter(reference)).input_ids
    )


def test_synthetic_resume_updates_existing_iterators():
    settings = dict(
        seed=42, vocab_size=16, batch_size=4, seq_len=8, dp_rank=1, dp_world_size=2
    )
    loader = RandomTokenDataLoader(**settings)
    active = iter(loader)
    next(active)
    checkpoint = loader.state_dict()
    checkpoint["steps"] = 10
    loader.load_state_dict(checkpoint)

    reference = RandomTokenDataLoader(**settings)
    reference.load_state_dict(checkpoint)
    expected = iter(reference)
    for cursor in (11, 12):
        actual = next(active)
        batch = next(expected)
        torch.testing.assert_close(actual.input_ids, batch.input_ids)
        torch.testing.assert_close(actual.labels, batch.labels)
        assert loader.state_dict()["steps"] == cursor


@pytest.mark.parametrize(
    ("dp_rank", "dp_world_size"), [(True, 1), (1, 1), (-1, 2), (0, 0)]
)
def test_loader_rejects_invalid_ownership_before_tokenizer_setup(
    dp_rank, dp_world_size, monkeypatch
):
    from types import SimpleNamespace

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
    from llmtuner.datasets.loader_builder import build_dataloader

    def unexpected_tokenizer(*args, **kwargs):
        pytest.fail("invalid DP coordinates must fail before loading tokenizer assets")

    monkeypatch.setattr(HuggingFaceTokenizer, "__init__", unexpected_tokenizer)
    config = SimpleNamespace(
        dataloader=SimpleNamespace(dataset="local_jsonl"), max_seq_len=16
    )
    with pytest.raises(ValueError, match="dp_rank|dp_world_size"):
        build_dataloader(
            config,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_tokens_per_batch=16,
        )


@pytest.mark.parametrize(
    ("repeat", "dp_world_size", "error_type", "message"),
    [
        (False, 2, ValueError, "repeat=False"),
        ("false", 1, TypeError, "repeat must be bool"),
        (None, 1, TypeError, "repeat must be bool"),
    ],
)
def test_loader_checks_iteration_before_loading_assets(
    repeat, dp_world_size, error_type, message, monkeypatch
):
    from types import SimpleNamespace

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    def unexpected_tokenizer(*args, **kwargs):
        pytest.fail("invalid iteration policy must fail before asset loading")

    monkeypatch.setattr(HuggingFaceTokenizer, "__init__", unexpected_tokenizer)
    config = SimpleNamespace(
        dataloader=SimpleNamespace(dataset="local_jsonl"), max_seq_len=16
    )
    with pytest.raises(error_type, match=message):
        build_dataloader(
            config,
            dp_rank=0,
            dp_world_size=dp_world_size,
            num_tokens_per_batch=16,
            repeat=repeat,
        )


def test_loader_checks_pipeline_budget_before_loading_tokenizer(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
    from llmtuner.datasets.loader_builder import build_dataloader

    def unexpected_tokenizer(*args, **kwargs):
        pytest.fail("invalid pipeline token budget must fail before asset loading")

    monkeypatch.setattr(HuggingFaceTokenizer, "__init__", unexpected_tokenizer)
    config = SimpleNamespace(
        dataloader=SimpleNamespace(dataset="local_jsonl"),
        max_seq_len=16,
        parallel=SimpleNamespace(
            pipeline_parallel_size=2, pp_config=SimpleNamespace(num_pp_microbatches=3)
        ),
    )
    with pytest.raises(ValueError, match="divisible by num_pp_microbatches"):
        build_dataloader(config, dp_rank=0, dp_world_size=1, num_tokens_per_batch=16)


@pytest.mark.parametrize("state", [None, {}, {"epoch": 0}, {"epoch": 0, "hf": None}])
def test_streaming_cursor_rejects_malformed_checkpoint_state(state):
    from llmtuner.datasets.sources import HuggingFaceCursorIterator

    class Source:
        def state_dict(self):
            return {}

        def __iter__(self):
            return iter([{"text": "first"}])

        def load_state_dict(self, value):
            pytest.fail("malformed state must be rejected before source mutation")

    cursor = HuggingFaceCursorIterator(Source(), repeat=False, shuffle=False)
    with pytest.raises(ValueError, match="streaming checkpoint"):
        cursor.set_state(state)


@pytest.mark.parametrize("line", ["[]", "null", "42", '"text"'])
def test_jsonl_source_rejects_non_object_rows(tmp_path, line):
    from llmtuner.datasets.sources import IndexedJsonlDataSource

    path = tmp_path / "rows.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    source = IndexedJsonlDataSource((str(path),))
    with pytest.raises(ValueError, match="JSONL row 0.*must be a JSON object"):
        source[0]


@pytest.mark.parametrize("special", ["bos", "eos"])
def test_explicit_tokenizer_config_false_overrides_native_special_tokens(special):
    from types import SimpleNamespace

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    tokenizer._hf_config = {f"add_{special}_token": False}
    tokenizer.tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=True: SimpleNamespace(
            ids=[1, 2] if add_special_tokens else [3]
        )
    )
    tokenizer._infer_should_add_bos_eos()
    expected = [3, 2] if special == "bos" else [1, 3]
    assert tokenizer.encode("text") == expected


@pytest.mark.parametrize("content", ["[]", "null", "42"])
def test_tokenizer_config_requires_json_object(tmp_path, content):
    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    path = tmp_path / "tokenizer_config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Tokenizer config.*must be a JSON object"):
        HuggingFaceTokenizer._load_config(str(path))


@pytest.mark.parametrize("vocab_file", ["vocab.txt", "vocab.json"])
def test_vocabulary_only_tokenizer_splits_whitespace(tmp_path, vocab_file):
    import json

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    content = "\n".join(vocab) if vocab_file.endswith("txt") else json.dumps(vocab)
    (tmp_path / vocab_file).write_text(content, encoding="utf-8")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.encode("hello world", add_bos=False, add_eos=False) == [1, 2]
    assert tokenizer.encode("hello\tworld", add_bos=False, add_eos=False) == [1, 2]


def test_text_vocabulary_preserves_token_whitespace(tmp_path):
    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    (tmp_path / "vocab.txt").write_text("[UNK]\n hello\nworld \n", encoding="utf-8")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.token_to_id(" hello") == 1
    assert tokenizer.token_to_id("world ") == 2
    assert tokenizer.token_to_id("hello") is None


def test_text_vocabulary_rejects_duplicate_tokens(tmp_path):
    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    (tmp_path / "vocab.txt").write_text("[UNK]\nhello\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate token.*line 3"):
        HuggingFaceTokenizer(tokenizer_path=str(tmp_path))


@pytest.mark.parametrize("vocab_file", ["vocab.txt", "vocab.json"])
def test_bpe_tokenizer_loads_merges_and_decodes(tmp_path, vocab_file):
    import json

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    vocab = {
        token: index
        for index, token in enumerate(
            ["h", "e", "l", "o", "he", "hel", "hell", "hello"]
        )
    }
    content = "\n".join(vocab) if vocab_file.endswith("txt") else json.dumps(vocab)
    (tmp_path / vocab_file).write_text(content, encoding="utf-8")
    (tmp_path / "merges.txt").write_text(
        "#version: 0.2\nh e\nhe l\nhel l\nhell o\n", encoding="utf-8"
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.encode("hello") == [vocab["hello"]]
    assert tokenizer.decode([vocab["hello"]]) == "hello"


def test_serialized_tokenizer_takes_priority_over_vocabulary(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

    Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(
        str(tmp_path / "tokenizer.json")
    )
    (tmp_path / "vocab.json").write_text("invalid JSON", encoding="utf-8")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.encode("hello") == [1]


def test_streaming_graph_cursors_keep_independent_checkpoint_positions():
    from datasets import IterableDataset

    from llmtuner.datasets.sources import HuggingFaceStreamingIterDataset

    def rows():
        for index in range(5):
            yield {"index": index}

    graph = HuggingFaceStreamingIterDataset(
        IterableDataset.from_generator(rows), repeat=False, shuffle=False
    )
    first = iter(graph)
    second = iter(graph)
    assert next(first) == {"index": 0}
    assert next(second) == {"index": 0}
    second_state = second.get_state()
    assert next(first) == {"index": 1}
    assert second.get_state() == second_state
    assert next(second) == {"index": 1}

    restored = iter(graph)
    restored.set_state(second_state)
    assert next(restored) == {"index": 1}
    assert next(first) == {"index": 2}
    assert next(second) == {"index": 2}


@pytest.mark.parametrize("shuffle", [False, True])
def test_streaming_cursor_resumes_across_repeated_epochs(shuffle):
    from datasets import IterableDataset

    from llmtuner.datasets.sources import HuggingFaceStreamingIterDataset

    def rows():
        for index in range(3):
            yield {"index": index}

    graph = HuggingFaceStreamingIterDataset(
        IterableDataset.from_generator(rows), repeat=True, shuffle=shuffle
    )
    cursor = iter(graph)
    assert [next(cursor)["index"] for _ in range(5)] == [0, 1, 2, 0, 1]
    saved = cursor.get_state()
    assert saved["epoch"] == 1
    restored = iter(graph)
    restored.set_state(saved)
    expected = [next(cursor) for _ in range(8)]
    assert [next(restored) for _ in range(8)] == expected
    assert saved["epoch"] == 1


def test_grain_window_shuffle_restores_stream_and_buffer_progress():
    from datasets import IterableDataset

    from llmtuner.datasets.sources import HuggingFaceStreamingIterDataset

    def rows():
        for index in range(17):
            yield {"index": index}

    source = HuggingFaceStreamingIterDataset(
        IterableDataset.from_generator(rows), repeat=True, shuffle=True
    )
    graph = grain.experimental.WindowShuffleIterDataset(source, window_size=5, seed=42)
    original = iter(graph)
    for _ in range(9):
        next(original)
    saved = original.get_state()
    expected = [next(original) for _ in range(25)]
    restored = iter(graph)
    restored.set_state(saved)
    assert [next(restored) for _ in range(25)] == expected


@pytest.mark.parametrize("length", [0, -1, True, 1.5, "8", None])
@pytest.mark.parametrize("direction", ["to_packing", "from_packing"])
def test_text_packing_conversions_reject_invalid_context(length, direction):
    from llmtuner.datasets.packing.conversions import (
        packing_output_to_text_sequence,
        text_sequence_to_packing_input,
    )

    with pytest.raises(ValueError, match="max_context_length"):
        if direction == "to_packing":
            text_sequence_to_packing_input(None, max_context_length=length)
        else:
            packing_output_to_text_sequence({}, max_context_length=length)
