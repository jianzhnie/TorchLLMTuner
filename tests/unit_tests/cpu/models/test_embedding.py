"""Vocab-parallel embedding: global ``padding_idx`` gradient suppression.

Upstream #4637 (``Fix padding gradients in vocab-parallel embeddings``):
``padding_idx`` is a *global* row id, so a shard that does not own that row must
not hand it to the local ``F.embedding`` -- doing so either crashes (the id is
beyond the shard's row count) or silently zeroes the gradient of an unrelated
row. Only the owning shard passes ``padding_idx - offset``.

Single-rank harness: the TP path only needs ``distributed.get_world_size`` /
``distributed.get_rank`` on a process-group handle, so a 1-rank fake pg plus
patched counts stands in for a real TP group.
"""

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from llmtuner.models.common import embedding as embedding_mod
from llmtuner.models.common.embedding import VocabParallelEmbedding


def _single_rank_fake_pg():
    store = FakeStore()
    dist.init_process_group("fake", store=store, rank=0, world_size=1)
    return dist.new_group()


def test_owning_shard_freezes_only_its_local_padding_row(monkeypatch):
    group = _single_rank_fake_pg()
    try:
        monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: group)
        # tp group of size 1: offset 0, the shard owns every row, so a global
        # padding_idx of 7 maps to local row 7.
        monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 1)
        monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 0)

        emb = VocabParallelEmbedding(num_embeddings=8, embedding_dim=4, padding_idx=7)
        emb.weight = torch.nn.Parameter(
            torch.randn(8, 4, dtype=torch.float64), requires_grad=True
        )
        input_ids = torch.tensor([[3, 7, 1]])
        out = emb(input_ids)
        out.sum().backward()
        grad = emb.weight.grad
        # The padding row's gradient is suppressed ...
        assert grad[7].abs().sum() == 0
        # ... while the other rows still learn (non-vacuity).
        assert grad[3].abs().sum() > 0
        assert grad[1].abs().sum() > 0
    finally:
        dist.destroy_process_group()


def test_non_owning_shard_passes_no_local_padding_idx(monkeypatch):
    group = _single_rank_fake_pg()
    try:
        monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: group)
        # tp group of size 2, this rank is 1: vocab 8 -> chunk 4, offset 4,
        # local rows [4, 8). Global padding_idx 3 lives on rank 0.
        monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 2)
        monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 1)

        emb = VocabParallelEmbedding(num_embeddings=8, embedding_dim=4, padding_idx=3)
        emb.weight = torch.nn.Parameter(
            torch.randn(4, 4, dtype=torch.float64), requires_grad=True
        )
        # Before #4637 this raised "Padding_idx must be within
        # num_embeddings" (3 is outside a 4-row shard ... only because the
        # *global* index leaked in); here it must run and keep every local
        # row trainable.
        input_ids = torch.tensor([[6, 7]])
        out = emb(input_ids)
        out.sum().backward()
        grad = emb.weight.grad
        assert grad[2].abs().sum() > 0  # local row for global id 6
        assert grad[3].abs().sum() > 0  # local row for global id 7
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("options", [{"scale_grad_by_freq": True}, {"max_norm": 1.0}])
def test_foreign_tokens_do_not_change_local_embedding_state(monkeypatch, options):
    monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: object())
    monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 2)
    monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 1)
    embedding = VocabParallelEmbedding(8, 3, **options)
    weight = torch.full((4, 3), 2.0)
    embedding.weight = torch.nn.Parameter(weight.clone())
    reference = torch.nn.Embedding(4, 3, **options)
    reference.weight = torch.nn.Parameter(weight.clone())

    # Global 0 and 7 used to be clamped to local rows 0 and 3. Only row 3
    # is actually owned here, so row 0 must not be renormalized and row 3's
    # gradient must count just the two real occurrences.
    output = embedding(torch.tensor([[0, 7, 0, 7]]))
    expected = reference(torch.tensor([3, 3]))
    torch.testing.assert_close(output[0, [1, 3]], expected)
    assert torch.count_nonzero(output[0, [0, 2]]) == 0
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(embedding.weight, reference.weight)
    torch.testing.assert_close(embedding.weight.grad, reference.weight.grad)


@pytest.mark.parametrize("shape", [(8, 3), (3, 3), (4, 2)])
def test_wrong_vocab_shard_shape_is_rejected_before_renormalization(monkeypatch, shape):
    monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: object())
    monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 2)
    monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 1)
    embedding = VocabParallelEmbedding(8, 3, max_norm=1.0)
    embedding.weight = torch.nn.Parameter(torch.full(shape, 2.0))
    original = embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="local vocabulary shard"):
        embedding(torch.tensor([[4, 7]]))
    torch.testing.assert_close(embedding.weight, original)


def test_empty_vocab_shard_returns_zero_with_empty_gradient(monkeypatch):
    monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: object())
    monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 4)
    monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 3)
    embedding = VocabParallelEmbedding(2, 3)
    embedding.weight = torch.nn.Parameter(torch.empty(0, 3))
    output = embedding(torch.tensor([[0, 1]]))
    torch.testing.assert_close(output, torch.zeros(1, 2, 3))
    output.sum().backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad.shape == (0, 3)


@pytest.mark.parametrize("invalid_token", [-1, 8, 100])
def test_invalid_global_tokens_fail_before_weight_renormalization(
    monkeypatch, invalid_token
):
    monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: object())
    monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: 2)
    monkeypatch.setattr(embedding_mod.distributed, "get_rank", lambda g: 1)
    embedding = VocabParallelEmbedding(8, 3, max_norm=1.0)
    embedding.weight = torch.nn.Parameter(torch.full((4, 3), 2.0))
    original = embedding.weight.detach().clone()
    with pytest.raises(RuntimeError, match="global vocabulary"):
        embedding(torch.tensor([4, invalid_token]))
    torch.testing.assert_close(embedding.weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("vocab_size,tp_size", [(7, 3), (2, 4)])
def test_summed_vocab_shards_match_full_embedding_and_gradients(
    monkeypatch, vocab_size, tp_size
):
    from llmtuner.components.loss import vocab_shard_bounds

    monkeypatch.setattr(embedding_mod, "spmd_mesh_group", lambda axis: object())
    monkeypatch.setattr(embedding_mod.distributed, "get_world_size", lambda g: tp_size)
    reference = torch.nn.Embedding(vocab_size, 3, padding_idx=vocab_size - 1)
    inputs = torch.tensor([list(range(vocab_size)) + [0, 0]])
    expected = reference(inputs)
    shards = []
    outputs = []
    for rank in range(tp_size):
        monkeypatch.setattr(
            embedding_mod.distributed, "get_rank", lambda g, rank=rank: rank
        )
        start, end = vocab_shard_bounds(vocab_size, tp_size, rank)
        shard = VocabParallelEmbedding(vocab_size, 3, padding_idx=vocab_size - 1)
        shard.weight = torch.nn.Parameter(reference.weight.detach()[start:end].clone())
        shards.append(shard)
        outputs.append(shard(inputs))
    actual = torch.stack(outputs).sum(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.arange(actual.numel(), dtype=actual.dtype).reshape(actual.shape)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        torch.cat([shard.weight.grad for shard in shards]),
        reference.weight.grad,
        rtol=0,
        atol=0,
    )
