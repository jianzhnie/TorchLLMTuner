"""Vocabulary-parallel declaration, ownership and local fallback checks."""

import pytest
import torch
import torch.nn as nn

from llmtuner.config import ParallelConfig
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.tensor_parallel.tensor_parallel import tp_sharded_param_ids
from llmtuner.parallel.vocab_parallel import VocabParallelEmbedding


def test_vocab_parallel_local_forward_and_backward() -> None:
    source = nn.Embedding(8, 4, padding_idx=3)
    weight = source.weight.detach().clone()
    start, end = 4, 8
    embedding = VocabParallelEmbedding(
        weight[start:end].clone(),
        source=source,
        vocab_start_index=start,
        vocab_end_index=end,
        tp_group=None,
    )
    ids = torch.tensor([[0, 3, 4, 7]])
    expected = source(ids)
    expected[:, :2] = 0
    actual = embedding(ids)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 1].abs().sum() == 0

    actual.sum().backward()
    source(ids).sum().backward()
    torch.testing.assert_close(embedding.weight.grad, source.weight.grad[start:end])


def test_apply_tp_realizes_embedding_plan(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        "llmtuner.parallel.tensor_parallel.tensor_parallel.logger.info",
        lambda template, *args: messages.append(template % args),
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

    class Mesh:
        def __init__(self) -> None:
            self.device_type = "cpu"

        def __getitem__(self, axis: str):
            assert axis == "tp"
            return self

        def size(self) -> int:
            return 2

        def get_group(self):
            return None

        def get_local_rank(self) -> int:
            return 1

    model = Model()
    original = model.embedding.weight.detach().clone()
    apply_tp(
        model,
        Mesh(),
        ParallelConfig(tensor_parallel_size=2),
        plan={"embedding": "embedding_rowwise"},
    )
    assert isinstance(model.embedding, VocabParallelEmbedding)
    torch.testing.assert_close(model.embedding.weight, original[4:])
    assert (model.embedding.vocab_start_index, model.embedding.vocab_end_index) == (
        4,
        8,
    )
    assert model._tp_checkpoint_layouts["embedding.weight"] == (0, False, None)
    assert tp_sharded_param_ids([model]) == {id(model.embedding.weight)}
    assert messages == [
        "Applied TP degree 2: dense=0, MoE=0, vocab=1, collective=functional"
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_norm": 1.0}, "max_norm"),
        ({"scale_grad_by_freq": True}, "scale_grad_by_freq"),
        ({"sparse": True}, "sparse"),
    ],
)
def test_apply_tp_rejects_unsupported_embedding_before_mutation(
    options, message
) -> None:
    model = nn.Sequential(nn.Embedding(8, 4, **options))
    original = model[0]
    with pytest.raises(ValueError, match=message):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={"0": "embedding_rowwise"},
        )
    assert model[0] is original


def test_apply_tp_rejects_tied_embedding_and_head() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)
            self.head.weight = self.embedding.weight

    model = Model()
    with pytest.raises(ValueError, match="cannot preserve tied weights"):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={"embedding": "embedding_rowwise"},
        )
    assert type(model.embedding) is nn.Embedding
    assert model.head.weight is model.embedding.weight


def test_vocab_parallel_requires_equal_shards() -> None:
    model = nn.Sequential(nn.Embedding(7, 4))
    with pytest.raises(ValueError, match="not divisible"):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={"0": "embedding_rowwise"},
        )


def test_independent_vp_plan_defaults_and_validation() -> None:
    from llmtuner.parallel.vocab_parallel import (
        VocabParallelPlan,
        alltoallv,
        alltoallv_with_gradient_divide_factor,
    )

    plan = VocabParallelPlan(apply_modules=["embedding"])
    assert plan.apply_vfsdp_modules == ["embedding"]
    assert plan.gradient_divide_factor == 1.0
    with pytest.raises(ValueError, match="must not be empty"):
        VocabParallelPlan(apply_modules=[])
    with pytest.raises(ValueError, match="must be positive"):
        VocabParallelPlan(apply_modules=["embedding"], gradient_divide_factor=0)

    inputs = torch.arange(6, dtype=torch.float32, requires_grad=True)
    assert alltoallv(inputs, None, [2, 4], [3, 3]) is inputs
    output = alltoallv_with_gradient_divide_factor(
        inputs, None, [2, 4], [3, 3], gradient_divide_factor=2.0
    )
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 0.5))
