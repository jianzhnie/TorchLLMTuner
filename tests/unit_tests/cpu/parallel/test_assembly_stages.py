"""The assembly stage list: the order contract, pinned.

``parallelize_hf_transformers``'s correctness is its call order (TP before EP
before CP, AC before compile, FSDP last), and which of those stages the PP
path runs (a chunk is checkpointed on its own layers, so AC is on both paths
-- upstream hands ``ac_config`` to each model part's own ``parallelize``).
That used to live in a docstring; it is now the ``STAGES`` table, and this
test is what keeps the table -- not the prose -- as the contract.
"""

import pytest
from torch import nn

import llmtuner.parallel as parallel_api
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel import parallelize
from llmtuner.parallel.stages import (
    STAGE_ORDER,
    STAGES,
)


def test_the_full_order_is_the_documented_contract() -> None:
    assert STAGE_ORDER == ("tp", "ep", "cp", "ac", "compile", "fsdp")


def test_parallel_public_entries_resolve_to_their_defining_modules():
    from importlib import import_module

    entries = {
        "apply_cp": "context_parallel.context_parallel",
        "apply_ep": "expert_parallel.expert_parallel",
        "apply_fsdp": "fully_shard_parallel.fully_shard_parallel",
        "apply_tp": "tensor_parallel.tensor_parallel",
        "parallelize_hf_transformers": "parallelize",
    }
    assert set(parallel_api.__all__) == set(entries)
    for name, source in entries.items():
        implementation = getattr(import_module(f"llmtuner.parallel.{source}"), name)
        assert getattr(parallel_api, name) is implementation
        assert implementation.__module__ == f"llmtuner.parallel.{source}"


def test_fsdp_wraps_last() -> None:
    assert STAGE_ORDER[-1] == "fsdp"


def test_sharding_wrappers_come_before_ac_and_compile() -> None:
    tp, ep, cp, ac, compile_ = (
        STAGE_ORDER.index(n) for n in ("tp", "ep", "cp", "ac", "compile")
    )
    assert tp < ep < cp < ac < compile_


def test_stage_table_matches_the_execution_order() -> None:
    assert STAGE_ORDER == tuple(s.name for s in STAGES)
    assert {s.name for s in STAGES} == set(STAGE_ORDER)


def test_every_stage_documents_why_it_sits_where_it_does() -> None:
    for stage in STAGES:
        assert stage.why_here


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_assembly_executes_stages_in_order(monkeypatch, compile_enabled):
    calls = []
    model = nn.Linear(2, 2)

    def runner(name):
        def apply(part, **kwargs):
            assert part is model
            calls.append(name)
            return part

        return apply

    for name, function in (
        ("tp", "apply_tp"),
        ("ep", "apply_ep"),
        ("cp", "apply_cp"),
        ("ac", "apply_ac"),
        ("compile", "apply_compile"),
        ("fsdp", "apply_fsdp"),
    ):
        monkeypatch.setattr(parallelize, function, runner(name))

    result = parallelize.parallelize_hf_transformers(
        model,
        cfg=ParallelConfig(),
        mesh=None,
        parallel_dims=None,
        compile=compile_enabled,
    )
    assert result is model
    assert calls == (
        ["tp", "ep", "cp", "ac", "compile", "fsdp"]
        if compile_enabled
        else ["tp", "ep", "cp", "ac", "fsdp"]
    )
