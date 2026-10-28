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


def test_pipeline_validates_tp_override_before_splitting(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.config.parallel_config import TensorParallelConfig

    cfg = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        tp_config=TensorParallelConfig(tp_colwise_modules=["missing.*"]),
    )
    topology = SimpleNamespace(pp_enabled=True, get_optional_mesh=lambda name: None)

    def unexpected_split(*args, **kwargs):
        pytest.fail("invalid global pattern reached pipeline splitting")

    monkeypatch.setattr(parallelize, "apply_pp", unexpected_split)
    with pytest.raises(ValueError, match="TP override pattern.*matched no module"):
        parallelize.parallelize_hf_transformers(
            nn.Sequential(nn.Linear(4, 4)),
            cfg=cfg,
            mesh=None,
            parallel_dims=topology,
            global_batch_size=2,
        )


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


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_pipeline_assembly_rebinds_transformed_parts_before_schedule(
    monkeypatch, compile_enabled
):
    from types import SimpleNamespace

    model = nn.Linear(2, 2)
    parts = [nn.Linear(2, 2), nn.Linear(2, 2)]
    stages = [SimpleNamespace(submod=part) for part in parts]
    expected_order = [
        name for name in STAGE_ORDER if name != "compile" or compile_enabled
    ]
    seen = []
    dense_mesh = object()
    cfg = ParallelConfig()
    dims = SimpleNamespace(
        pp_enabled=True,
        get_optional_mesh=lambda name: None,
        spmd_dense_mesh=lambda: dense_mesh,
    )
    monkeypatch.setattr(
        parallelize, "apply_pp", lambda *args, **kwargs: (stages, parts, True, True)
    )

    def runner(name):
        def apply(part, **kwargs):
            history = getattr(part, "history", [])
            assert name == expected_order[len(history)]
            if name in {"tp", "cp"}:
                assert kwargs["mesh"] is dense_mesh
            if name == "fsdp":
                assert kwargs["cfg"] is cfg
            replacement = nn.Identity()
            replacement.history = [*history, name]
            seen.append(name)
            return replacement

        return apply

    for name in STAGE_ORDER:
        monkeypatch.setattr(parallelize, f"apply_{name}", runner(name))

    schedule = object()

    def build_schedule(bound_stages, **kwargs):
        assert bound_stages is stages
        for stage, part in zip(stages, parts, strict=True):
            assert stage.submod is part
            assert part.history == expected_order
        return schedule

    monkeypatch.setattr(parallelize, "build_pipeline_schedule", build_schedule)
    result = parallelize.parallelize_hf_transformers(
        model,
        cfg=cfg,
        mesh=None,
        parallel_dims=dims,
        compile=compile_enabled,
        global_batch_size=4,
    )
    assert result.schedule is schedule
    assert result.model_parts is parts
    assert seen == expected_order * 2


@pytest.mark.parametrize("pipeline", [False, True])
def test_assembly_rebinds_fsdp_exclusions_after_tp(monkeypatch, pipeline):
    import copy
    from types import SimpleNamespace

    from llmtuner.config.parallel_config import FSDPConfig

    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    originals = list(model.parameters())
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                str(index): {"ignored_params": {parameter}}
                for index, parameter in enumerate(originals)
            }
        )
    )
    parts = [copy.deepcopy(model), copy.deepcopy(model)]
    del parts[0][1]
    parts[1]._modules.pop("0")
    stages = [SimpleNamespace(submod=part) for part in parts]
    dims = SimpleNamespace(
        pp_enabled=True,
        get_optional_mesh=lambda name: None,
        spmd_dense_mesh=lambda: None,
    )
    monkeypatch.setattr(
        parallelize, "apply_pp", lambda *args, **kwargs: (stages, parts, True, True)
    )
    monkeypatch.setattr(parallelize, "build_pipeline_schedule", lambda *a, **kw: None)
    for name in ("ep", "cp", "ac"):
        monkeypatch.setattr(parallelize, f"apply_{name}", lambda part, **kw: part)

    def replace_parameters(part, **kwargs):
        for child in part.children():
            child.weight = nn.Parameter(child.weight.detach().clone())
        return part

    calls = []

    def check_fsdp(part, *, cfg, **kwargs):
        for name, options in cfg.fsdp_config.fsdp_apply_modules.items():
            assert options["ignored_params"] == {part.get_submodule(name).weight}
        calls.append(part)
        return part

    monkeypatch.setattr(parallelize, "apply_tp", replace_parameters)
    monkeypatch.setattr(parallelize, "apply_fsdp", check_fsdp)
    parallelize.parallelize_hf_transformers(
        model,
        cfg=cfg,
        mesh=None,
        parallel_dims=dims if pipeline else None,
        global_batch_size=4,
    )
    assert len(calls) == (2 if pipeline else 1)
    for index, parameter in enumerate(originals):
        assert cfg.fsdp_config.fsdp_apply_modules[str(index)]["ignored_params"] == {
            parameter
        }


@pytest.mark.parametrize("dispatcher", ["domino", "mc2"])
def test_fused_ep_dispatch_receives_stage_wide_error_group(monkeypatch, dispatcher):
    from types import SimpleNamespace

    model = nn.Linear(2, 2)
    ep_group, loss_group = object(), object()
    ep_mesh = SimpleNamespace(get_group=lambda: ep_group)
    loss_mesh = SimpleNamespace(get_group=lambda: loss_group)
    dims = SimpleNamespace(
        pp_enabled=False,
        get_optional_mesh=lambda name: ep_mesh if name == "ep" else None,
        get_mesh=lambda name: loss_mesh,
    )
    cfg = ParallelConfig(expert_parallel_size=2)
    cfg.ep_config.ep_token_dispatcher = dispatcher
    observed = {}
    for name in STAGE_ORDER:
        monkeypatch.setattr(parallelize, f"apply_{name}", lambda part, **kwargs: part)

    def apply_ep(part, **kwargs):
        observed.update(kwargs)
        return part

    monkeypatch.setattr(parallelize, "apply_ep", apply_ep)
    parallelize.parallelize_hf_transformers(
        model, cfg=cfg, mesh=None, parallel_dims=dims
    )
    assert observed["ep_group"] is ep_group
    assert observed["error_group"] is loss_group
