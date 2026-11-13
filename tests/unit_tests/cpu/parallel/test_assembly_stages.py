"""The assembly stage list: the order contract, pinned.

``parallelize_model``'s correctness is its call order (TP before EP
before CP, AC before compile, FSDP last), and which of those stages the PP
path runs (a chunk is checkpointed on its own layers, so AC is on both paths
-- upstream hands ``ac_config`` to each model part's own ``parallelize``).
That used to live in a docstring; it is now the ``STAGES`` table, and this
test is what keeps the table -- not the prose -- as the contract.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from torch import nn

import llmtuner.parallel as parallel_api
from llmtuner.config.activation_checkpoint_config import MemoryBudgetACConfig
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel import parallelize
from llmtuner.parallel.stages import (
    STAGE_ORDER,
    STAGES,
)


def test_the_full_order_is_the_documented_contract() -> None:
    assert STAGE_ORDER == (
        "tp",
        "ep",
        "cp",
        "ac",
        "activation_swap",
        "compile",
        "fsdp",
    )


def test_parallel_public_entries_resolve_to_their_defining_modules():
    from importlib import import_module

    entries = {
        "ParallelPlan": "plan",
        "PlanTarget": "plan",
        "apply_cp": "context_parallel.context_parallel",
        "apply_ep": "expert_parallel.expert_parallel",
        "apply_fsdp": "fully_shard_parallel.fully_shard_parallel",
        "apply_tp": "tensor_parallel.tensor_parallel",
        "parallelize_model": "parallelize",
    }
    assert set(parallel_api.__all__) == set(entries)
    for name, source in entries.items():
        implementation = getattr(import_module(f"llmtuner.parallel.{source}"), name)
        assert getattr(parallel_api, name) is implementation
        assert implementation.__module__ == f"llmtuner.parallel.{source}"


@pytest.mark.parametrize("flag", ["activation_swap", "compile"])
@pytest.mark.parametrize("invalid", [1, "false", None])
def test_direct_assembly_rejects_non_boolean_execution_flags(flag, invalid):
    from llmtuner.errors import ConfigError

    model = nn.Linear(2, 2)
    forward = model.forward
    options = {"activation_swap": False, "compile": False}
    options[flag] = invalid
    with pytest.raises(ConfigError, match=rf"{flag} must be bool"):
        parallelize.parallelize_model(
            model, cfg=ParallelConfig(), mesh=None, parallel_dims=None, **options
        )
    assert model.forward == forward
    assert model._compiled_call_impl is None


def test_fsdp_wraps_last() -> None:
    assert STAGE_ORDER[-1] == "fsdp"


@pytest.mark.parametrize(
    "mode,options,match",
    [
        ("unknown-mode", {}, "Unknown activation checkpointing mode"),
        ("selective", {}, "SelectiveACConfig"),
        ("region", {}, "RegionACConfig"),
        ("memory_budget", {}, "MemoryBudgetACConfig"),
        (
            "memory_budget",
            {"memory_budget_ac": MemoryBudgetACConfig()},
            "requires compile",
        ),
    ],
)
@pytest.mark.parametrize("pipeline", [False, True])
def test_invalid_ac_options_fail_before_parallel_transforms(
    monkeypatch, mode, options, match, pipeline
):
    from types import SimpleNamespace

    def unexpected_transform(*args, **kwargs):
        pytest.fail("invalid AC options reached a model transformation")

    monkeypatch.setattr(parallelize, "apply_tp", unexpected_transform)
    monkeypatch.setattr(parallelize, "apply_pp", unexpected_transform)
    with pytest.raises(ValueError, match=match):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=(SimpleNamespace(pp_enabled=True) if pipeline else None),
            activation_checkpoint=mode,
            **options,
        )


@pytest.mark.parametrize("pipeline", [False, True])
def test_unavailable_fsdp_engine_fails_before_model_transforms(monkeypatch, pipeline):
    from types import SimpleNamespace

    from llmtuner.errors import EnvironmentUnsupportedError

    def unavailable(*args):
        raise EnvironmentUnsupportedError("custom engine unavailable")

    def unexpected_transform(*args, **kwargs):
        pytest.fail("unavailable FSDP engine reached a model transformation")

    monkeypatch.setattr(parallelize, "validate_fsdp_environment", unavailable)
    monkeypatch.setattr(parallelize, "apply_tp", unexpected_transform)
    monkeypatch.setattr(parallelize, "apply_pp", unexpected_transform)
    with pytest.raises(EnvironmentUnsupportedError, match="custom engine unavailable"):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=SimpleNamespace(pp_enabled=pipeline),
        )


@pytest.mark.parametrize("pipeline", [False, True])
def test_invalid_compile_backend_fails_before_parallel_transforms(
    monkeypatch, pipeline
):
    from types import SimpleNamespace

    from torch._dynamo.exc import InvalidBackend

    def unexpected_transform(*args, **kwargs):
        pytest.fail("invalid compile backend reached a model transformation")

    monkeypatch.setattr(parallelize, "apply_tp", unexpected_transform)
    monkeypatch.setattr(parallelize, "apply_pp", unexpected_transform)
    with pytest.raises(InvalidBackend):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=SimpleNamespace(pp_enabled=pipeline),
            compile=True,
            compile_config=CompileConfig(backend="llmtuner_nonexistent_backend"),
        )


def test_unsupported_mc2_degree_fails_in_ep_stage_before_conversion(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.config.parallel_config import ExpertParallelConfig
    from llmtuner.errors import EnvironmentUnsupportedError
    from llmtuner.parallel.expert_parallel import expert_parallel

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("unsupported MC2 degree reached MoE conversion")

    monkeypatch.setattr(expert_parallel, "swap_moe_blocks", unexpected_conversion)
    with pytest.raises(EnvironmentUnsupportedError, match="world size in"):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(
                expert_parallel_size=2,
                ep_config=ExpertParallelConfig(ep_token_dispatcher="mc2"),
            ),
            mesh=None,
            parallel_dims=SimpleNamespace(
                pp_enabled=False,
                get_optional_mesh=lambda name: (
                    SimpleNamespace(get_group=lambda: object())
                    if name == "ep"
                    else None
                ),
                get_mesh=lambda name: SimpleNamespace(get_group=lambda: object()),
            ),
        )


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
        parallelize.parallelize_model(
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

    result = parallelize.parallelize_model(
        model,
        cfg=ParallelConfig(),
        mesh=None,
        parallel_dims=None,
        compile=compile_enabled,
        compile_config=CompileConfig(backend="aot_eager"),
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
        name
        for name in STAGE_ORDER
        if name != "activation_swap" and (name != "compile" or compile_enabled)
    ]
    seen = []
    dense_mesh = object()
    cfg = ParallelConfig()
    dims = SimpleNamespace(
        pp_enabled=True,
        dp_replicate=1,
        dp_shard=1,
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
        function = (
            "apply_activation_swap" if name == "activation_swap" else f"apply_{name}"
        )
        monkeypatch.setattr(parallelize, function, runner(name))

    schedule = object()

    def build_schedule(bound_stages, **kwargs):
        assert bound_stages is stages
        for stage, part in zip(stages, parts, strict=True):
            assert stage.submod is part
            assert part.history == expected_order
        return schedule

    monkeypatch.setattr(parallelize, "build_pipeline_schedule", build_schedule)
    result = parallelize.parallelize_model(
        model,
        cfg=cfg,
        mesh=None,
        parallel_dims=dims,
        compile=compile_enabled,
        compile_config=CompileConfig(backend="aot_eager"),
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
        dp_replicate=1,
        dp_shard=1,
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
    parallelize.parallelize_model(
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
    cfg = ParallelConfig(expert_parallel_size=8 if dispatcher == "mc2" else 2)
    cfg.ep_config.ep_token_dispatcher = dispatcher
    # This test isolates mesh wiring; model geometry is covered separately.
    monkeypatch.setattr(parallelize, "resolve_ep_plan", lambda *args, **kwargs: None)
    observed = {}
    for name in STAGE_ORDER:
        function = (
            "apply_activation_swap" if name == "activation_swap" else f"apply_{name}"
        )
        monkeypatch.setattr(parallelize, function, lambda part, **kwargs: part)

    def apply_ep(part, **kwargs):
        observed.update(kwargs)
        return part

    monkeypatch.setattr(parallelize, "apply_ep", apply_ep)
    parallelize.parallelize_model(model, cfg=cfg, mesh=None, parallel_dims=dims)
    assert observed["ep_group"] is ep_group
    assert observed["error_group"] is loss_group


def test_full_ac_skips_activation_swap(monkeypatch):
    calls = []
    monkeypatch.setattr(parallelize, "apply_ac", lambda part, **kwargs: part)
    monkeypatch.setattr(
        parallelize,
        "apply_activation_swap",
        lambda part: calls.append(part) or part,
    )
    parallelize.parallelize_model(
        nn.Linear(2, 2),
        cfg=ParallelConfig(),
        mesh=None,
        parallel_dims=None,
        activation_checkpoint="full",
        activation_swap=True,
    )
    assert calls == []


@pytest.mark.parametrize(
    ("compile_enabled", "ac_mode", "message"),
    [
        (True, "none", "requires compile=False"),
        (False, "selective", "incompatible with activation_checkpoint='selective'"),
        (False, "region", "incompatible with activation_checkpoint='region'"),
    ],
)
def test_activation_swap_rejects_unsupported_direct_api_combinations(
    compile_enabled, ac_mode, message
):
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match=message):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=None,
            compile=compile_enabled,
            activation_checkpoint=ac_mode,
            activation_swap=True,
        )


def test_cp_plan_failure_precedes_cp_execution_and_later_stages(monkeypatch):
    def reject_plan(*args, **kwargs):
        raise ValueError("invalid CP plan")

    def unexpected(*args, **kwargs):
        pytest.fail("CP execution or a later stage began before plan validation")

    monkeypatch.setattr(parallelize, "resolve_cp_plan", reject_plan, raising=False)
    monkeypatch.setattr(parallelize, "apply_tp", lambda model, **kwargs: model)
    monkeypatch.setattr(parallelize, "apply_ep", lambda model, **kwargs: model)
    monkeypatch.setattr(parallelize, "apply_cp", unexpected)
    monkeypatch.setattr(parallelize, "apply_ac", unexpected)
    monkeypatch.setattr(parallelize, "apply_fsdp", unexpected)
    with pytest.raises(ValueError, match="invalid CP plan"):
        parallelize.parallelize_model(
            nn.Linear(2, 2),
            cfg=ParallelConfig(context_parallel_kv_allgather_size=2),
            mesh=None,
            parallel_dims=None,
        )


@pytest.mark.parametrize("has_topology", [False, True])
def test_cp_preflight_uses_explicit_group_without_wrapping_execution(
    monkeypatch, has_topology
):
    active = {"boundary": False}
    calls = []
    expected_group = object()
    plan = object()
    model = nn.Linear(2, 2)

    def get_group():
        assert not active["boundary"]
        calls.append("group")
        return expected_group

    @contextmanager
    def boundary(phase, *, group):
        assert has_topology and phase == "CP plan validation"
        assert group is expected_group
        calls.append("enter")
        active["boundary"] = True
        try:
            yield
        finally:
            active["boundary"] = False
            calls.append("exit")

    def resolve(part, *, cfg):
        assert part is model and active["boundary"] is has_topology
        calls.append("resolve")
        return plan

    def execute(part, **kwargs):
        assert part is model and not active["boundary"]
        assert kwargs["resolved_plan"] is plan
        calls.append("execute")
        return part

    dims = SimpleNamespace(
        pp_enabled=False,
        get_optional_mesh=lambda name: None,
        build_device_mesh=lambda: SimpleNamespace(get_group=get_group),
    )
    monkeypatch.setattr(parallelize, "synchronize_local_errors", boundary)
    monkeypatch.setattr(parallelize, "resolve_cp_plan", resolve)
    monkeypatch.setattr(parallelize, "apply_cp", execute)
    for name in ("apply_tp", "apply_ep", "apply_ac", "apply_fsdp"):
        monkeypatch.setattr(parallelize, name, lambda part, **kwargs: part)
    assert (
        parallelize.parallelize_model(
            model,
            cfg=ParallelConfig(context_parallel_kv_allgather_size=2),
            mesh=None,
            parallel_dims=dims if has_topology else None,
        )
        is model
    )
    assert calls == (
        ["group", "enter", "resolve", "exit", "execute"]
        if has_topology
        else ["resolve", "execute"]
    )
