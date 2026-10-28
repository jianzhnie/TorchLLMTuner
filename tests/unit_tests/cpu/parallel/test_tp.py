"""Tests for the tensor-parallel declaration layer and engine wiring.

The fused collectives (``AllGatherLinear`` / ``LinearReduceScatter``) run only
on CUDA symmetric memory; the functional-collective fallback that lets TP run
on CPU needs a real process group, so the end-to-end forward/backward
equivalence lives in ``tests/integration_tests/tp_equivalence.py`` (torchrun).
What is testable single-process here is everything around the collectives: the
declaration, the weight layout each kind produces, plan resolution, the
loud-raise guards, and that ``apply_tp`` leaves a model alone when TP is off.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
import torch.nn as nn

from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import TensorParallelConfig
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
    ShardingConfig,
    colwise,
    match_tp_sharding_spec,
    resolve_tp_sharding_plan,
    rowwise,
)


def test_declaration_factories_pick_the_right_realizer() -> None:
    assert colwise() == ShardingConfig(
        kind="colwise", implementation=ColumnParallelLinear
    )
    assert rowwise() == ShardingConfig(kind="rowwise", implementation=RowParallelLinear)


@pytest.mark.parametrize("failure", ["bias", "dimension"])
def test_invalid_later_dense_target_is_rejected_before_mutation(failure):
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 3 if failure == "dimension" else 4, bias=failure == "bias"),
    )
    original_modules = list(model)
    original_parameters = list(model.parameters())
    with pytest.raises(ValueError, match="bias|not divisible"):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={"0": "colwise", "1": "colwise"},
        )
    assert list(model) == original_modules
    assert all(
        current is original
        for current, original in zip(
            model.parameters(), original_parameters, strict=True
        )
    )


def test_colwise_layout_is_cut_on_output_features() -> None:
    # nn.Linear weight is [out, in]; colwise keeps that layout and cuts dim 0,
    # storing [out/tp, in] -- the w_shard_n = [N/R, K] contract AllGatherLinear
    # documents (the op self-transposes inside the GEMM). An earlier version
    # transposed first and cut the last dim, feeding the op [K, N/R] and
    # producing garbage on the fused path.
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)  # out=4, in=8
    mod = ColumnParallelLinear(W, tp_size=2, tp_rank=0, group=None)
    assert mod.weight.shape == (2, 8)
    assert torch.equal(mod.weight, W[0:2])
    # rank 1 holds the other half
    other = ColumnParallelLinear(W, tp_size=2, tp_rank=1, group=None)
    assert torch.equal(other.weight, W[2:4])


def test_rowwise_layout_is_cut_on_input_features() -> None:
    W = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    mod = RowParallelLinear(W, tp_size=2, tp_rank=1, group=None)
    assert mod.weight.shape == (4, 4)
    assert torch.equal(mod.weight, W[:, 4:8])


def test_sharded_weights_reconstruct_the_original() -> None:
    W = torch.randn(6, 12)
    for cls in (ColumnParallelLinear, ColwiseLinearNoGather, RowParallelLinear):
        shards = [cls(W, tp_size=3, tp_rank=r, group=None).weight for r in range(3)]
        if cls is RowParallelLinear:
            rec = torch.cat(shards, dim=1)
        else:
            rec = torch.cat(shards, dim=0)
        assert torch.equal(rec, W)


def test_plan_resolution_from_hf_string_map() -> None:
    class M(nn.Module):
        _tp_config = {
            "layers.*.q_proj": "colwise",
            "layers.*.o_proj": "rowwise",
        }

    plan = resolve_tp_sharding_plan(M(), plan=None)
    assert {k: v.kind for k, v in plan.items()} == {
        "layers.*.q_proj": "colwise",
        "layers.*.o_proj": "rowwise",
    }
    assert match_tp_sharding_spec("layers.3.q_proj", plan=plan).kind == "colwise"
    assert match_tp_sharding_spec("layers.0.o_proj", plan=plan).kind == "rowwise"
    assert match_tp_sharding_spec("layers.0.up_proj", plan=plan) is None


@pytest.mark.parametrize("overrides", [[], ["projection"]])
def test_tp_target_resolution_reads_the_model_plan_once(overrides) -> None:
    from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_targets

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4, bias=False)
            self.plan_reads = 0

        @property
        def tp_config(self):
            self.plan_reads += 1
            if self.plan_reads > 1:
                pytest.fail("target resolution must reuse the original plan")
            return {"projection": "colwise"}

    model = Model()
    targets, moe_blocks = resolve_tp_targets(
        model,
        cfg=ParallelConfig(
            tensor_parallel_size=2,
            tp_config=TensorParallelConfig(tp_rowwise_modules=overrides),
        ),
        plan=None,
    )
    assert model.plan_reads == 1
    assert targets[0][:2] == ("projection", model.projection)
    assert targets[0][2].kind == ("rowwise" if overrides else "colwise")
    assert moe_blocks == []


@pytest.mark.parametrize("field", ["tp_colwise_modules", "tp_rowwise_modules"])
def test_configured_tp_pattern_typo_is_not_hidden_by_valid_model_plan(field):
    from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_targets

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    original = model[0]
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        tp_config=TensorParallelConfig(**{field: ["missing.*"]}),
    )
    with pytest.raises(ValueError, match="TP override pattern.*matched no module"):
        resolve_tp_targets(model, cfg=cfg, plan={"0": "colwise"})
    assert model[0] is original


def test_pipeline_part_permits_override_owned_by_another_stage():
    from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_targets

    part = nn.Sequential(nn.Linear(4, 4, bias=False))
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        tp_config=TensorParallelConfig(tp_rowwise_modules=["other_stage.projection"]),
    )
    targets, _ = resolve_tp_targets(part, cfg=cfg, plan={"0": "colwise"})
    assert targets[0][0] == "0"
    assert targets[0][2].kind == "colwise"


@pytest.mark.parametrize("cls", [ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("shape", [(), (4,), (1, 2, 3, 4)])
def test_tp_projection_rejects_unsupported_rank_before_communication(
    monkeypatch, cls, shape
):
    import llmtuner.parallel.tensor_parallel.tensor_parallel as tp_module

    def unexpected(*args):
        pytest.fail("invalid layout reached communication")

    monkeypatch.setattr(tp_module, "all_gather_linear", unexpected)
    monkeypatch.setattr(tp_module, "linear_reduce_scatter", unexpected)
    projection = cls(
        torch.ones(6, 4), tp_size=1, tp_rank=0, group=None, use_symm_mem=False
    )
    with pytest.raises(ValueError, match="TP projection input must be"):
        projection(torch.empty(shape))


@pytest.mark.parametrize("cls", [ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("shape", [(0, 4), (1, 0, 4)])
def test_tp_projection_preserves_empty_token_shape_and_gradients(
    monkeypatch, cls, shape
):
    import llmtuner.parallel.tensor_parallel.tensor_parallel as tp_module

    def local_linear(inputs, weight, group):
        return torch.nn.functional.linear(inputs, weight)

    monkeypatch.setattr(tp_module, "all_gather_linear", local_linear)
    monkeypatch.setattr(tp_module, "linear_reduce_scatter", local_linear)
    projection = cls(
        torch.ones(6, 4), tp_size=1, tp_rank=0, group=None, use_symm_mem=False
    )
    inputs = torch.empty(shape, requires_grad=True)
    output = projection(inputs)
    assert output.shape == (*shape[:-1], 6)
    output.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape
    torch.testing.assert_close(
        projection.weight.grad, torch.zeros_like(projection.weight)
    )


def test_target_resolution_applies_configured_pattern_overrides():
    from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_targets

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    cfg = ParallelConfig(
        tensor_parallel_size=2, tp_config=TensorParallelConfig(tp_rowwise_modules=["0"])
    )
    targets, moe_blocks = resolve_tp_targets(model, cfg=cfg, plan={"0": "colwise"})
    assert len(targets) == 1
    assert targets[0][0] == "0"
    assert targets[0][1] is model[0]
    assert targets[0][2].kind == "rowwise"
    assert not moe_blocks


def test_plan_resolution_prefers_the_tp_config_property_over_the_attribute() -> None:
    """A wrapper that re-parents the model must win over the raw attribute.

    ``HFTransformerModel`` holds the HF model under ``self.model``, so its module
    paths carry a ``model.`` prefix the raw HF plan does not. Reading ``_tp_config``
    off such a wrapper yields patterns that match nothing.
    """

    class Wrapper(nn.Module):
        _tp_config = {"layers.*.q_proj": "colwise"}

        @property
        def tp_config(self) -> dict[str, str]:
            return {"model.layers.*.q_proj": "colwise"}

    plan = resolve_tp_sharding_plan(Wrapper(), plan=None)
    assert set(plan) == {"model.layers.*.q_proj"}


def test_a_wrapper_tp_config_matches_the_modules_it_exposes() -> None:
    """The regression this locks in: TP matched 0 of 15 projections.

    Two things had to line up and neither did. The plan lives on the inner HF
    model, not the wrapper, so ``resolve_tp_sharding_plan`` found nothing; and even once
    found, HF's patterns are spelled relative to the HF model while the
    wrapper's ``named_modules`` paths sit under ``model.``. ``apply_tp`` then
    matched nothing and left the model replicated -- a TP run that silently
    trains a non-sharded model and looks like a success.

    Checked on the real wrapper so the path spelling is the real one.
    """
    from llmtuner.models.hf.factory import build_model_config
    from llmtuner.models.hf.model import HFTransformerModel

    config = build_model_config(
        "llama",
        seq_len=32,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config)
    plan = resolve_tp_sharding_plan(model, plan=None)

    assert plan, "the wrapper exposed no usable TP plan"

    matched = [
        path
        for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and match_tp_sharding_spec(path, plan=plan) is not None
    ]
    # 7 projections per layer x 2 layers. ``lm_head`` is deliberately absent:
    # HF's plan does not shard it, which is why the vocab-parallel loss path
    # stays unreachable until a Shard(0) head plan lands.
    assert len(matched) == 14
    assert not any(path.endswith("lm_head") for path in matched)


def test_qwen3_plan_resolves_rather_than_raising_on_its_qk_norms() -> None:
    """The plan an llmtuner-supported family actually ships must resolve.

    Qwen3 -- the architecture the repo's own example trains -- once marked
    ``q_norm`` / ``k_norm`` with HF's ``replicated_with_grad_allreduce``, a spec
    ``resolve_tp_sharding_plan`` had no branch for; every Qwen3 TP run died
    in ``apply_tp``
    before touching a weight, and no test caught it because the only plan under
    test was a hand-written ``{colwise, rowwise}`` map.

    Transformers 4.57 no longer ships that spec for Qwen3 (its
    ``base_model_tp_config`` is projections only), so the branch is pinned by
    injecting the entry into the real plan rather than by asserting on whatever
    the installed version happens to declare. The branch's meaning is what the
    test is for: a norm entry resolves to ``None`` (left whole on every rank,
    its gradient summed by ``Trainer._allreduce_replicated_tp_grads``), while
    the projections still resolve to real realizers.
    """
    from llmtuner.models.hf.factory import build_model_config
    from llmtuner.models.hf.model import HFTransformerModel

    config = build_model_config(
        "qwen3",
        seq_len=32,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    model = HFTransformerModel(config)

    # The real plan must resolve, whatever the installed transformers declares.
    plan = resolve_tp_sharding_plan(model, plan=None)

    # And the branch that used to raise, on the same model's real plan.
    injected = dict(model.tp_config)
    injected["model.layers.*.self_attn.q_norm"] = "replicated_with_grad_allreduce"
    injected["model.layers.*.self_attn.k_norm"] = "replicated_with_grad_allreduce"
    resolved = resolve_tp_sharding_plan(model, plan=injected)
    norms = [p for p in resolved if p.endswith(("q_norm", "k_norm"))]
    assert len(norms) >= 2, "the injected norm entries are not in the plan"
    assert all(resolved[p] is None for p in norms), "a norm must not be sharded"

    matched = [
        path
        for path, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and match_tp_sharding_spec(path, plan=plan) is not None
    ]
    # 7 projections per layer x 2; the norms are neither nn.Linear nor sharded.
    assert len(matched) == 14
    # And the norms really are parameters of this model, so leaving them
    # unsharded is a decision about a module that exists.
    norm_params = [
        n
        for n, _ in model.named_parameters()
        if n.endswith(("q_norm.weight", "k_norm.weight"))
    ]
    assert norm_params


def test_apply_tp_is_a_noop_when_tp_is_one() -> None:
    model = nn.Linear(4, 4)
    cfg = ParallelConfig()  # tp defaults to 1
    assert apply_tp(model, mesh=None, cfg=cfg) is model
    assert isinstance(model, nn.Linear)  # not swapped


def test_apply_tp_raises_when_the_model_has_no_plan() -> None:
    """TP with no plan must refuse, not silently run replicated.

    The raise fires before the mesh is touched (plan validation comes first),
    so a sentinel stands in for the mesh here -- reaching for it would prove
    the guard did not fire.
    """
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="no TP plan"):
        apply_tp(nn.Linear(4, 4), mesh=object(), cfg=cfg)


def test_apply_tp_raises_when_the_plan_matches_no_module() -> None:
    """A plan that matches nothing is the same silent-replication failure."""

    class Model(nn.Module):
        _tp_config = {"*.q_proj": "colwise"}

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        apply_tp(Model(), mesh=object(), cfg=cfg)


# -- head-count divisibility ---------------------------------------------------
#
# The check itself is exercised in ``test_attention_heads.py`` (ungated); these two
# pin that ``apply_tp`` reaches it, and that it fires before the mesh so a bad
# head count is a startup error rather than a shape error inside attention.


class _HeadStub(nn.Module):
    """Just enough of HFTransformerModel for apply_tp's validation path."""

    def __init__(self, num_attention_heads: int, num_key_value_heads: int) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.model = SimpleNamespace(
            config=SimpleNamespace(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        )


def test_apply_tp_refuses_heads_that_do_not_divide_tp() -> None:
    """8 KV heads at head_dim=128 is 1024 features, which tp=16 divides -- so
    nothing downstream rejects it, and HF's head reshape would be the first
    thing to notice. Upstream refuses this at config parse; llmtuner refuses it
    here, before the mesh is touched (the sentinel proves the ordering)."""
    model = _HeadStub(num_attention_heads=8, num_key_value_heads=8)
    model.projection = nn.Linear(1024, 1024, bias=False)
    cfg = ParallelConfig(tensor_parallel_size=16)
    with pytest.raises(ValueError, match=r"num_attention_heads \(8\).*tp \(16\)"):
        apply_tp(model, mesh=object(), cfg=cfg, plan={"projection": "colwise"})


def test_a_divisible_head_count_gets_past_the_guard() -> None:
    """Not a blanket refusal: 8 heads over tp=8 reaches the next gate, which is
    the plan check -- the stub declares no plan, so that is what raises."""
    model = _HeadStub(num_attention_heads=8, num_key_value_heads=8)
    cfg = ParallelConfig(tensor_parallel_size=8)
    with pytest.raises(ValueError, match="no TP plan"):
        apply_tp(model, mesh=object(), cfg=cfg)


# -- HF plan styles this engine does not realize -------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "colwise_rep",
        "rowwise_rep",
        "local_colwise",
        "local_rowwise",
        "gather",
        "replicate",
    ],
)
def test_replicated_activation_styles_are_refused_not_reinterpreted(spec: str) -> None:
    """``colwise_rep`` means "shard the weight but keep the activation full on
    every rank" -- the replicated-activation formulation this engine does not
    implement. Realizing it as plain ``colwise`` would hand the next op a
    feature-sharded activation it does not expect (and, the other way round, a
    left-whole attention module would attend a sequence shard without the
    boundary gather). So the plan is refused by name, with the accepted
    vocabulary and the reason in the message.

    The families that ship these styles (Apertus, GLM-4V, Phi-4-multimodal,
    Llama-4, FlexOlmo) therefore need their plan rewritten or TP left off.
    """
    plan = {"layers.*.self_attn.q_proj": spec}
    with pytest.raises(ValueError, match=spec) as err:
        resolve_tp_sharding_plan(nn.Linear(4, 4), plan=plan)
    message = str(err.value)
    assert "colwise" in message and "rowwise" in message
    assert "replicated-activation" in message


def test_an_unknown_plan_spec_still_raises_with_the_same_advice() -> None:
    """A typo must not be mistaken for a style this engine can honor."""
    with pytest.raises(ValueError, match="not_a_style"):
        resolve_tp_sharding_plan(
            nn.Linear(4, 4), plan={"layers.*.q_proj": "not_a_style"}
        )


@pytest.mark.parametrize(
    "cls", [ColumnParallelLinear, ColwiseLinearNoGather, RowParallelLinear]
)
def test_tp_linear_preserves_frozen_parameters(cls):
    weight = nn.Parameter(torch.randn(8, 8), requires_grad=False)
    module = cls(weight, tp_size=2, tp_rank=0, group=None)
    assert not module.weight.requires_grad


def test_explicit_tp_pattern_overrides_default_wildcard(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.parallel.tensor_parallel import tensor_parallel as tp_module

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    model._tp_config = {"*": "colwise"}
    mesh = {
        "tp": SimpleNamespace(
            get_group=lambda: object(), size=lambda: 2, get_local_rank=lambda: 0
        )
    }
    monkeypatch.setattr(tp_module, "supports_symm_mem", lambda mesh: False)
    apply_tp(
        model,
        mesh,
        ParallelConfig(
            tensor_parallel_size=2,
            tp_config=TensorParallelConfig(tp_rowwise_modules=["0"]),
        ),
    )
    assert isinstance(model[0], RowParallelLinear)
    assert model[0].weight.shape == (4, 2)


@pytest.mark.parametrize("pattern", ["*", ""])
def test_apply_tp_rejects_root_projection_before_mutation(pattern):
    model = nn.Linear(4, 4, bias=False)
    weight = model.weight
    with pytest.raises(ValueError, match="root nn.Linear"):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={pattern: "colwise"},
        )
    assert model.weight is weight
    assert model.weight.shape == (4, 4)
    assert list(model.named_children()) == []


@pytest.mark.parametrize("shared_module", [True, False])
def test_apply_tp_rejects_shared_projection_weights_before_mutation(shared_module):
    model = nn.ModuleDict(
        {"first": nn.Linear(4, 4, bias=False), "second": nn.Linear(4, 4, bias=False)}
    )
    if shared_module:
        model["second"] = model["first"]
    else:
        model["second"].weight = model["first"].weight
    original_modules = list(model.children())
    original_weight = model["first"].weight
    with pytest.raises(ValueError, match="shared weight paths"):
        apply_tp(
            model,
            mesh=object(),
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={"first": "colwise"},
        )
    assert list(model.children()) == original_modules
    assert model["first"].weight is original_weight
    assert model["second"].weight is original_weight


def test_apply_tp_rejects_missing_mesh_when_enabled():
    with pytest.raises(ValueError, match="requires a device mesh"):
        apply_tp(nn.Linear(4, 4), None, ParallelConfig(tensor_parallel_size=2))


def test_apply_tp_rejects_mesh_degree_mismatch_before_mutation():
    from types import SimpleNamespace

    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    original = model[0]
    mesh = {"tp": SimpleNamespace(size=lambda: 4)}
    with pytest.raises(ValueError, match="does not match configured"):
        apply_tp(
            model, mesh, ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
        )
    assert model[0] is original


def test_tp_constructor_failure_preserves_modules_and_allows_retry(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.parallel.tensor_parallel import tensor_parallel as tp_module

    class FailingProjection(nn.Module):
        def __init__(self, weight, **kwargs):
            super().__init__()
            raise RuntimeError("projection construction failed")

    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    model[0].eval()
    original_modules = list(model)
    original_parameters = list(model.parameters())
    original_values = [parameter.detach().clone() for parameter in original_parameters]
    mesh = {
        "tp": SimpleNamespace(
            get_group=lambda: None, size=lambda: 2, get_local_rank=lambda: 0
        )
    }
    monkeypatch.setattr(tp_module, "supports_symm_mem", lambda mesh: False)
    moe_calls = []
    monkeypatch.setattr(
        tp_module,
        "realize_moe_blocks_tp",
        lambda *args, **kwargs: moe_calls.append(args),
    )
    cfg = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(RuntimeError, match="projection construction failed"):
        apply_tp(
            model,
            mesh,
            cfg,
            plan={
                "0": colwise(),
                "1": ShardingConfig(kind="rowwise", implementation=FailingProjection),
            },
        )
    assert list(model) == original_modules
    assert not moe_calls
    for actual, original, value in zip(
        model.parameters(), original_parameters, original_values, strict=True
    ):
        assert actual is original
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    apply_tp(model, mesh, cfg, plan={"0": colwise(), "1": rowwise()})
    assert isinstance(model[0], ColumnParallelLinear)
    assert isinstance(model[1], RowParallelLinear)
    assert not model[0].training
    assert model[1].training
    assert model[0].weight.shape == (2, 4)
    assert model[1].weight.shape == (4, 2)
