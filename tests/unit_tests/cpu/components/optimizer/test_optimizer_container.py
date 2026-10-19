"""``OptimizersContainer``: grouping, validation, and the FQN state round-trip.

The trainer exercises the container end-to-end in ``test_trainer.py``, but only
through a checkpoint round-trip. The pieces worth pinning here are the ones a
round-trip cannot see: which parameters land in which group, that a pattern
matching nothing fails loudly instead of silently optimizing nothing, and that
a ``step()`` fires a registered hook exactly once -- the contract the aux-loss
roll-up depends on.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("torch_param_names")


import pytest
import torch
import torch.nn as nn

from llmtuner.components.optimizer import OptimizersContainer
from llmtuner.components.optimizer import optimizer as optimizer_module
from llmtuner.config import OptimizerConfig, ParamGroupConfig


def _model() -> nn.Module:
    """A model with both a norm (no decay) and projections (decay)."""
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 4),
    )


def _cfg(
    *groups: ParamGroupConfig, implementation: str = "for-loop"
) -> OptimizerConfig:
    return OptimizerConfig(implementation=implementation, param_groups=list(groups))


def _catch_all(lr: float = 0.1, **kwargs) -> ParamGroupConfig:
    return ParamGroupConfig(
        pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": lr, **kwargs}
    )


def test_unmatched_trainable_parameters_fail_before_optimizer_creation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Incomplete parameter plans must fail before optimizer creation")

    monkeypatch.setattr(torch.optim, "AdamW", unexpected)
    group = ParamGroupConfig(
        pattern=r"\.bias$", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.1}
    )
    with pytest.raises(ValueError, match="unassigned.*0.weight"):
        OptimizersContainer(_cfg(group), model_parts=[_model()])


def test_unmatched_frozen_parameters_need_no_optimizer_group():
    model = nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    group = ParamGroupConfig(
        pattern=r"bias$", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.1}
    )
    container = OptimizersContainer(_cfg(group), model_parts=[model])
    assert container.optimizers[0].param_groups[0]["params"] == [model.bias]


def test_unknown_later_optimizer_fails_before_any_optimizer_creation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("All optimizer names must be resolved before construction")

    monkeypatch.setattr(torch.optim, "AdamW", unexpected)
    groups = [
        ParamGroupConfig(
            pattern=r"\.bias$", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.1}
        ),
        ParamGroupConfig(pattern=".*", optimizer_name="unknown"),
    ]
    with pytest.raises(NotImplementedError, match="unknown"):
        OptimizersContainer(_cfg(*groups), model_parts=[_model()])


def test_the_default_config_becomes_a_single_catch_all_group() -> None:
    """An empty ``param_groups`` must not mean "optimize nothing"."""
    cfg = OptimizerConfig(learning_rate=0.25)
    assert len(cfg.param_groups) == 1
    assert cfg.param_groups[0].optimizer_kwargs["lr"] == 0.25

    model = _model()
    container = OptimizersContainer(cfg, model_parts=[model])
    assert len(container) == 1
    assert len(container.optimizers[0].param_groups[0]["params"]) == len(
        list(model.parameters())
    )


def test_patterns_partition_by_first_match() -> None:
    """The norm takes its own weight decay; everything else takes the fallback."""
    model = _model()
    container = OptimizersContainer(
        _cfg(
            ParamGroupConfig(
                pattern=r"^1\.weight$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0},
            ),
            _catch_all(weight_decay=0.01),
        ),
        model_parts=[model],
    )

    norm, rest = container.optimizers[0].param_groups
    assert norm["weight_decay"] == 0.0
    assert rest["weight_decay"] == 0.01
    # First match wins, so the norm is not also in the catch-all.
    assert len(norm["params"]) == 1
    assert len(rest["params"]) == len(list(model.parameters())) - 1


def test_a_pattern_matching_nothing_is_rejected() -> None:
    """A typo'd regex is otherwise a silent no-op that optimizes nothing."""
    unmatched = ParamGroupConfig(
        pattern=r"no_such_parameter",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    # A catch-all is present too, so the only thing wrong is the dead pattern.
    with pytest.raises(ValueError, match="matched no parameters"):
        OptimizersContainer(_cfg(unmatched, _catch_all()), model_parts=[_model()])


def test_a_trainable_parameter_left_out_of_every_group_is_rejected() -> None:
    """Explicit groups without a catch-all would freeze the leftovers."""
    model = _model()
    only_norm = ParamGroupConfig(
        pattern=r"1\.weight$",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    with pytest.raises(ValueError, match="unassigned"):
        OptimizersContainer(_cfg(only_norm), model_parts=[model])


def test_the_unassigned_error_names_the_parameters() -> None:
    """The counts alone cannot say which way the id sets differ, so name them.

    A parameter left out and a parameter double-assigned are both mismatches,
    but only the first is reachable through the public path, and the fix (add a
    catch-all, or a pattern for the named parameters) needs the names.
    """
    model = _model()
    only_norm = ParamGroupConfig(
        pattern=r"1\.weight$",
        optimizer_name="AdamW",
        optimizer_kwargs={"lr": 0.1},
    )
    with pytest.raises(ValueError, match=r"0\.weight \(\(8, 8\)\)"):
        OptimizersContainer(_cfg(only_norm), model_parts=[model])


def test_frozen_parameters_are_not_required_in_a_group() -> None:
    """``requires_grad=False`` parameters are skipped, not counted as missing."""
    model = _model()
    for param in model[0].parameters():
        param.requires_grad = False
    # Only the remaining parameters need a group, and the catch-all takes them.
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    assert (
        len(container.optimizers[0].param_groups[0]["params"])
        == len(list(model.parameters())) - 2
    )


def test_each_model_part_gets_its_own_inner_optimizer() -> None:
    """PP puts each stage's parameters in its own optimizer, hook count aside."""
    parts = [nn.ModuleDict({"first": _model()}), nn.ModuleDict({"second": _model()})]
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=parts)
    assert len(container) == 2
    assert list(container) == container.optimizers


@pytest.mark.parametrize("implementation", ["for-loop", "foreach", "fused"])
def test_the_implementation_flag_reaches_the_inner_optimizer(
    implementation: str,
    monkeypatch,
) -> None:
    """On a device with the fused kernel the flag is passed through verbatim."""
    monkeypatch.setattr(optimizer_module, "device_type", "cuda")
    container = OptimizersContainer(
        _cfg(_catch_all(), implementation=implementation), model_parts=[_model()]
    )
    group = container.optimizers[0].param_groups[0]
    assert group["fused"] is (implementation == "fused")
    assert group["foreach"] is (implementation == "foreach")


def test_fused_downshifts_where_torch_ships_no_fused_kernel(monkeypatch) -> None:
    """``fused`` is the default, so a device without the kernel has to get the
    for-loop one. torch checks the device only for the ``fused=`` *argument*;
    the flag rides in the param group here, so the bare flag used to survive
    construction and kill the first ``step()`` with a NotImplementedError."""
    monkeypatch.setattr(optimizer_module, "device_type", "cpu")

    container = OptimizersContainer(
        _cfg(_catch_all(), implementation="fused"), model_parts=[_model()]
    )

    assert container.optimizers[0].param_groups[0]["fused"] is False
    assert container.optimizers[0].param_groups[0]["foreach"] is False


def test_the_default_implementation_steps_on_a_device_without_a_fused_kernel() -> None:
    """The regression itself: the default config must survive a real step.

    No monkeypatching -- this is the CPU the suite runs on, and the default
    ``implementation="fused"`` is what a laptop run gets.
    """
    model = _model()
    container = OptimizersContainer(
        OptimizerConfig(learning_rate=0.1), model_parts=[model]
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    container.step()

    # A step that ran leaves ``step`` behind in every state dict entry.
    states = container.optimizers[0].state
    assert states and all(state["step"] > 0 for state in states.values())


def test_an_unknown_optimizer_name_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="not added"):
        OptimizersContainer(
            _cfg(
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 0.1},
                )
            ),
            model_parts=[_model()],
        )


def test_an_unknown_implementation_is_rejected() -> None:
    """``implementation`` is a closed enum; a typo must not silently pick a kernel."""
    with pytest.raises(ValueError, match="Unknown optimizer implementation"):
        OptimizersContainer(
            _cfg(_catch_all(), implementation="quantum"), model_parts=[_model()]
        )


def test_a_closure_is_rejected_rather_than_dropped() -> None:
    """``step(closure)`` cannot be honoured, and ignoring it skips the caller's work.

    Returning ``None`` like ``Optimizer.step`` does would look like success while
    the closure -- typically a loss recomputation -- never ran.
    """
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[_model()])
    with pytest.raises(ValueError, match="does not support closures"):
        container.step(lambda: 1.0)


def test_zero_grad_and_step_reach_every_inner_optimizer() -> None:
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    loss = model(torch.ones(2, 8)).pow(2).mean()
    container.zero_grad(set_to_none=True)
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    container.step()
    assert any(
        not torch.equal(a, p) for a, p in zip(before, model.parameters(), strict=False)
    )


def test_a_step_hook_fires_once_per_container_step_not_per_optimizer() -> None:
    """The aux-loss roll-up is registered on the container, so PP must not multiply it.

    Two model parts means two inner optimizers. If the hook were registered on
    each of them it would fire twice per training step and double-count the
    rolled-up aux losses.
    """
    container = OptimizersContainer(
        _cfg(_catch_all()),
        model_parts=[
            nn.ModuleDict({"first": _model()}),
            nn.ModuleDict({"second": _model()}),
        ],
    )
    assert len(container) == 2

    calls: list[int] = []
    container.register_step_pre_hook(lambda *a, **k: calls.append(1))

    model = _model()
    loss = model(torch.ones(2, 8)).pow(2).mean()
    container.zero_grad(set_to_none=True)
    loss.backward()
    container.step()

    assert len(calls) == 1


def test_state_dict_is_flat_and_fqn_keyed() -> None:
    """The format DCP needs: one key per (parameter, state tensor), no indices."""
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    container.zero_grad(set_to_none=True)
    model(torch.ones(2, 8)).pow(2).mean().backward()
    container.step()

    state = container.state_dict()
    assert "state.0.weight.exp_avg" in state
    assert "state.0.weight.exp_avg_sq" in state
    # No integer-indexed nesting survives the flattening.
    assert all(isinstance(key, str) for key in state)
    assert not any(key.startswith("state.0.") and key.count(".") == 2 for key in state)


def test_state_round_trips_into_a_fresh_container() -> None:
    """A resumed run's cold optimizer must come back warm, bit for bit."""
    model = _model()
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=[model])
    for _ in range(2):
        container.zero_grad(set_to_none=True)
        model(torch.ones(2, 8)).pow(2).mean().backward()
        container.step()
    saved = {
        key: value.clone()
        for key, value in container.state_dict().items()
        if key.endswith(("exp_avg", "exp_avg_sq"))
    }
    assert saved, "sanity: the run must have produced optimizer state"

    fresh_model = _model()
    fresh = OptimizersContainer(_cfg(_catch_all()), model_parts=[fresh_model])
    # A cold container reports no state until something materializes it, which
    # is exactly what load_state_dict does first.
    fresh.load_state_dict(container.state_dict())

    restored = {
        key: value
        for key, value in fresh.state_dict().items()
        if key.endswith(("exp_avg", "exp_avg_sq"))
    }
    assert restored.keys() == saved.keys()
    for key, value in saved.items():
        assert torch.equal(restored[key], value)


def test_container_owns_model_part_collection():
    model = _model()
    parts = [model]
    container = OptimizersContainer(_cfg(_catch_all()), model_parts=parts)
    parts.clear()
    assert container.model_parts == [model]
    assert container.state_dict()


@pytest.mark.parametrize("field", ["params", "param_names"])
def test_param_group_options_cannot_override_parameter_ownership(field):
    model = torch.nn.Linear(2, 2)
    config = OptimizerConfig(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1, field: []},
            )
        ],
    )
    with pytest.raises(ValueError, match="managed parameter fields"):
        OptimizersContainer(config, model_parts=[model])


def test_optimizer_rejects_parameter_names_colliding_after_wrapper_removal():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
    model.add_module("_checkpoint_wrapped_module", torch.nn.Linear(2, 2))

    with pytest.raises(ValueError, match="FQNs collide after wrapper removal"):
        OptimizersContainer(
            OptimizerConfig(implementation="for-loop"), model_parts=[model]
        )


@pytest.mark.parametrize("invalid_layout", ["repeated_part", "later_pattern_mismatch"])
def test_all_model_parts_are_validated_before_building_optimizers(
    monkeypatch, invalid_layout
):
    def unexpected_constructor(*args, **kwargs):
        pytest.fail("all parameter groups must be validated before construction")

    monkeypatch.setattr(torch.optim, "AdamW", unexpected_constructor)
    model = _model()
    if invalid_layout == "repeated_part":
        parts = [model, model]
        config = _cfg(_catch_all())
        error = "more than one group"
    else:
        parts = [model, nn.Linear(8, 8)]
        config = _cfg(
            ParamGroupConfig(
                pattern=r"^1\.weight$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
            ParamGroupConfig(pattern=r"^[01]\.", optimizer_name="AdamW"),
        )
        error = "unassigned"
    with pytest.raises(ValueError, match=error):
        OptimizersContainer(config, model_parts=parts)


def test_parameter_group_pattern_matches_a_shared_parameter_alias():
    model = nn.Module()
    model.first = nn.Linear(2, 2, bias=False)
    model.second = nn.Linear(2, 2, bias=False)
    model.second.weight = model.first.weight
    config = _cfg(
        ParamGroupConfig(
            pattern=r"^second\.weight$",
            optimizer_name="AdamW",
            optimizer_kwargs={"lr": 0.01},
        )
    )
    container = OptimizersContainer(config, model_parts=[model])
    group = container.optimizers[0].param_groups[0]
    assert len(group["params"]) == 1
    assert group["params"][0] is model.first.weight
    assert group["param_names"] == ["first.weight"]
    model.first.weight.grad = torch.ones_like(model.first.weight)
    container.step()
    assert len(container.optimizers[0].state) == 1

    from copy import deepcopy

    restored_model = deepcopy(model)
    restored = OptimizersContainer(
        _cfg(
            ParamGroupConfig(
                pattern=r"^first\.weight$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.5},
            )
        ),
        model_parts=[restored_model],
    )
    restored.load_state_dict(deepcopy(container.state_dict()))
    model.first.weight.grad = torch.full_like(model.first.weight, 2)
    restored_model.first.weight.grad = model.first.weight.grad.clone()
    container.step()
    restored.step()
    torch.testing.assert_close(model.first.weight, restored_model.first.weight)
    assert restored_model.first.weight is restored_model.second.weight


def test_cross_part_checkpoint_name_collision_fails_before_optimizer_creation(
    monkeypatch,
):
    def unexpected(*args, **kwargs):
        pytest.fail("Checkpoint FQN conflicts must fail before optimizer creation")

    monkeypatch.setattr(torch.optim, "AdamW", unexpected)
    with pytest.raises(ValueError, match="checkpoint FQNs.*weight"):
        OptimizersContainer(
            _cfg(_catch_all()), model_parts=[nn.Linear(2, 2), nn.Linear(2, 2)]
        )


def test_patterns_can_target_different_local_model_parts():
    first = nn.Module()
    first.register_parameter("first_weight", nn.Parameter(torch.tensor([1.0])))
    second = nn.Module()
    second.register_parameter("second_weight", nn.Parameter(torch.tensor([2.0])))
    config = _cfg(
        ParamGroupConfig(
            pattern="^first_weight$",
            optimizer_name="AdamW",
            optimizer_kwargs={"lr": 0.01},
        ),
        ParamGroupConfig(
            pattern="^second_weight$",
            optimizer_name="AdamW",
            optimizer_kwargs={"lr": 0.02},
        ),
    )
    optimizer = OptimizersContainer(config, model_parts=[first, second])
    assert len(optimizer) == 2
    assert [inner.param_groups[0]["lr"] for inner in optimizer] == [0.01, 0.02]
    reference_parameters = [nn.Parameter(torch.tensor([value])) for value in (1.0, 2.0)]
    references = [
        torch.optim.AdamW([parameter], lr=rate)
        for parameter, rate in zip(reference_parameters, (0.01, 0.02), strict=True)
    ]
    for parameter in (first.first_weight, second.second_weight, *reference_parameters):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for reference in references:
        reference.step()
    for actual, expected in zip(
        (first.first_weight, second.second_weight), reference_parameters, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_duplicate_rules_remain_invalid_across_multiple_local_parts():
    parts = []
    for name in ("first_weight", "second_weight"):
        part = nn.Module()
        part.register_parameter(name, nn.Parameter(torch.ones(1)))
        parts.append(part)
    with pytest.raises(ValueError, match="matched no parameters"):
        OptimizersContainer(_cfg(_catch_all(), _catch_all()), model_parts=parts)


@pytest.mark.parametrize("name", ["Adam", "AdamW"])
def test_flat_optimizer_type_controls_runtime_and_decay_update(name):
    model = nn.Linear(2, 1, bias=False)
    reference = nn.Linear(2, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    cfg = OptimizerConfig(
        optimizer_type=name,
        implementation="for-loop",
        learning_rate=0.01,
        weight_decay=0.3,
    )
    container = OptimizersContainer(cfg, model_parts=[model])
    expected_type = getattr(torch.optim, name)
    assert type(container.optimizers[0]) is expected_type
    expected = expected_type(
        reference.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
        foreach=False,
        fused=False,
    )
    for value in (0.2, -0.1):
        model.weight.grad = torch.full_like(model.weight, value)
        reference.weight.grad = torch.full_like(reference.weight, value)
        container.step()
        expected.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["SGD", "adam", "", None, True])
def test_flat_optimizer_type_rejects_unknown_values(name):
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="optimizer_type"):
        OptimizerConfig(optimizer_type=name)
