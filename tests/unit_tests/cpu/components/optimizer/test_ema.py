"""``EMA``: update math, decay schedule, gating, and the FQN state round-trip.

The pieces pinned here are the ones a training run cannot see: the exact lerp
a firing computes, the step-derived firing count (which is what makes the decay
schedule survive a resume -- only ``ema_params`` are checkpointed), and that the
state dict is the same flat, FQN-keyed layout the real optimizer container
produces, which is the contract DCP resharding relies on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llmtuner.components.optimizer import EMA, OptimizersContainer
from llmtuner.config import EMAConfig, OptimizerConfig


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _ema_params(ema: EMA) -> dict[str, torch.Tensor]:
    """Live ``ema_params`` tensors, keyed by FQN."""
    out = {}
    for ema_opt in ema.optimizers:
        for group in ema_opt.param_groups:
            for fqn, t in zip(group["param_names"], group["params"], strict=True):
                out[fqn] = ema_opt.state[t]["ema_params"]
    return out


def test_fixed_decay_matches_a_hand_computed_lerp() -> None:
    model = _model()
    ema = EMA(model_parts=[model], decay=0.75)
    initial = {fqn: t.clone() for fqn, t in _ema_params(ema).items()}

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.arange(p.numel(), dtype=p.dtype).view(p.shape))
    current = {name: p.detach().clone() for name, p in model.named_parameters()}

    ema.step(1)

    for fqn, t in _ema_params(ema).items():
        expected = 0.75 * initial[fqn] + 0.25 * current[fqn]
        torch.testing.assert_close(t, expected)


def test_the_first_dynamic_update_nearly_overwrites_the_average() -> None:
    """``decay = 2 ** (-1 / (half_life_fraction * num_updates))``: at firing 1
    with the default 0.05 that is 2**-20, so the EMA starts from the live
    weights rather than dragging its random-init clone around."""
    model = _model()
    ema = EMA(model_parts=[model])  # half_life_fraction=0.05
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema.step(1)

    decay = 2.0**-20
    for name, p in model.named_parameters():
        expected = decay * (p.detach() - 1.0) + (1 - decay) * p.detach()
        torch.testing.assert_close(_ema_params(ema)[name], expected)


@pytest.mark.parametrize(
    "step, fixed_decay, expected_decay",
    [(1, None, 2.0**-20), (20, None, 0.5), (200, None, 2.0**-0.1), (1, 0.9, 0.9)],
)
def test_decay_schedule_controls_the_actual_update(
    step, fixed_decay, expected_decay
) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = EMA(model_parts=[model], decay=fixed_decay)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.step(step)
    torch.testing.assert_close(
        _ema_params(ema)["weight"], torch.full_like(model.weight, 1 - expected_decay)
    )


def test_step_gating_against_start_step_and_update_every_n_steps() -> None:
    model = _model()
    ema = EMA(model_parts=[model], decay=0.5, start_step=2, update_every_n_steps=3)
    snapshot = {fqn: t.clone() for fqn, t in _ema_params(ema).items()}

    def fire_times(steps):
        for s in steps:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1.0)
            ema.step(s)

    # First firing lands at start_step + update_every_n_steps = 5.
    fire_times([3, 4])
    for fqn, t in _ema_params(ema).items():
        torch.testing.assert_close(t, snapshot[fqn])
    fire_times([5])
    assert not torch.equal(_ema_params(ema)["0.weight"], snapshot["0.weight"]), (
        "step 5 should fire"
    )


def test_step_bias_counts_extra_firings() -> None:
    model = _model()
    ema = EMA(model_parts=[model], step_bias=19)
    initial = {fqn: t.clone() for fqn, t in _ema_params(ema).items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema.step(1)

    # The first firing computes decay at num_updates = 1 + 19 = 20, i.e. 0.5,
    # instead of the 2**-20 an unbiased first firing would use.
    for name, p in model.named_parameters():
        expected = 0.5 * initial[name] + 0.5 * p.detach()
        torch.testing.assert_close(_ema_params(ema)[name], expected)


def test_frozen_parameters_are_not_tracked() -> None:
    model = _model()
    model[1].weight.requires_grad_(False)
    ema = EMA(model_parts=[model])
    assert "state.1.weight.ema_params" not in ema.state_dict()
    assert "state.1.bias.ema_params" in ema.state_dict()


def test_state_dict_is_flat_fqn_keyed_like_the_optimizer_container() -> None:
    model = _model()
    ema_keys = set(EMA(model_parts=[model]).state_dict())
    assert ema_keys == {
        f"state.{name}.ema_params" for name, _ in model.named_parameters()
    }

    container = OptimizersContainer(
        OptimizerConfig(implementation="for-loop"), model_parts=[model]
    )
    optim_fqns = {
        key[len("state.") :].rsplit(".", 1)[0]
        for key in container.state_dict()
        if key.startswith("state.")
    }
    ema_fqns = {key[len("state.") :].rsplit(".", 1)[0] for key in ema_keys}
    assert ema_fqns == optim_fqns


def _drive(ema: EMA, model: nn.Module, steps) -> None:
    """Deterministic pseudo-training: bump the weights, then fire the EMA."""
    for s in steps:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * s)
        ema.step(s)


def test_state_dict_round_trip_and_continued_updates() -> None:
    model = _model()
    ema = EMA(model_parts=[model])
    _drive(ema, model, range(1, 6))
    saved = {k: v.clone() for k, v in ema.state_dict().items()}

    restored = EMA(model_parts=[model])
    restored.load_state_dict(saved)
    assert set(restored.state_dict()) == set(saved)
    for fqn in _ema_params(ema):
        torch.testing.assert_close(_ema_params(restored)[fqn], _ema_params(ema)[fqn])

    # A resumed EMA keeps aging from the step number, not from zero: driving
    # both through the same later steps leaves them identical. Bump the model
    # once per step and fire both EMAs against that single trajectory.
    for s in range(6, 11):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * s)
        ema.step(s)
        restored.step(s)
    for fqn in _ema_params(ema):
        torch.testing.assert_close(_ema_params(restored)[fqn], _ema_params(ema)[fqn])


def test_a_resume_reproduces_the_uninterrupted_average() -> None:
    """num_updates is derived from the step, so resuming from a mid-run
    checkpoint continues the dynamic schedule instead of collapsing the decay
    to a full overwrite (what a stored, zeroed counter would do)."""
    model = _model()
    continuous = EMA(model_parts=[model])
    _drive(continuous, model, range(1, 6))
    saved = {k: v.clone() for k, v in continuous.state_dict().items()}
    _drive(continuous, model, range(6, 11))

    torch.manual_seed(0)
    resumed_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    # Replay the pre-checkpoint weight trajectory: the resume also reloads the
    # model checkpoint at step 5, so the weights start from init + sum of the
    # step 1..5 bumps, not from init.
    with torch.no_grad():
        for p in resumed_model.parameters():
            p.add_(0.1 * sum(range(1, 6)))
    resumed = EMA(model_parts=[resumed_model])
    resumed.load_state_dict(saved)
    _drive(resumed, resumed_model, range(6, 11))

    for fqn in _ema_params(continuous):
        torch.testing.assert_close(
            _ema_params(resumed)[fqn], _ema_params(continuous)[fqn]
        )


def test_an_empty_state_dict_cold_starts_from_the_model_weights() -> None:
    model = _model()
    ema = EMA(model_parts=[model])
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)

    ema.load_state_dict({})

    for name, p in model.named_parameters():
        torch.testing.assert_close(_ema_params(ema)[name], p.detach())


def test_matching_buffers_are_tracked_under_the_same_flat_layout() -> None:
    model = _model()
    model.register_buffer("expert_bias_E", torch.zeros(4))
    model.register_buffer("num_batches_tracked_like", torch.zeros(4))
    ema = EMA(model_parts=[model], decay=0.5, buffer_patterns=["expert_bias_E$"])

    assert "state.expert_bias_E.ema_params" in ema.state_dict()
    assert "state.num_batches_tracked_like.ema_params" not in ema.state_dict()

    model.expert_bias_E.add_(2.0)
    ema.step(1)
    torch.testing.assert_close(_ema_params(ema)["expert_bias_E"], torch.full((4,), 1.0))

    # And the cold start reseeds buffers too.
    model.expert_bias_E.fill_(8.0)
    ema.load_state_dict({})
    torch.testing.assert_close(_ema_params(ema)["expert_bias_E"], torch.full((4,), 8.0))


def test_an_integer_buffer_match_is_rejected() -> None:
    model = _model()
    model.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
    with pytest.raises(ValueError, match="step_counter"):
        EMA(model_parts=[model], buffer_patterns=["step_counter"])


@pytest.mark.parametrize("invalid_pattern", [True, False])
def test_invalid_buffer_selection_fails_before_averages_are_allocated(
    monkeypatch, invalid_pattern
):
    import re

    from llmtuner.components.optimizer import ema as ema_module

    def fail_if_allocated(*args, **kwargs):
        pytest.fail("EMA allocated parameter averages before validating patterns")

    monkeypatch.setattr(ema_module, "EMAStateOptimizer", fail_if_allocated)
    model = _model()
    model.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
    with pytest.raises(re.error if invalid_pattern else ValueError):
        EMA(
            model_parts=[model],
            buffer_patterns=["[" if invalid_pattern else "step_counter"],
        )


def test_a_replaced_parameter_is_a_loud_error_not_a_keyerror() -> None:
    model = _model()
    ema = EMA(model_parts=[model])
    model[0].weight = nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(RuntimeError, match="no state for"):
        ema.step(1)


def test_the_inner_optimizer_cannot_be_stepped() -> None:
    ema = EMA(model_parts=[_model()])
    with pytest.raises(RuntimeError, match="must not be step"):
        ema.optimizers[0].step()


@pytest.mark.parametrize("name", ["start_step", "step_bias", "update_every_n_steps"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_ema_step_configuration_rejects_invalid_integers(name, value):
    with pytest.raises(ValueError, match=name):
        EMAConfig(**{name: value})
    with pytest.raises(ValueError, match=name):
        EMA(model_parts=[_model()], **{name: value})


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_invalid_step_preserves_ema_weights(step):
    model = _model()
    ema = EMA(model_parts=[model])
    before = {name: value.clone() for name, value in _ema_params(ema).items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    with pytest.raises(ValueError, match="current_step"):
        ema.step(step)
    for name, value in _ema_params(ema).items():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize(
    "name, value",
    [
        ("decay", -0.1),
        ("decay", 1.0),
        ("decay", float("nan")),
        ("decay", float("inf")),
        ("decay", True),
        ("decay", False),
        ("decay", "0.5"),
        ("half_life_fraction", 0.0),
        ("half_life_fraction", -0.1),
        ("half_life_fraction", float("nan")),
        ("half_life_fraction", float("inf")),
        ("half_life_fraction", True),
        ("half_life_fraction", False),
        ("half_life_fraction", "0.05"),
        ("half_life_fraction", None),
    ],
)
def test_invalid_ema_schedule_is_rejected_before_allocation(monkeypatch, name, value):
    from llmtuner.components.optimizer import ema as ema_module

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("invalid schedule must fail before allocating averages")

    monkeypatch.setattr(ema_module, "EMAStateOptimizer", unexpected_allocation)
    with pytest.raises(ValueError, match=name):
        EMAConfig(**{name: value})
    with pytest.raises(ValueError, match=name):
        EMA(model_parts=[_model()], **{name: value})


def test_config_validation() -> None:
    EMAConfig()  # defaults are valid
    with pytest.raises(ValueError, match="update_every_n_steps"):
        EMAConfig(update_every_n_steps=0)
    with pytest.raises(ValueError, match="half_life_fraction"):
        EMAConfig(half_life_fraction=0.0)
    with pytest.raises(ValueError, match="step_bias"):
        EMAConfig(step_bias=-1)
    with pytest.raises(ValueError, match="decay"):
        EMAConfig(decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        EMAConfig(decay=-0.1)


def test_cold_start_rejects_replaced_parameter_without_partial_updates():
    model = _model()
    ema = EMA(model_parts=[model])
    before = {key: value.clone() for key, value in _ema_params(ema).items()}
    model[1].weight = nn.Parameter(torch.zeros_like(model[1].weight))
    replacement = model[1].weight
    with pytest.raises(RuntimeError, match="no state for"):
        ema.load_state_dict({})
    assert replacement not in ema.optimizers[0].state
    for key, value in _ema_params(ema).items():
        torch.testing.assert_close(value, before[key])


def test_cold_start_rejects_replaced_buffer_without_inserting_state():
    model = _model()
    model.register_buffer("expert_bias_E", torch.ones(4))
    ema = EMA(model_parts=[model], buffer_patterns=["expert_bias_E$"])
    model.expert_bias_E = torch.zeros(4)
    with pytest.raises(RuntimeError, match="no state for"):
        ema.load_state_dict({})
    assert model.expert_bias_E not in ema._buffer_optimizers[0].state


def test_ema_owns_model_part_collection():
    model = _model()
    parts = [model]
    ema = EMA(model_parts=parts, decay=0.5)
    params = list(model.parameters())
    initial = [p.detach().clone() for p in params]
    parts.clear()
    with torch.no_grad():
        for p in params:
            p.add_(2)
    ema.step(1)
    averages = [ema._param_optimizers[0].state[p]["ema_params"] for p in params]
    for average, before in zip(averages, initial, strict=True):
        torch.testing.assert_close(average, before + 1)


def test_repeated_model_part_is_rejected_before_allocating_averages(monkeypatch):
    from llmtuner.components.optimizer import ema as ema_module

    def fail_if_allocated(*args, **kwargs):
        pytest.fail("EMA allocated averages before checking parameter ownership")

    monkeypatch.setattr(ema_module, "EMAStateOptimizer", fail_if_allocated)
    model = _model()
    with pytest.raises(ValueError, match="more than one group"):
        EMA(model_parts=[model, model])


def test_update_rejects_later_replaced_parameter_before_updating_earlier_part():
    first, second = _model(), _model()
    ema = EMA(
        model_parts=[
            nn.ModuleDict({"first": first}),
            nn.ModuleDict({"second": second}),
        ],
        decay=0.5,
    )
    averages = [
        state["ema_params"]
        for optimizer in ema.optimizers
        for state in optimizer.state.values()
    ]
    before = [average.clone() for average in averages]
    with torch.no_grad():
        for param in first.parameters():
            param.add_(1)
    second[0].weight = torch.nn.Parameter(second[0].weight.detach().clone())
    with pytest.raises(RuntimeError, match="EMA has no state"):
        ema.step(1)
    for average, expected in zip(averages, before, strict=True):
        torch.testing.assert_close(average, expected)


@pytest.mark.parametrize("operation", ["update", "cold_start"])
@pytest.mark.parametrize(
    "change", ["remove_parameter", "freeze_parameter", "remove_buffer"]
)
def test_ema_rejects_shrinking_scope_before_changing_averages(operation, change):
    first, second = _model(), _model()
    second.register_buffer("expert_bias_E", torch.ones(4))
    ema = EMA(
        model_parts=[
            nn.ModuleDict({"first": first}),
            nn.ModuleDict({"second": second}),
        ],
        decay=0.5,
        buffer_patterns=["expert_bias_E$"],
    )
    averages = [
        state["ema_params"]
        for optimizer in ema.optimizers
        for state in optimizer.state.values()
    ]
    before = [average.clone() for average in averages]
    with torch.no_grad():
        for parameter in first.parameters():
            parameter.add_(2)
    if change == "remove_parameter":
        second[0].register_parameter("weight", None)
    elif change == "freeze_parameter":
        second[0].weight.requires_grad_(False)
    else:
        del second.expert_bias_E

    with pytest.raises(RuntimeError, match="tracked scope lost"):
        if operation == "update":
            ema.step(1)
        else:
            ema.load_state_dict({})

    for average, expected in zip(averages, before, strict=True):
        torch.testing.assert_close(average, expected)


@pytest.mark.parametrize("track_buffers", [False, True])
def test_ema_rejects_tensor_names_colliding_after_wrapper_removal(track_buffers):
    from llmtuner.components.optimizer.ema import EMAStateOptimizer

    tensor_factory = (
        torch.ones if track_buffers else lambda n: nn.Parameter(torch.ones(n))
    )
    tensors = [
        ("weight", tensor_factory(2)),
        ("_checkpoint_wrapped_module.weight", tensor_factory(2)),
    ]
    with pytest.raises(ValueError, match="EMA tensor FQNs collide"):
        EMAStateOptimizer(tensors)


@pytest.mark.parametrize("state", [None, [], "", [1], "invalid"])
def test_invalid_checkpoint_cannot_cold_start_or_change_ema(state):
    model = _model()
    ema = EMA(model_parts=[model], decay=0.75)
    before = {name: value.clone() for name, value in _ema_params(ema).items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2)
    with pytest.raises(ValueError, match="dictionary"):
        ema.load_state_dict(state)
    for name, average in _ema_params(ema).items():
        torch.testing.assert_close(average, before[name], rtol=0, atol=0)
    ema.step(1)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            _ema_params(ema)[name], before[name] * 0.75 + parameter.detach() * 0.25
        )


def test_buffer_pattern_matches_shared_alias_without_duplicate_averaging():
    model = _model()
    model.register_buffer("first_bias", torch.zeros(4))
    model.register_buffer("second_bias", model.first_bias)
    ema = EMA(model_parts=[model], decay=0.5, buffer_patterns=["second_bias$"])
    assert len(ema._buffer_optimizers[0].state) == 1
    assert ema._buffer_optimizers[0].param_groups[0]["param_names"] == ["first_bias"]
    model.first_bias.fill_(2)
    ema.step(1)
    torch.testing.assert_close(_ema_params(ema)["first_bias"], torch.ones(4))

    from copy import deepcopy

    resumed_model = deepcopy(model)
    resumed = EMA(
        model_parts=[resumed_model], decay=0.5, buffer_patterns=["first_bias$"]
    )
    resumed.load_state_dict(deepcopy(ema.state_dict()))
    model.first_bias.fill_(4)
    resumed_model.first_bias.fill_(4)
    ema.step(2)
    resumed.step(2)
    torch.testing.assert_close(
        _ema_params(ema)["first_bias"], _ema_params(resumed)["first_bias"]
    )
    assert resumed_model.first_bias is resumed_model.second_bias


@pytest.mark.parametrize("second_buffer", [False, True])
def test_cross_part_ema_names_fail_before_state_allocation(monkeypatch, second_buffer):
    from llmtuner.components.optimizer import ema as ema_module

    first, second = nn.Module(), nn.Module()
    first.register_parameter("weight", nn.Parameter(torch.ones(2)))
    if second_buffer:
        second.register_buffer("weight", torch.ones(2))
    else:
        second.register_parameter("weight", nn.Parameter(torch.ones(2)))

    def unexpected(*args, **kwargs):
        pytest.fail("EMA checkpoint FQN conflicts must fail before state allocation")

    monkeypatch.setattr(ema_module, "EMAStateOptimizer", unexpected)
    with pytest.raises(ValueError, match="EMA tensor FQNs collide.*weight"):
        EMA(
            model_parts=[first, second], buffer_patterns=[".*"] if second_buffer else []
        )


@pytest.mark.parametrize("operation", ["step", "cold_start"])
@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_metadata_change_is_rejected_before_any_average_mutation(operation, change):
    model = nn.Linear(2, 2)
    model.register_buffer("history", torch.zeros(2))
    ema = EMA(model_parts=[model], decay=0.5, buffer_patterns=["history"])
    before = {name: value.clone() for name, value in _ema_params(ema).items()}
    with torch.no_grad():
        model.weight.add_(1)
        if change == "shape":
            model.history.resize_(3)
        else:
            model.history.data = model.history.double()
    with pytest.raises(RuntimeError, match="metadata changed"):
        if operation == "step":
            ema.step(1)
        else:
            ema.load_state_dict({})
    for name, value in _ema_params(ema).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["step", "cold_start", "save", "load"])
@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_renamed_tensor_is_rejected_before_any_average_mutation(operation, kind):
    model = nn.Linear(2, 2)
    model.register_buffer("history", torch.zeros(2))
    ema = EMA(model_parts=[model], decay=0.5, buffer_patterns=["history"])
    before = {name: value.clone() for name, value in _ema_params(ema).items()}
    saved = {name: value.clone() for name, value in ema.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1)
    name = "bias" if kind == "parameter" else "history"
    tensor = getattr(model, name)
    delattr(model, name)
    if kind == "parameter":
        model.register_parameter("renamed_bias", tensor)
    else:
        model.register_buffer("renamed_history", tensor)
    with pytest.raises(RuntimeError, match="checkpoint name changed"):
        if operation == "step":
            ema.step(1)
        elif operation == "cold_start":
            ema.load_state_dict({})
        elif operation == "save":
            ema.state_dict()
        else:
            ema.load_state_dict(saved)
    for name, value in _ema_params(ema).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_dynamic_ema_sparse_firings_resume_parameter_and_buffer_against_formula():
    import copy

    model = nn.Module()
    model.register_parameter(
        "weight", nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    )
    model.register_buffer("history", torch.tensor([2.0], dtype=torch.float64))
    options = dict(
        half_life_fraction=0.5,
        start_step=2,
        step_bias=3,
        update_every_n_steps=3,
        buffer_patterns=["history$"],
    )
    continuous = EMA(model_parts=[model], **options)
    expected = {"weight": 1.0, "history": 2.0}
    resumed = None
    resumed_model = None
    for step in range(1, 15):
        with torch.no_grad():
            model.weight.fill_(1.0 + step * 0.25)
            model.history.fill_(2.0 - step * 0.5)
        continuous.step(step)
        elapsed = step - 2
        if elapsed > 0 and elapsed % 3 == 0:
            firing = elapsed // 3 + 3
            decay = 2 ** (-1 / (0.5 * firing))
            for name in expected:
                live = getattr(model, name).item()
                expected[name] = decay * expected[name] + (1 - decay) * live
        if resumed is not None:
            assert resumed_model is not None
            with torch.no_grad():
                resumed_model.weight.copy_(model.weight)
                resumed_model.history.copy_(model.history)
            resumed.step(step)
            torch.testing.assert_close(
                resumed.state_dict(), continuous.state_dict(), rtol=0, atol=0
            )
        for name, value in expected.items():
            torch.testing.assert_close(
                continuous.state_dict()[f"state.{name}.ema_params"],
                torch.tensor([value], dtype=torch.float64),
                rtol=1e-14,
                atol=1e-14,
            )
        if step == 7:  # Between firings at steps 5 and 8.
            resumed_model = copy.deepcopy(model)
            resumed = EMA(model_parts=[resumed_model], **options)
            resumed.load_state_dict(copy.deepcopy(continuous.state_dict()))
    assert resumed is not None


@pytest.mark.parametrize("complex_dtype", [torch.complex64, torch.complex128])
def test_mixed_real_complex_buffer_ema_resume_matches_independent_formula(
    complex_dtype,
):
    import copy

    model = nn.Linear(2, 2, dtype=torch.float64)
    # Dyadic values keep both independent formula and lerp exactly representable.
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(-0.25)
    model.register_buffer(
        "history", torch.tensor([1 + 2j, -3 + 4j], dtype=complex_dtype)
    )
    options = dict(decay=0.5, buffer_patterns=["history$"])
    ema = EMA(model_parts=[model], **options)
    expected = {
        name: tensor.detach().clone()
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
    }
    resumed_model = resumed = None
    for step in range(1, 5):
        with torch.no_grad():
            model.weight.fill_(step * 0.25)
            model.bias.fill_(-step * 0.5)
            model.history.copy_(
                torch.tensor([step + 2j, -3 - step * 1j], dtype=complex_dtype)
            )
        live = dict([*model.named_parameters(), *model.named_buffers()])
        for name in expected:
            expected[name] = expected[name] * 0.5 + live[name].detach() * 0.5
        ema.step(step)
        state = ema.state_dict()
        for name, tensor in expected.items():
            torch.testing.assert_close(
                state[f"state.{name}.ema_params"], tensor, rtol=0, atol=0
            )
        if resumed is not None:
            resumed_model.load_state_dict(model.state_dict())
            resumed.step(step)
            torch.testing.assert_close(resumed.state_dict(), state, rtol=0, atol=0)
        if step == 2:
            resumed_model = copy.deepcopy(model)
            resumed = EMA(model_parts=[resumed_model], **options)
            resumed.load_state_dict(copy.deepcopy(state))
            torch.testing.assert_close(resumed.state_dict(), state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_buffer_only_ema_with_all_parameters_frozen_can_update_and_resume(dtype):
    from copy import deepcopy

    model = torch.nn.Linear(2, 2, dtype=dtype).requires_grad_(False)
    model.register_buffer("expert_bias", torch.tensor([0.2, -0.2], dtype=dtype))
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema = EMA(model_parts=[model], decay=0.5, buffer_patterns=["expert_bias$"])
    assert len(ema.param_groups[0]["params"]) == 1
    assert ema.param_groups[0]["params"][0] is model.expert_bias
    assert set(ema.state_dict()) == {"state.expert_bias.ema_params"}
    initial = model.expert_bias.clone()
    model.expert_bias.add_(1)
    ema.step(1)
    expected = initial.lerp(model.expert_bias, 0.5)
    torch.testing.assert_close(
        ema.state_dict()["state.expert_bias.ema_params"], expected, rtol=0, atol=0
    )
    restored_model = deepcopy(model)
    restored = EMA(
        model_parts=[restored_model], decay=0.5, buffer_patterns=["expert_bias$"]
    )
    restored.load_state_dict(deepcopy(ema.state_dict()))
    for current in (model, restored_model):
        current.expert_bias.sub_(0.25)
    ema.step(2)
    restored.step(2)
    expected = expected.lerp(model.expert_bias, 0.5)
    for current in (ema, restored):
        torch.testing.assert_close(
            current.state_dict()["state.expert_bias.ema_params"],
            expected,
            rtol=0,
            atol=0,
        )
    for current in (model, restored_model):
        for name, parameter in current.named_parameters():
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)


@pytest.mark.parametrize("parts", ["frozen", "parameterless", "none"])
def test_empty_ema_scope_can_step_checkpoint_and_cold_start(parts):
    model = torch.nn.Linear(2, 2).requires_grad_(False)
    model_parts = {
        "frozen": [model],
        "parameterless": [torch.nn.Identity()],
        "none": [],
    }[parts]
    ema = EMA(model_parts=model_parts, decay=0.5)
    assert ema.param_groups[0]["params"] == []
    assert ema.state_dict() == {}
    ema.step(1)
    ema.load_state_dict({})
    assert ema.state_dict() == {}
