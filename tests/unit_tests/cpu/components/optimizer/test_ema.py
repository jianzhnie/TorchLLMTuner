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
    ema = EMA(model_parts=[first, second], decay=0.5)
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
        model_parts=[first, second], decay=0.5, buffer_patterns=["expert_bias_E$"]
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
