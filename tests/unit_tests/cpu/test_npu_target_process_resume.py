"""Pin the schedule and checkpoint contract of the long recovery experiment."""

from types import SimpleNamespace

import pytest

from tests.integration_tests import npu_target_process_resume as recovery


@pytest.mark.parametrize(
    "phase", ["continuous", "initial", "resume", "verify-continuous"]
)
def test_interruption_keeps_full_schedule_and_all_training_states(
    monkeypatch, tmp_path, phase
):
    checkpoint = SimpleNamespace(
        initial_load_path="/weights",
        initial_load_in_hf=True,
        load_step=-1,
    )
    config = SimpleNamespace(
        training=SimpleNamespace(steps=2, checkpoint_config=checkpoint),
        optimizer=SimpleNamespace(
            lr_scheduler_config=SimpleNamespace(total_steps=None)
        ),
    )
    monkeypatch.setattr(recovery, "build_candidate_30b_config", lambda *args: config)
    actual = recovery.recovery_config({}, tmp_path, phase, 8)
    assert actual.training.steps == 100
    assert actual.optimizer.lr_scheduler_config.total_steps == 100
    assert checkpoint.interval == 50
    assert checkpoint.keep_latest_k == 0
    assert checkpoint.load_only is False
    assert checkpoint.last_save_model_only is False
    if phase == "resume":
        assert checkpoint.load_step == 50
        assert checkpoint.initial_load_path == "/weights"
        assert checkpoint.initial_load_in_hf is True
    elif phase == "verify-continuous":
        assert checkpoint.load_step == 100
        assert checkpoint.initial_load_in_hf is True
    else:
        assert checkpoint.load_step == -1
        assert checkpoint.initial_load_path == "/weights"
        assert checkpoint.initial_load_in_hf is True
