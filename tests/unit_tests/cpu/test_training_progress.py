"""The native progress format is independent of training services."""

import pytest

from llmtuner.trainer.runtime.progress import TrainingProgress


def test_progress_load_preserves_native_keys_and_identity() -> None:
    progress = TrainingProgress(step=3, ntokens_seen=24)
    checkpoint_reference = progress
    saved = progress.state_dict()
    progress.step = 8
    assert saved == {"step": 3, "ntokens_seen": 24}
    progress.load_state_dict({"step": 5, "ntokens_seen": 96})
    assert checkpoint_reference is progress
    assert checkpoint_reference.state_dict() == {"step": 5, "ntokens_seen": 96}


@pytest.mark.parametrize("field", ["step", "ntokens_seen"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_counter_does_not_partially_restore(field: str, value: object) -> None:
    progress = TrainingProgress(step=7, ntokens_seen=42)
    state = {"step": 9, "ntokens_seen": 72, field: value}
    with pytest.raises(ValueError, match=f"trainer {field}"):
        progress.load_state_dict(state)
    assert progress.state_dict() == {"step": 7, "ntokens_seen": 42}


@pytest.mark.parametrize("missing", ["step", "ntokens_seen"])
def test_missing_counter_does_not_partially_restore(missing: str) -> None:
    progress = TrainingProgress(step=7, ntokens_seen=42)
    state = {"step": 9, "ntokens_seen": 72}
    del state[missing]
    with pytest.raises(KeyError):
        progress.load_state_dict(state)
    assert progress.state_dict() == {"step": 7, "ntokens_seen": 42}
