import copy

import pytest
import torch

from tests.integration_tests.compare_compile_training_states import (
    compare_training_states,
)


@pytest.fixture
def snapshots(tmp_path):
    state = {
        "logits": torch.ones(2, 3),
        "candidate": {"weight": {"weight": torch.ones(3), "gradient": None}},
        "reference": {"weight": {"weight": torch.ones(3), "gradient": torch.zeros(3)}},
    }
    folders = (tmp_path / "expected", tmp_path / "actual")
    for folder in folders:
        folder.mkdir()
        torch.save(state, folder / "rank0-step2.pt")
    return folders, state


def test_exact_comparison_and_required_rank_coverage(snapshots):
    folders, _ = snapshots
    assert compare_training_states(*folders, world_size=1, step=2) == 4
    with pytest.raises(FileNotFoundError):
        compare_training_states(*folders, world_size=2, step=2)


@pytest.mark.parametrize("change", ["value", "dtype", "nan", "empty", "missing"])
def test_rejects_incorrect_or_vacuous_snapshot(snapshots, change):
    folders, state = snapshots
    state = copy.deepcopy(state)
    if change == "value":
        state["logits"][0, 0] += 0.01
    elif change == "dtype":
        state["logits"] = state["logits"].double()
    elif change == "nan":
        state["logits"][0, 0] = float("nan")
    elif change == "empty":
        state["candidate"] = {}
    else:
        del state["reference"]["weight"]["gradient"]
    torch.save(state, folders[1] / "rank0-step2.pt")
    with pytest.raises((AssertionError, ValueError)):
        compare_training_states(*folders, world_size=1, step=2)
