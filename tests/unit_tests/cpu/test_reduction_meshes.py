"""Train and validation mesh ownership, without constructing process groups."""

from itertools import product
from unittest.mock import Mock

import pytest

from llmtuner.trainer.runtime.reductions import (
    resolve_loss_reduction_meshes,
    resolve_training_reduction_meshes,
)


class Topology:
    def __init__(self, *, dp, cp, tp, pp):
        self.cp_enabled = cp
        self.tp_enabled = tp
        self.dp_cp_enabled = dp or cp
        self.meshes = {
            "dp": object() if dp else None,
            "pp": object() if pp else None,
            "loss": object() if dp or cp or tp else None,
            "cp": object() if cp else None,
            "tp": object() if tp else None,
            ("cp", "tp"): Mock() if cp and tp else None,
        }

    def get_optional_mesh(self, axes):
        if isinstance(axes, list):
            axes = axes[0] if len(axes) == 1 else tuple(axes)
        return self.meshes[axes]


@pytest.mark.parametrize("dp,cp,tp,pp", product([False, True], repeat=4))
def test_training_and_validation_choose_the_same_loss_meshes(dp, cp, tp, pp):
    topology = Topology(dp=dp, cp=cp, tp=tp, pp=pp)
    meshes = resolve_training_reduction_meshes(topology)
    token_mesh, loss_mesh = resolve_loss_reduction_meshes(topology)

    assert meshes.dp_mesh is token_mesh is topology.meshes["dp"]
    assert meshes.loss_mesh is loss_mesh is topology.meshes["loss"]
    assert meshes.pp_mesh is topology.meshes["pp"]
    axes = ("cp", "tp") if cp and tp else "cp" if cp else "tp" if tp else None
    if cp and tp:
        topology.meshes[axes]._flatten.assert_called_once_with("sequence_loss")
        assert meshes.sequence_mesh is topology.meshes[axes]._flatten.return_value
    else:
        assert meshes.sequence_mesh is (None if axes is None else topology.meshes[axes])


def test_single_process_requires_no_meshes():
    assert resolve_loss_reduction_meshes(None) == (None, None)
    meshes = resolve_training_reduction_meshes(None)
    assert (meshes.dp_mesh, meshes.pp_mesh, meshes.loss_mesh, meshes.sequence_mesh) == (
        None,
    ) * 4
