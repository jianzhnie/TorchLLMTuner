"""Train and validation mesh ownership, without constructing process groups."""

from itertools import product

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
            ("cp", "tp"): object() if cp and tp else None,
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

    assert meshes.dp is token_mesh is topology.meshes["dp"]
    assert meshes.loss is loss_mesh is topology.meshes["loss"]
    assert meshes.pp is topology.meshes["pp"]
    axes = ("cp", "tp") if cp and tp else "cp" if cp else "tp" if tp else None
    assert meshes.sequence is (None if axes is None else topology.meshes[axes])


def test_single_process_requires_no_meshes():
    assert resolve_loss_reduction_meshes(None) == (None, None)
    meshes = resolve_training_reduction_meshes(None)
    assert (meshes.dp, meshes.pp, meshes.loss, meshes.sequence) == (None,) * 4
