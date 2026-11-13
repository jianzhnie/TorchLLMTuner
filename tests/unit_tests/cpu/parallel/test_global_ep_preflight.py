"""Invalid expert ownership must fail before TP or PP changes the model."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel import parallelize
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("pipeline", [False, True])
def test_invalid_expert_geometry_leaves_tp_pp_and_model_untouched(
    monkeypatch, pipeline
):
    model = Qwen3MoeForCausalLM(model_config(num_experts=3))
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        expert_parallel_size=2,
        pipeline_parallel_size=2 if pipeline else 1,
    )
    group = SimpleNamespace(size=lambda: 2)
    dims = SimpleNamespace(
        pp_enabled=pipeline,
        get_optional_mesh=lambda name: SimpleNamespace(get_group=lambda: group),
    )
    parameters = tuple(model.parameters())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    forwards = tuple(module.forward for module in model.modules())
    rng = torch.get_rng_state().clone()

    def reject_transform(*args, **kwargs):
        pytest.fail("Invalid EP geometry reached a model transformation")

    monkeypatch.setattr(parallelize, "apply_tp", reject_transform)
    monkeypatch.setattr(parallelize, "apply_pp", reject_transform)
    with pytest.raises(ValueError, match="EP degree 2 does not divide num_experts=3"):
        parallelize.parallelize_model(
            model,
            cfg=cfg,
            mesh=object(),
            parallel_dims=dims,
            global_batch_size=4,
        )

    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert forwards == tuple(module.forward for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert tuple(state) == tuple(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
