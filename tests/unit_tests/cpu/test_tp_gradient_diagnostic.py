"""Pre-clipping observations must preserve the gradient used by Trainer."""

# requires-env: dtensor

from types import SimpleNamespace

import pytest
import torch

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from tests.integration_tests import npu_tp_gradient_diagnostic as diagnostic
from tests.integration_tests.npu_tp_gradient_diagnostic import local_gradient_records


def test_gradient_observation_preserves_storage_values_and_dtype():
    model = torch.nn.Linear(2, 1, dtype=torch.bfloat16)
    model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    before = model.weight.grad.clone()
    pointer = model.weight.grad.data_ptr()
    records = local_gradient_records([model], {id(model.weight)})
    assert len(records) == 1  # Bias has no gradient and must not be fabricated.
    assert records[0]["local_l2_norm"] == 5.0
    assert records[0]["tp_sharded"] is True
    assert model.weight.grad.data_ptr() == pointer
    assert model.weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(model.weight.grad, before, rtol=0, atol=0)


def test_no_gradients_requires_no_collective_or_host_transfer():
    assert local_gradient_records([torch.nn.Linear(2, 1)], set()) == []


@pytest.mark.parametrize("offload", [False, True])
def test_entry_validates_nested_fsdp_configuration(monkeypatch, offload):
    config = SimpleNamespace(
        parallel=ParallelConfig(fsdp_config=FSDPConfig(fsdp_cpu_offload=offload))
    )
    trained = []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(torch, "set_default_dtype", lambda dtype: None)
    monkeypatch.setattr(diagnostic, "qwen3_30b_npu_config", lambda: config)
    monkeypatch.setattr(
        diagnostic,
        "GradientDiagnosticTrainer",
        lambda cfg: SimpleNamespace(train=lambda: trained.append(cfg)),
    )
    monkeypatch.setenv("LLMTUNER_DUMP_FOLDER", "/tmp/unused-gradient-test")
    if offload:
        with pytest.raises(ValueError, match="CPU-offloaded gradients"):
            diagnostic.main()
        assert trained == []
    else:
        diagnostic.main()
        assert trained == [config]
