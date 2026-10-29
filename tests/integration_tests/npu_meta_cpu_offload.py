"""Real NPU meta materialization keeps unmanaged parameters on the device."""

import torch
from torch import nn

from llmtuner.models.hf.materialization import materialize_meta_model
from llmtuner.trainer.runtime.assembly import _fsdp_managed_parameters


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This test requires an available NPU")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)

    with torch.device("meta"):
        model = nn.Linear(3, 2)
    weight = model.weight
    materialize_meta_model(
        model,
        device,
        parameter_device=torch.device("cpu"),
        offloaded_parameters=_fsdp_managed_parameters(model),
    )
    assert model.weight is weight
    assert model.weight.device == device and model.bias.device == device
    output = model(torch.randn(2, 3, device=device))
    assert output.shape == (2, 2) and output.device == device

    with torch.device("meta"):
        partial = nn.Module()
        partial.managed = nn.Linear(3, 2)
        partial.unmanaged = nn.Linear(3, 2)
    offloaded = set(partial.managed.parameters())
    unmanaged_weight = partial.unmanaged.weight
    materialize_meta_model(
        partial,
        device,
        parameter_device=torch.device("cpu"),
        offloaded_parameters=offloaded,
    )
    assert all(parameter.device.type == "cpu" for parameter in offloaded)
    assert partial.unmanaged.weight is unmanaged_weight
    assert all(
        parameter.device == device for parameter in partial.unmanaged.parameters()
    )
    print("PASS: no-op and partial FSDP meta CPU-offload device layout on NPU")


if __name__ == "__main__":
    main()
