"""Device selection contracts that do not require accelerator hardware."""

from __future__ import annotations

import torch

from llmtuner.accelerator import device


def test_device_priority_prefers_npu(monkeypatch) -> None:
    available = {"npu", "cuda"}
    monkeypatch.setattr(
        device, "is_device_type_available", lambda kind: kind in available
    )

    assert device.get_device_type() == "npu"


def test_current_device_uses_local_rank(monkeypatch) -> None:
    monkeypatch.setattr(device, "device_type", "cuda")
    monkeypatch.setenv("LOCAL_RANK", "3")

    assert device.get_current_device() == torch.device("cuda:3")
    assert device.get_current_device(use_cpu=True) == torch.device("cpu")


def test_distributed_backend_follows_device_type(monkeypatch) -> None:
    monkeypatch.setattr(device, "device_type", "npu")
    assert device.get_distributed_backend() == "hccl"

    monkeypatch.setattr(device, "device_type", "cpu")
    assert device.get_distributed_backend() == "gloo"


def test_cpu_is_available_but_not_pinned() -> None:
    cpu = torch.device("cpu")
    assert device.is_device_available(cpu)
    assert not device.should_use_pin_memory(cpu)


def test_cuda_peak_query_resets_the_requested_device(monkeypatch) -> None:
    requested_device = torch.device("cuda:3")
    calls = []

    def peak(*, device):
        calls.append(("query", device))
        return 7 * 1024 * 1024 + 512

    def reset(*, device):
        calls.append(("reset", device))

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)

    assert device.get_max_cuda_memory(requested_device) == 7
    assert calls == [("query", requested_device), ("reset", requested_device)]
