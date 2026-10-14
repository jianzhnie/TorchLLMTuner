from types import SimpleNamespace

from llmtuner.utils import monitoring


def test_hardware_probes_use_rank_current_device(monkeypatch):
    calls = []

    def name(index):
        calls.append(index)
        return "rank-local device"

    def properties(index):
        calls.append(index)
        return SimpleNamespace(total_memory=8192)

    monkeypatch.setattr(monitoring, "device_type", "cuda")
    monkeypatch.setattr(
        monitoring,
        "device_module",
        SimpleNamespace(
            current_device=lambda: 3,
            get_device_name=name,
            get_device_properties=properties,
        ),
    )
    assert monitoring.get_device_name() == "rank-local device"
    assert monitoring.get_device_capacity_bytes() == 8192
    assert calls == [3, 3]


def test_xpu_peak_flops_uses_resolved_device_module(monkeypatch):
    properties_calls = []

    def properties(index):
        properties_calls.append(index)
        return SimpleNamespace(max_compute_units=10)

    monkeypatch.setattr(monitoring, "device_type", "xpu")
    monkeypatch.setattr(
        monitoring,
        "device_module",
        SimpleNamespace(current_device=lambda: 2, get_device_properties=properties),
    )

    assert monitoring.get_peak_flops("Intel Data Center GPU Max 1550") == 6.656e12
    assert properties_calls == [2]
