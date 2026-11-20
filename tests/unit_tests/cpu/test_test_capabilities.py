"""Dependency gates distinguish cached modules from blocked imports."""

import sys
from types import ModuleType

import pytest

from tests import caps


@pytest.mark.parametrize("cached", [None, ModuleType("test_cached_dependency")])
def test_importable_respects_cached_module_value(monkeypatch, cached):
    monkeypatch.setitem(sys.modules, "test_cached_dependency", cached)
    assert caps.importable("test_cached_dependency") is (cached is not None)


@pytest.mark.parametrize("torch_available", [False, True])
@pytest.mark.parametrize("spmd_available", [False, True])
def test_spmd_probe_checks_torch_first(monkeypatch, torch_available, spmd_available):
    calls = []

    def probe(name):
        calls.append(name)
        return {"torch": torch_available, "spmd_types": spmd_available}[name]

    monkeypatch.setattr(caps, "importable", probe)
    assert caps.CAPS["spmd_types"]() is (torch_available and spmd_available)
    assert calls == (["torch", "spmd_types"] if torch_available else ["torch"])
