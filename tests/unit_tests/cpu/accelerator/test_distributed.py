from types import SimpleNamespace

from llmtuner.accelerator import distributed


def test_default_group_uses_public_world_handle(monkeypatch):
    world = object()
    monkeypatch.setattr(distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(distributed.torch_dist, "group", SimpleNamespace(WORLD=world))

    assert distributed.get_default_group() is world


def test_default_group_is_none_before_initialization(monkeypatch):
    monkeypatch.setattr(distributed, "is_distributed", lambda: False)

    assert distributed.get_default_group() is None
