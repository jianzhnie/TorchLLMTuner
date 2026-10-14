"""HF state conversion preserves tied weights and caller-owned mappings."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter


@pytest.mark.parametrize("stores_head", [False, True])
def test_tied_head_export_respects_checkpoint_index(stores_head):
    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=True), None
    )
    adapter._hf_keys = {"model.embed_tokens.weight"}
    if stores_head:
        adapter._hf_keys.add("lm_head.weight")
    weight = torch.randn(4, 2)
    state = {
        "model.model.embed_tokens.weight": weight,
        "model.lm_head.weight": weight,
    }
    exported = adapter.to_hf(state)
    assert set(exported) == adapter._hf_keys
    assert set(state) == {"model.model.embed_tokens.weight", "model.lm_head.weight"}


def test_tied_head_import_does_not_mutate_source():
    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=True), None
    )
    weight = torch.randn(4, 2)
    source = {"model.embed_tokens.weight": weight}
    imported = adapter.from_hf(source)
    assert set(source) == {"model.embed_tokens.weight"}
    assert imported["model.lm_head.weight"] is weight
    assert imported["model.model.embed_tokens.weight"] is weight


@pytest.mark.parametrize("complete", [False, True])
def test_hf_index_checks_global_pipeline_keys(monkeypatch, complete):
    import llmtuner.models.hf.state_dict_adapter as adapter_module

    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=False), None
    )
    adapter._hf_keys = {"model.layers.0.weight", "model.layers.1.weight"}
    local_weight = torch.randn(2, 2)
    local = {"model.model.layers.0.weight": local_weight}
    monkeypatch.setattr(adapter_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter_module.dist, "get_world_size", lambda: 2)

    def gather(outputs, keys):
        outputs[0] = keys
        outputs[1] = {"model.layers.1.weight"} if complete else set()

    monkeypatch.setattr(adapter_module.dist, "all_gather_object", gather)
    if complete:
        exported = adapter.to_hf(local)
        assert set(exported) == {"model.layers.0.weight"}
        assert exported["model.layers.0.weight"] is local_weight
    else:
        with pytest.raises(ValueError, match="missing=.*model.layers.1.weight"):
            adapter.to_hf(local)


def test_hf_index_still_rejects_missing_keys_without_distributed():
    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=False), None
    )
    adapter._hf_keys = {"model.layers.0.weight", "model.layers.1.weight"}
    with pytest.raises(ValueError, match="missing=.*model.layers.1.weight"):
        adapter.to_hf({"model.model.layers.0.weight": torch.zeros(2, 2)})
