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
