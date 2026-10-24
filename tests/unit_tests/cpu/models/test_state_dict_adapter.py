"""HF state conversion preserves tied weights and caller-owned mappings."""

import json
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


@pytest.mark.parametrize("text_tied", [False, True])
def test_composite_checkpoint_uses_text_weight_ties(text_tied):
    from transformers import AutoConfig

    config = AutoConfig.for_model(
        "llava", text_config={"tie_word_embeddings": text_tied}
    )
    config.tie_word_embeddings = not text_tied
    adapter = HFTransformerStateDictAdapter(config, None)
    adapter._hf_keys = {"model.embed_tokens.weight"}
    weight = torch.randn(4, 2)
    source = {"model.embed_tokens.weight": weight}
    imported = adapter.from_hf(source)

    assert adapter.model_config is config.text_config
    assert ("model.lm_head.weight" in imported) == text_tied
    assert source == {"model.embed_tokens.weight": weight}
    if text_tied:
        assert imported["model.lm_head.weight"] is weight
        assert adapter.to_hf(imported) == source
    else:
        imported["model.lm_head.weight"] = weight.clone()
        with pytest.raises(ValueError, match="unexpected=.*lm_head.weight"):
            adapter.to_hf(imported)


@pytest.mark.parametrize("complete", [False, True])
def test_hf_index_checks_global_pipeline_keys(monkeypatch, complete):
    import llmtuner.models.hf.state_dict_adapter as adapter_module

    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=False), None
    )
    adapter._hf_keys = {"model.layers.0.weight", "model.layers.1.weight"}
    local_weight = torch.randn(2, 2)
    local = {"model.model.layers.0.weight": local_weight}
    # This unit test supplies only the key-gather result, not a process group.
    # Real error coordination is exercised by hf_adapter_coordination.py.
    from contextlib import nullcontext

    monkeypatch.setattr(
        adapter_module, "synchronize_local_errors", lambda phase: nullcontext()
    )
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


@pytest.mark.parametrize("stores_head", [False, True])
def test_single_file_checkpoint_loads_tied_head(tmp_path, stores_head):
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    saved = {"model.embed_tokens.weight": weight}
    if stores_head:
        saved["lm_head.weight"] = weight.clone()
    save_file(saved, str(tmp_path / "model.safetensors"))
    adapter = HFTransformerStateDictAdapter(
        SimpleNamespace(tie_word_embeddings=True), str(tmp_path)
    )
    local_weight = torch.zeros_like(weight)
    target = {
        "model.model.embed_tokens.weight": local_weight,
        "model.lm_head.weight": local_weight,
    }

    hf_state = adapter.to_hf(target)
    assert set(hf_state) == set(saved)
    dcp.load(hf_state, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)))
    restored = adapter.from_hf(hf_state)
    torch.testing.assert_close(restored["model.lm_head.weight"], weight)
    torch.testing.assert_close(restored["model.model.embed_tokens.weight"], weight)


def test_hf_key_mapping_rejects_collisions():
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), None)
    with pytest.raises(ValueError, match="key mapping collision"):
        adapter.to_hf({"model.weight": torch.ones(1), "weight": torch.zeros(1)})


def test_hf_reader_selects_each_tensor_from_its_indexed_file(tmp_path):
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    save_file(
        {"a": torch.tensor([1.0]), "b": torch.tensor([999.0])},
        str(tmp_path / "first.safetensors"),
    )
    save_file(
        {"a": torch.tensor([999.0]), "b": torch.tensor([2.0])},
        str(tmp_path / "second.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}
        ),
        encoding="utf-8",
    )
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), None)
    restored = {"a": torch.zeros(1), "b": torch.zeros(1)}
    dcp.load(restored, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)))
    torch.testing.assert_close(
        restored, {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "filenames",
    [
        ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        ("decoder.safetensors", "embedding.safetensors"),
        ("v2-model-00001.safetensors", "v2-model-00002.safetensors"),
        ("model-00003.safetensors", "model-00009.safetensors"),
    ],
)
def test_index_shard_grouping_survives_export_round_trip(tmp_path, filenames):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    source_path = tmp_path / "source"
    source_path.mkdir()
    weight_map = {"a": filenames[0], "b": filenames[1], "c": filenames[0]}
    (source_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), str(source_path))
    mapping = adapter.fqn_to_index_mapping
    assert set(mapping) == set(weight_map)
    assert set(mapping.values()) == {1, 2}
    assert mapping["a"] == mapping["c"] != mapping["b"]

    source = {
        f"model.{key}": torch.tensor([float(i)]) for i, key in enumerate(weight_map)
    }
    output_path = str(tmp_path / "export")
    dcp.save(
        adapter.to_hf(source),
        storage_writer=HuggingFaceStorageWriter(
            output_path, fqn_to_index_mapping=mapping
        ),
    )
    restored = {
        key: torch.zeros_like(value) for key, value in adapter.to_hf(source).items()
    }
    dcp.load(restored, storage_reader=adapter.get_hf_storage_reader(output_path))
    for key, value in adapter.from_hf(restored).items():
        torch.testing.assert_close(value, source[key])


@pytest.mark.parametrize(
    "index",
    [
        [],
        {},
        {"weight_map": {}},
        {"weight_map": []},
        {"weight_map": {"a": None}},
        {"weight_map": {"a": ""}},
        {"weight_map": {"": "model.safetensors"}},
    ],
)
def test_invalid_index_is_rejected_during_adapter_construction(tmp_path, index):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="weight_map"):
        HFTransformerStateDictAdapter(SimpleNamespace(), str(tmp_path))
