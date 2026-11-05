"""The trainer-facing seam for multimodal recipes: ``build_dataloader``.

``test_multimodal_data`` exercises the multimodal subtree directly; this file
checks the one path that did not exist until the subtree was wired into the
factory -- naming a multimodal recipe in ``DataloaderConfig.dataset`` and
getting a working loader back, plus the two failure modes at that seam
(unknown name, missing optional dependencies).

The corpus and tokenizer are local fakes, so nothing here touches the network.
Tests that need the real multimodal stack skip when torchvision is absent --
which is also the state the missing-dependency test simulates.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("grain")


import base64
import json
import os
import pathlib
import subprocess
import sys
from functools import partial

import pytest

from llmtuner.config import (
    DataloaderConfig,
    LLMTunerConfig,
    ModelConfig,
    TrainingConfig,
)
from llmtuner.datasets import (
    SingleDataset,
    build_dataloader,
    corpus_identity,
    loader_builder,
)
from llmtuner.datasets.text.recipes import DATASETS as TEXT_DATASETS
from tests.data_fixtures import VOCAB, Base64JsonlSource, png_bytes, write_tokenizer

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
PAD_TOKEN = "[PAD]"

MM_TOKENS = (IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END, PAD_TOKEN)


def _mm_tokenizer_path(tmp_path) -> str:
    return write_tokenizer(
        str(tmp_path / "mm_tokenizer"),
        extra_vocab={
            token: max(VOCAB.values()) + 1 + i for i, token in enumerate(MM_TOKENS)
        },
    )


def _build(config: DataloaderConfig, *, max_context_length=256, num_tokens=64):
    return build_dataloader(
        LLMTunerConfig(
            model=ModelConfig(vocab_size=128),
            training=TrainingConfig(
                global_batch_size=1,
                max_seq_len=max_context_length,
                seed=1,
                dataloader_config=config,
            ),
        ),
        dp_rank=0,
        dp_world_size=1,
        num_tokens_per_batch=num_tokens,
    )


def test_build_dataloader_names_a_multimodal_recipe(tmp_path, monkeypatch):
    """The end-to-end seam: a config names ``cc12m-test`` and the loader that
    comes back carries image patches and grids in its batches."""
    pytest.importorskip("torchvision")
    from llmtuner.datasets.multimodal import packing
    from llmtuner.datasets.multimodal.processors import (
        MultimodalProcessor,
        process_cc12m_sample,
    )
    from llmtuner.datasets.multimodal.recipes import MULTIMODAL_DATASETS

    seen = {}
    original_pack = packing.grain.experimental.FirstFitPackIterDataset

    def capture_pack(*args, **kwargs):
        seen["max_sequences_per_bin"] = kwargs.get("max_sequences_per_bin")
        return original_pack(*args, **kwargs)

    monkeypatch.setattr(
        packing.grain.experimental, "FirstFitPackIterDataset", capture_pack
    )
    original_build = packing.build_multimodal_packing

    def capture_packing(*args, **kwargs):
        seen.update(kwargs)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(loader_builder, "build_multimodal_packing", capture_packing)

    corpus = str(tmp_path / "pairs.jsonl")
    encoded = base64.b64encode(png_bytes(64, 64)).decode()
    with open(corpus, "w") as handle:
        for i in range(8):
            handle.write(
                json.dumps({"txt": f"w{i} hello", "jpg": {"bytes": encoded}}) + "\n"
            )

    # Point the registered recipe at the local corpus: the recipe key is what
    # is under test here, not the Hub snapshot it normally reads.
    monkeypatch.setitem(
        MULTIMODAL_DATASETS,
        "cc12m-test",
        SingleDataset(
            source=Base64JsonlSource(patterns=(corpus,)),
            processor=partial(MultimodalProcessor, process_sample=process_cc12m_sample),
            post_filters=(lambda sample: sample is not None,),
        ),
    )

    loader = _build(
        DataloaderConfig(
            dataset="cc12m-test",
            tokenizer_path=_mm_tokenizer_path(tmp_path),
            mm_pad_token=PAD_TOKEN,
            num_packing_bins=3,
            max_num_documents=2,
        ),
        num_tokens=256,
    )
    assert seen["num_packing_bins"] == 3
    assert seen["max_sequences_per_bin"] == 2
    batch = next(iter(loader))
    assert batch["input"].shape == (256,)
    assert batch["labels"].shape == (256,)
    # The media made the trip from jsonl row to collated patches: one patch
    # sequence per image, each grid entry counting its own length.
    assert batch["pixel_values"] is not None
    assert batch["grid_thw"] is not None
    assert batch["pixel_values"].shape[0] == int(batch["grid_thw"].prod(-1).sum())
    loader.close()


def test_unknown_recipe_lists_both_registries(tmp_path):
    """A typo must show the caller every name that would have worked, from
    both catalogs, in one error."""
    pytest.importorskip("torchvision")
    from llmtuner.datasets.multimodal.recipes import MULTIMODAL_DATASETS

    with pytest.raises(ValueError, match="unknown dataset") as excinfo:
        _build(
            DataloaderConfig(
                dataset="not-a-recipe",
                tokenizer_path=_mm_tokenizer_path(tmp_path),
            )
        )
    message = str(excinfo.value)
    for key in sorted(TEXT_DATASETS) + sorted(MULTIMODAL_DATASETS):
        assert key in message


def test_missing_multimodal_dependencies_raise_with_install_guidance(
    tmp_path, monkeypatch
):
    """Missing media dependencies must report the installation guidance."""
    monkeypatch.setattr(
        loader_builder,
        "MULTIMODAL_IMPORT_ERROR",
        ModuleNotFoundError("torchvision"),
    )

    with pytest.raises(ImportError, match="pip install torchvision"):
        _build(
            DataloaderConfig(
                dataset="obelics",
                tokenizer_path=_mm_tokenizer_path(tmp_path),
            )
        )


def test_text_loader_works_when_media_dependencies_are_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        loader_builder, "MULTIMODAL_IMPORT_ERROR", ModuleNotFoundError("torchvision")
    )

    corpus = str(tmp_path / "rows.jsonl")
    with open(corpus, "w") as handle:
        handle.write(json.dumps({"text": "lorem ipsum"}) + "\n")
    loader = _build(
        DataloaderConfig(
            dataset="local_jsonl",
            tokenizer_path=str(write_tokenizer(str(tmp_path / "tokenizer"))),
            dataset_path=corpus,
        ),
        max_context_length=8,
        num_tokens=32,
    )
    loader.close()
    assert isinstance(loader.state_dict(), dict)


def test_importing_the_text_path_needs_no_multimodal_dependencies():
    """Text imports succeed even if optional media dependencies are absent."""
    source = (
        "import sys; sys.modules['torchvision'] = None\n"
        "from llmtuner.datasets import loader_builder\n"
        "assert isinstance(loader_builder.MULTIMODAL_IMPORT_ERROR, ImportError)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(__file__).parents[3])
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_base_tokenizer_import_does_not_load_optional_implementations():
    source = (
        "import sys; sys.modules['tokenizers'] = None\n"
        "from llmtuner.components.tokenizer import BaseTokenizer\n"
        "from llmtuner.components.tokenizer.base import BaseTokenizer as Defined\n"
        "assert BaseTokenizer is Defined\n"
        "assert 'llmtuner.components.tokenizer.huggingface' not in sys.modules\n"
        "assert 'llmtuner.components.tokenizer.multimodal' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("historical_identity", [False, True])
def test_multimodal_factory_restores_packed_images_and_rejects_old_identity(
    tmp_path, monkeypatch, historical_identity
):
    from copy import deepcopy

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from llmtuner.datasets.multimodal.processors import (
        MultimodalProcessor,
        process_cc12m_sample,
    )
    from llmtuner.datasets.multimodal.recipes import MULTIMODAL_DATASETS
    from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal

    corpus = tmp_path / "pairs.jsonl"
    encoded = base64.b64encode(png_bytes(64, 64)).decode()
    corpus.write_text(
        "".join(
            json.dumps({"txt": f"w{i} hello", "jpg": {"bytes": encoded}}) + "\n"
            for i in range(12)
        )
    )
    monkeypatch.setitem(
        MULTIMODAL_DATASETS,
        "cc12m-cursor-test",
        SingleDataset(
            source=Base64JsonlSource(patterns=(str(corpus),)),
            processor=partial(MultimodalProcessor, process_sample=process_cc12m_sample),
            post_filters=(lambda sample: sample is not None,),
        ),
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    vocab = {"[UNK]": 0, "hello": 1}
    vocab.update({f"w{i}": i + 2 for i in range(12)})
    vocab.update({token: i + 14 for i, token in enumerate(MM_TOKENS)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tokenizer_path / "tokenizer.json"))
    config = DataloaderConfig(
        dataset="cc12m-cursor-test",
        tokenizer_path=str(tokenizer_path),
        mm_pad_token=PAD_TOKEN,
        data_identity="immutable-local-png-snapshot",
        num_packing_bins=2,
        max_num_documents=2,
    )
    dumps = corpus_identity.json.dumps

    def legacy_settings(settings, **kwargs):
        settings = settings.copy()
        settings.pop("multimodal_encoding", None)
        return dumps(settings, **kwargs)

    loaders = []
    try:
        with monkeypatch.context() as context:
            if historical_identity:
                context.setattr(corpus_identity.json, "dumps", legacy_settings)
            source = _build(config, num_tokens=128)
            loaders.append(source)
        reference = _build(config, num_tokens=128)
        target = _build(config, num_tokens=128)
        loaders.extend((reference, target))
        for _ in range(2):
            assert_state_equal(next(iter(source)), next(iter(reference)))
        saved = deepcopy(source.state_dict())
        assert saved["batch_layout"]["data_identity"]
        before = deepcopy(target.state_dict())
        if historical_identity:
            with pytest.raises(ValueError, match="batch layout"):
                target.load_state_dict(saved)
            assert_state_equal(target.state_dict(), before)
            # Rejection leaves the actual media iterator at its original cursor.
            oracle = _build(config, num_tokens=128)
            loaders.append(oracle)
            assert_state_equal(next(iter(target)), next(iter(oracle)))
        else:
            target.load_state_dict(saved)
            assert_state_equal(target.state_dict(), saved)
            for _ in range(3):
                actual, expected = next(iter(target)), next(iter(reference))
                assert_state_equal(actual, expected)
                assert actual["pixel_values"].numel() > 0
                assert actual["grid_thw"].numel() > 0
                assert torch.any(actual["input"] == vocab[IMAGE_TOKEN])
                assert not torch.any(actual["labels"] == vocab[IMAGE_TOKEN])
    finally:
        for loader in loaders:
            loader.close()
