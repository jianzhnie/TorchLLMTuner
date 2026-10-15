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
)
from llmtuner.datasets.text.recipes import DATASETS as TEXT_DATASETS
from tests.data_fixtures import VOCAB, Base64JsonlSource, png_bytes, write_tokenizer

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
PAD_TOKEN = "[PAD]"

MM_TOKENS = (IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END, PAD_TOKEN)

# Every module the multimodal path pulls in, the subpackage itself included.
# The names are listed rather than matched with a ``startswith`` on the
# subpackage prefix so that renaming a module shows up here as a failure
# instead of as an assertion that silently stops matching anything.
MM_MODULES = (
    "llmtuner.datasets.multimodal",
    "llmtuner.datasets.multimodal.collator",
    "llmtuner.datasets.multimodal.processors",
    "llmtuner.datasets.multimodal.recipes",
    "llmtuner.datasets.multimodal.packing",
    "llmtuner.datasets.multimodal.image",
    "llmtuner.datasets.multimodal.placeholders",
    "llmtuner.datasets.multimodal.video",
)

# The text recipe module, which the text path *does* import. Listed for the
# same reason: it keeps the guard meaningful across a rename of the file, which
# would otherwise leave a stale string matching nothing.
TEXT_RECIPE_MODULE = "llmtuner.datasets.text.recipes"


def _loaded(names: tuple[str, ...]) -> list[str]:
    """The modules from ``names`` currently imported, for the lazy-import tests."""
    return [name for name in sys.modules if name in names]


def _loaded_mm_modules() -> list[str]:
    return _loaded(MM_MODULES)


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
        process_cc12_wd_sample,
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

    monkeypatch.setattr(packing, "build_multimodal_packing", capture_packing)

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
            processor=partial(
                MultimodalProcessor, sample_processor=process_cc12_wd_sample
            ),
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
    """Without torchvision, naming a multimodal recipe must fail with the
    install hint rather than a bare ModuleNotFoundError -- while the text and
    random paths keep working, which is the point of the lazy import."""
    for module in [m for m in sys.modules if m.startswith("torchvision")]:
        monkeypatch.delitem(sys.modules, module)
    for module in _loaded_mm_modules():
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, "torchvision", None)

    with pytest.raises(ImportError, match="pip install torchvision"):
        _build(
            DataloaderConfig(
                dataset="obelics",
                tokenizer_path=_mm_tokenizer_path(tmp_path),
            )
        )


def test_text_path_does_not_import_the_multimodal_trunk(tmp_path, monkeypatch):
    """The lazy-import guarantee, checked in an environment that has
    torchvision: building a text loader must leave the multimodal modules
    out of ``sys.modules``."""
    for module in _loaded_mm_modules():
        monkeypatch.delitem(sys.modules, module)

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
    # The half of the guard that is armed whatever the multimodal path does:
    # the module the text path imports really did load, so a name that stopped
    # resolving cannot pass this test by matching nothing on both sides.
    assert _loaded((TEXT_RECIPE_MODULE,)) == [TEXT_RECIPE_MODULE]
    assert _loaded_mm_modules() == []
    assert isinstance(loader.state_dict(), dict)


def test_importing_the_text_path_needs_no_multimodal_dependencies():
    """The other half of the lazy-import guarantee, in a fresh interpreter:
    the text path must not drag torchvision in even on the *first* import.

    The test above builds a text loader inside this process, so it can only
    see a multimodal import that has already been undone -- deleting the
    module from ``sys.modules`` and watching it not come back. A multimodal
    import hoisted to module scope in ``build.py`` is not undone before the
    loader is built, so it slips past that check and past every other test
    here. It cannot slip past this one, which starts from nothing.
    """
    source = (
        "import sys; sys.modules['torchvision'] = None\n"
        "import llmtuner.datasets, llmtuner.datasets.loader_builder\n"
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
