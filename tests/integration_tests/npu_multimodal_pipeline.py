"""Local PNG packing, loader resume and NPU tensor transfer with real processors.

PYTHONPATH=. python tests/integration_tests/npu_multimodal_pipeline.py
Append --obelics for two-image interleaved rows and a 256-token batch.

Uses generated PNG/JSONL and a local tokenizer; no vision model is exercised.
The fixture source only decodes base64 bytes from JSONL, not network storage.
"""

import argparse
import base64
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from llmtuner.accelerator.tensor_tree import move_tensor_tree
from llmtuner.config import (
    DataloaderConfig,
    LLMTunerConfig,
    ModelConfig,
    TrainingConfig,
)
from llmtuner.datasets.composition import SingleDataset
from llmtuner.datasets.loader_builder import build_dataloader
from llmtuner.datasets.multimodal.processors import (
    MultimodalProcessor,
    process_cc12m_sample,
    process_obelics_sample,
)
from llmtuner.datasets.multimodal.recipes import MULTIMODAL_DATASETS
from tests.data_fixtures import VOCAB, Base64JsonlSource, png_bytes, write_tokenizer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal


class InterleavedJsonlSource(Base64JsonlSource):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        return {
            "texts": [row["txt"], None, "hello", None, "world"],
            "images": [None, row["jpg"], None, row["jpg"], None],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obelics", action="store_true")
    args = parser.parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This test requires an available NPU")
    recipe = (
        "local-interleaved-npu-equivalence"
        if args.obelics
        else "local-png-npu-equivalence"
    )
    assert recipe not in MULTIMODAL_DATASETS
    loaders = []
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "images.jsonl"
        encoded = base64.b64encode(png_bytes(64, 64)).decode()
        corpus.write_text(
            "".join(
                json.dumps({"txt": f"w{index} hello", "jpg": {"bytes": encoded}}) + "\n"
                for index in range(16)
            )
        )
        tokens = (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
        tokenizer_path = write_tokenizer(
            str(root / "tokenizer"),
            extra_vocab={
                token: max(VOCAB.values()) + 1 + index
                for index, token in enumerate(tokens)
            },
        )
        MULTIMODAL_DATASETS[recipe] = SingleDataset(
            source=(InterleavedJsonlSource if args.obelics else Base64JsonlSource)(
                patterns=(str(corpus),)
            ),
            processor=partial(
                MultimodalProcessor,
                process_sample=process_obelics_sample
                if args.obelics
                else process_cc12m_sample,
            ),
            post_filters=(lambda sample: sample is not None,),
        )
        try:
            config = LLMTunerConfig(
                model=ModelConfig(vocab_size=128),
                training=TrainingConfig(
                    global_batch_size=1,
                    max_seq_len=256,
                    seed=1,
                    dataloader_config=DataloaderConfig(
                        dataset=recipe,
                        tokenizer_path=tokenizer_path,
                        mm_pad_token="[PAD]",
                        data_identity="generated-png-npu-v1",
                        num_packing_bins=2,
                        max_num_documents=2,
                    ),
                ),
            )
            for _ in range(2):
                loaders.append(
                    build_dataloader(
                        config,
                        dp_rank=0,
                        dp_world_size=1,
                        num_tokens_per_batch=256 if args.obelics else 128,
                    )
                )
            source, restored = loaders
            iterator = iter(source)
            next(iterator)
            saved = deepcopy(source.state_dict())
            expected = next(iterator)
            restored.load_state_dict(saved)
            actual = next(iter(restored))
            assert_state_equal(actual, expected)
            for name in ("pixel_values", "grid_thw"):
                assert isinstance(actual[name], torch.Tensor) and actual[name].numel()
            if args.obelics:
                assert actual["grid_thw"].shape[0] >= 2
                assert actual["grid_thw"].shape[0] % 2 == 0
            image_token_id = max(VOCAB.values()) + 1
            image_tokens = actual["input"] == image_token_id
            expected_image_tokens = actual["grid_thw"].prod(dim=-1).sum() // 4
            assert image_tokens.sum() == expected_image_tokens
            assert (actual["labels"][image_tokens] == -100).all()
            assert (
                actual["pixel_values"].shape[0] == actual["grid_thw"].prod(dim=-1).sum()
            )
            tensors = {
                name: value
                for name, value in actual.items()
                if isinstance(value, torch.Tensor)
            }
            transferred = move_tensor_tree(tensors, torch.device("npu:0"))
            for name, value in transferred.items():
                assert value.device.type == "npu"
                torch.testing.assert_close(value.cpu(), tensors[name], rtol=0, atol=0)
                if value.is_floating_point():
                    assert torch.isfinite(value).all()
            print(
                f"PASS: {'OBELICS interleaved' if args.obelics else 'CC12M paired'} "
                "PNG decode, packed images, exact loader resume and NPU transfer"
            )
        finally:
            for loader in loaders:
                loader.close()
            del MULTIMODAL_DATASETS[recipe]


if __name__ == "__main__":
    main()
