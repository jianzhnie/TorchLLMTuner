"""Multimodal recipes: streaming sources and their sample processors."""

from functools import partial

from ..composition import SingleDataset, is_not_none
from ..sources import HuggingFaceStreamingSource
from .processors import (
    MultimodalProcessor,
    process_cc12_wd_sample,
    process_obelics_sample,
)

__all__ = ["MULTIMODAL_DATASETS"]


MULTIMODAL_DATASETS: dict[str, SingleDataset] = {
    "obelics": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="HuggingFaceM4/OBELICS",
            split="train",
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_obelics_sample,
        ),
        post_filters=(is_not_none,),
    ),
    "cc12m": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="pixparse/cc12m-wds",
            split="train",
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_cc12_wd_sample,
        ),
        post_filters=(is_not_none,),
    ),
    "cc12m-test": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="tests/assets/cc12m_test",
            split="train",
            load_dataset_kwargs={
                "data_files": {"train": "*.tar"},
            },
        ),
        processor=partial(
            MultimodalProcessor,
            sample_processor=process_cc12_wd_sample,
        ),
        post_filters=(is_not_none,),
    ),
}
