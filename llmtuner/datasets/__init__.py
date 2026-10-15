"""Public data contracts, recipes, packing builders, and dataloaders.

build_dataloader consumes LLMTunerConfig plus runtime DP ownership and token
budgets. Text recipes are loaded with the builder; multimodal recipes and media
backends are imported only when a multimodal dataset is requested. Concrete
catalogs live in text/recipes.py and multimodal/recipes.py and are not re-exported.
Exports resolve on first access so importing a standalone helper does not require
Grain or load dataset catalogs. Implementation code imports defining modules.
"""

from typing import Any

from ..utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "HAS_PIN_MEMORY": "collators",
    "Collator": "collators",
    "TextCollator": "collators",
    "DatasetConcat": "composition",
    "DatasetMix": "composition",
    "DatasetRecipe": "composition",
    "SingleDataset": "composition",
    "WeightedDataset": "composition",
    "build_dataset": "composition",
    "BaseDataLoader": "contracts",
    "DataloaderExhaustedError": "contracts",
    "DatasetBuildContext": "contracts",
    "DatasetIterationPolicy": "contracts",
    "GrainDataset": "contracts",
    "SampleProcessor": "contracts",
    "TextSequence": "contracts",
    "TrainerBatch": "contracts",
    "GrainDataLoader": "dataloader",
    "build_dataloader": "loader_builder",
    "build_concat_then_split_packing": "packing.build",
    "build_first_fit_packing": "packing.build",
    "HuggingFaceRandomAccessSource": "sources",
    "HuggingFaceStreamingSource": "sources",
    "IndexedJsonlSource": "sources",
    "RandomAccessDataSource": "sources",
    "build_source": "sources",
}

__all__ = [
    "BaseDataLoader",
    "Collator",
    "DatasetBuildContext",
    "DatasetConcat",
    "DatasetIterationPolicy",
    "DatasetMix",
    "DatasetRecipe",
    "DataloaderExhaustedError",
    "GrainDataLoader",
    "GrainDataset",
    "HAS_PIN_MEMORY",
    "HuggingFaceRandomAccessSource",
    "HuggingFaceStreamingSource",
    "IndexedJsonlSource",
    "RandomAccessDataSource",
    "SampleProcessor",
    "SingleDataset",
    "TextCollator",
    "TextSequence",
    "TrainerBatch",
    "WeightedDataset",
    "build_concat_then_split_packing",
    "build_dataloader",
    "build_dataset",
    "build_first_fit_packing",
    "build_source",
]


def __getattr__(name: str) -> Any:
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
