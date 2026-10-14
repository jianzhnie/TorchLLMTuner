"""Tokenizer interfaces with implementations loaded only when requested."""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "BaseTokenizer": "base",
    "HuggingFaceTokenizer": "huggingface",
    "MultimodalTokenizer": "multimodal",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
