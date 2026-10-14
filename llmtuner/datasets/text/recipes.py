"""Text dataset recipes: source selection, processors and sample filters."""

from __future__ import annotations

from functools import partial
from typing import Any

from ..composition import SingleDataset, is_not_none
from ..contracts import SampleProcessorFactory
from ..sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)
from .processors import ChatProcessor, TextProcessor

__all__ = [
    "DATASETS",
    "build_local_jsonl",
    "build_local_jsonl_sft",
    "build_local_jsonl_sft_multiturn",
]


def _build_local_jsonl_recipe(
    *, path: str, processor: SampleProcessorFactory
) -> SingleDataset:
    """One indexed JSONL file, read through ``processor``.

    The three ``local_jsonl*`` factories differ only in the processor: same
    source, same "a processor drops a row by returning ``None``" filter. Built
    here so a new variant cannot forget either half.
    """
    return SingleDataset(
        source=IndexedJsonlSource(patterns=(path,)),
        processor=processor,
        post_filters=(is_not_none,),
    )


def build_local_jsonl(*, path: str) -> SingleDataset:
    """Build the ``local_jsonl`` recipe over a caller-supplied corpus.

    A function rather than an entry in :data:`DATASETS`: the path is a runtime
    argument, so it cannot live in a module-level dict without a global.
    """
    return _build_local_jsonl_recipe(path=path, processor=TextProcessor)


def build_local_jsonl_sft(
    *, path: str, prompt_field: str, response_field: str
) -> SingleDataset:
    """Build a single-turn supervised-chat recipe from a local JSONL file."""

    return _build_local_jsonl_recipe(
        path=path,
        processor=partial(
            ChatProcessor,
            messages_fn=partial(
                _read_prompt_response,
                prompt_field=prompt_field,
                response_field=response_field,
            ),
        ),
    )


def build_local_jsonl_sft_multiturn(
    *, path: str, messages_field: str, renderer: Any
) -> SingleDataset:
    """Build a multi-turn supervised-chat recipe from a local JSONL file.

    Each row's ``messages_field`` holds the conversation as a list of
    ``{"role": ..., "content": ...}`` messages ending with an assistant
    turn. ``renderer`` is a built ``renderers`` renderer (see
    ``datasets/text/renderer.build_chat_renderer``); it owns tokenization and
    the per-turn loss mask.
    """

    return _build_local_jsonl_recipe(
        path=path,
        processor=partial(
            ChatProcessor,
            messages_fn=partial(_read_conversation, messages_field=messages_field),
            renderer=renderer,
        ),
    )


def _read_prompt_response(
    sample: dict[str, Any], *, prompt_field: str, response_field: str
) -> list[dict[str, str]]:
    """Convert configured single-turn fields to chat messages."""
    try:
        prompt = sample[prompt_field]
        response = sample[response_field]
    except KeyError as exc:
        raise KeyError(
            f"local_jsonl_sft row lacks configured field {exc.args[0]!r}"
        ) from exc
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("local_jsonl_sft prompt and response fields must be strings")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _read_conversation(
    sample: dict[str, Any], *, messages_field: str
) -> list[dict[str, str]]:
    """Read a configured multi-turn conversation field."""
    try:
        conversation = sample[messages_field]
    except KeyError as exc:
        raise KeyError(
            f"local_jsonl_sft row lacks configured messages field {exc.args[0]!r}"
        ) from exc
    if not isinstance(conversation, list) or not all(
        isinstance(message, dict) and "role" in message for message in conversation
    ):
        raise TypeError(
            "local_jsonl_sft messages field must be a list of "
            "message dicts with a 'role' key"
        )
    return conversation


DATASETS: dict[str, SingleDataset] = {
    "c4": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="allenai/c4",
            name="en",
            split="train",
        ),
        processor=TextProcessor,
        post_filters=(is_not_none,),
    ),
    "c4_test": SingleDataset(
        source=HuggingFaceRandomAccessSource(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/c4_test/data.json",
            },
        ),
        processor=TextProcessor,
        post_filters=(is_not_none,),
    ),
    "c4_validation": SingleDataset(
        source=HuggingFaceStreamingSource(
            path="allenai/c4",
            name="en",
            split="validation",
        ),
        processor=TextProcessor,
        post_filters=(is_not_none,),
    ),
}
