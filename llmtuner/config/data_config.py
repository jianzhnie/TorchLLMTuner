"""Dataloader config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.errors import ConfigError


@dataclass(kw_only=True)
class DataloaderConfig:
    """Where the micro-batches come from."""

    dataset: str = field(
        default="random",
        metadata={
            "help": "Corpus selector: 'random' (synthetic, no assets) | "
            "'local_jsonl' | 'local_jsonl_sft' | a key of "
            "datasets.text.recipes.DATASETS | a key of "
            "datasets.multimodal.recipes.MULTIMODAL_DATASETS (needs torchvision)"
        },
    )
    tokenizer_path: str | None = field(
        default=None,
        metadata={
            "help": "Directory holding the tokenizer. Required unless --dataset random."
        },
    )
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Corpus path. Required for local_jsonl and local_jsonl_sft."},
    )
    data_identity: str | None = field(
        default=None,
        metadata={
            "help": "Stable source/recipe snapshot identifier for corpus resume. "
            "For registered or remote data, change it when source contents or "
            "recipe semantics change; local text also retains content hashing."
        },
    )
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "Prompt field used by dataset=local_jsonl_sft."},
    )
    response_field: str = field(
        default="response",
        metadata={"help": "Assistant field used by dataset=local_jsonl_sft."},
    )
    chat_renderer: str | None = field(
        default=None,
        metadata={
            "help": "Optional renderers config class for multi-turn SFT; requires a "
            "compatible SFT recipe."
        },
    )
    messages_field: str = field(
        default="messages",
        metadata={
            "help": "Row field holding the multi-turn conversation. Used only "
            "when chat_renderer is set."
        },
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Shuffle globally before map-data DP sharding; "
            "shuffle streaming rows within each rank."
        },
    )
    streaming_shuffle_buffer_size: int = field(
        default=1_000,
        metadata={"help": "Streaming rows retained per rank for approximate shuffle"},
    )
    num_prefetch_batches: int = field(
        default=2,
        metadata={"help": "Collated batches queued per rank for the trainer"},
    )
    packing: Literal["concat_then_split", "first_fit"] = field(
        default="concat_then_split",
        metadata={
            "help": "Text packing: concat_then_split chunks concatenated documents; "
            "first_fit packs documents into rows."
        },
    )
    num_packing_bins: int = field(
        default=8,
        metadata={
            "help": "Candidate rows 'first_fit' keeps open. More bins can "
            "reduce padding, but buffer more documents. Also applies to "
            "multimodal packing; ignored for text concat_then_split."
        },
    )
    max_num_documents: int | None = field(
        default=None,
        metadata={
            "help": "Cap on documents packed into one row. None leaves the "
            "frontier unconstrained."
        },
    )
    mm_image_token: str = field(
        default="<|image_pad|>",
        metadata={"help": "Image placeholder token. Multimodal recipes only."},
    )
    mm_video_token: str = field(
        default="<|video_pad|>",
        metadata={"help": "Video placeholder token. Multimodal recipes only."},
    )
    mm_vision_start_token: str = field(
        default="<|vision_start|>",
        metadata={
            "help": "Token opening a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_vision_end_token: str = field(
        default="<|vision_end|>",
        metadata={
            "help": "Token closing a vision placeholder run. Multimodal recipes only."
        },
    )
    mm_pad_token: str = field(
        default="<|endoftext|>",
        metadata={"help": "Padding token. Multimodal recipes only."},
    )

    def attention_mask_type(
        self, *, dataset: str | None = None
    ) -> Literal["causal", "block_causal"]:
        """Choose document isolation for this layout and optional corpus override."""
        corpus = self.dataset if dataset is None else dataset
        return (
            "causal"
            if corpus == "random" or self.max_num_documents == 1
            else "block_causal"
        )

    def __post_init__(self) -> None:
        if self.data_identity is not None and (
            not isinstance(self.data_identity, str) or not self.data_identity.strip()
        ):
            raise ConfigError(
                "dataloader.data_identity must be a nonempty string or None"
            )
        if self.dataset == "random" and self.data_identity is not None:
            raise ConfigError(
                "dataloader.data_identity applies only to corpus datasets"
            )
        if not isinstance(self.shuffle, bool):
            raise ConfigError(f"dataloader.shuffle must be bool, got {self.shuffle!r}")
        if self.packing not in ("concat_then_split", "first_fit"):
            raise ConfigError(
                "dataloader.packing must be 'concat_then_split' or 'first_fit', "
                f"got {self.packing!r}"
            )
        if self.dataset != "random" and not self.tokenizer_path:
            raise ConfigError(
                f"dataloader.tokenizer_path is required for dataset "
                f"{self.dataset!r}; only 'random' runs without a tokenizer."
            )
        if self.dataset in {"local_jsonl", "local_jsonl_sft"} and not self.dataset_path:
            raise ConfigError(
                f"dataloader.dataset_path is required for dataset {self.dataset!r}"
            )
        if (
            self.dataset == "local_jsonl_sft"
            and self.chat_renderer is None
            and any(
                not isinstance(value, str) or not value.strip()
                for value in (self.prompt_field, self.response_field)
            )
        ):
            raise ConfigError(
                "dataloader.prompt_field/response_field cannot be empty for "
                "dataset='local_jsonl_sft'"
            )
        if self.chat_renderer is not None:
            if self.dataset != "local_jsonl_sft":
                raise ConfigError(
                    f"dataloader.chat_renderer requires "
                    f"dataset='local_jsonl_sft', got {self.dataset!r}"
                )
            if (
                not isinstance(self.messages_field, str)
                or not self.messages_field.strip()
            ):
                raise ConfigError(
                    "dataloader.messages_field cannot be empty when "
                    "chat_renderer is set"
                )
        # Membership in ``datasets.text.recipes.DATASETS`` /
        # ``datasets.multimodal.recipes.MULTIMODAL_DATASETS`` is checked by
        # ``datasets/loader_builder.py`` at build time, not here: reading the registries
        # would import the datasets package into the config layer.
        if self.max_num_documents is not None:
            require_integer_at_least(self, "max_num_documents", group="dataloader")
        require_integer_at_least(
            self,
            "streaming_shuffle_buffer_size",
            "num_prefetch_batches",
            "num_packing_bins",
            group="dataloader",
        )
        # num_packing_bins is validated here even though only 'first_fit' reads
        # it: the field is always parsed, so a bad value would otherwise be
        # accepted silently under the default recipe and only fail after
        # switching to first_fit.
