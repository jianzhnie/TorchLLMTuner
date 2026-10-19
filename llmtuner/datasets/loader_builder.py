"""Assemble dataloaders from run configuration and data-parallel ownership.

Resolve a text or multimodal recipe, build its tokenizer and packing graph,
then construct the loader. Synthetic data bypasses tokenizer and packing setup.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import grain.python as grain

from .collators import CollatorFactory, PipelineTextCollator, TextCollator
from .contracts import (
    BaseDataLoader,
    DatasetBuildContext,
    DatasetIterationPolicy,
    require_positive_integer,
    validate_dp_coordinates,
    validate_loader_iteration,
)
from .dataloader import GrainDataLoader
from .packing.build import (
    build_concat_then_split_packing,
    build_first_fit_packing,
)
from .sources import file_patterns_to_paths
from .synthetic import RandomTokenDataLoader
from .text.recipes import (
    DATASETS,
    build_local_jsonl,
    build_local_jsonl_sft,
    build_local_jsonl_sft_multiturn,
)

if TYPE_CHECKING:
    from ..components.tokenizer.base import BaseTokenizer
    from ..config.data_config import DataloaderConfig
    from ..config.llmtuner_config import LLMTunerConfig
    from .composition import DatasetRecipe

__all__ = ["build_dataloader"]


def corpus_data_identity(config: LLMTunerConfig, *, dataset: str, repeat: bool) -> str:
    """Fingerprint tokenizer assets, processing policy and source identity.

    Local text includes file contents. Other recipes require a caller-declared
    snapshot identifier; this cannot infer remote contents or callable semantics.
    This is a construction-time snapshot, not a file mutation lock.
    """
    data_settings = asdict(config.dataloader)
    # Ranks may stage identical assets in different local directories. DCP
    # shares loader metadata, so paths cannot define content identity.
    for name in ("dataset_path", "tokenizer_path", "num_prefetch_batches"):
        data_settings.pop(name)
    if data_settings["data_identity"] is None:
        # Preserve existing local-text-v1 checkpoint identities.
        data_settings.pop("data_identity")
    settings = {
        "version": 1,
        "dataset": dataset,
        "seed": config.seed,
        "repeat": repeat,
        "data": data_settings,
    }
    if dataset not in {"local_jsonl", "local_jsonl_sft"} and dataset not in DATASETS:
        # Registering previously unregistered visual markers changes packed
        # samples with unchanged assets. Old media cursors cannot be reused.
        settings["multimodal_encoding"] = "atomic-markers-bounded-packing-v2"
    elif config.dataloader.packing == "first_fit":
        # Chunk boundaries now respect row capacity as well as context length.
        # Grain's pending chunk offsets from the old graph are not portable.
        settings["text_packing"] = "bounded-first-fit-v1"
    tokenizer_json = Path(config.dataloader.tokenizer_path) / "tokenizer.json"
    if tokenizer_json.is_file():
        with tokenizer_json.open(encoding="utf-8") as handle:
            tokenizer_settings = json.load(handle)
        if any(
            tokenizer_settings.get(name) is not None
            for name in ("padding", "truncation")
        ):
            # Older loaders inherited saved batch policies. Their cursor can
            # describe a different packed token stream even with identical files.
            settings["tokenizer_encoding"] = "unrestricted-samples-v1"
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    local_text = dataset in {"local_jsonl", "local_jsonl_sft"}
    if not local_text and config.dataloader.data_identity is None:
        raise ValueError("registered corpus identity requires dataloader.data_identity")
    files = (
        list(file_patterns_to_paths((config.dataloader.dataset_path,)))
        if local_text
        else []
    )
    tokenizer_path = Path(config.dataloader.tokenizer_path)
    for name in (
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = tokenizer_path / name
        digest.update(name.encode())
        if path.is_file():
            files.append(str(path))
        else:
            digest.update(b"absent")
    for filename in files:
        file_digest = hashlib.sha256()
        with open(filename, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return (
        "local-text-v1:" if local_text else "declared-corpus-v1:"
    ) + digest.hexdigest()


class _PackingBuilder(Protocol):
    """Build a packing graph after recipe-specific options have been bound."""

    def __call__(
        self,
        recipe: DatasetRecipe,
        *,
        context: DatasetBuildContext,
        iteration_policy: DatasetIterationPolicy,
    ) -> grain.IterDataset: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class _ResolvedRecipe:
    """Recipe-specific graph and collation choices for shared loader assembly."""

    recipe: DatasetRecipe
    build_packing: _PackingBuilder
    collator: CollatorFactory


def _resolve_multimodal_recipe(
    dataset_name: str, *, num_packing_bins: int
) -> _ResolvedRecipe:
    """Resolve media packing lazily, keeping registries out of configuration."""
    try:
        from .multimodal.collator import MultimodalCollator
        from .multimodal.packing import build_multimodal_packing
        from .multimodal.recipes import MULTIMODAL_DATASETS
    except ImportError as exc:
        raise ImportError(
            f"dataset {dataset_name!r} is not one of the text "
            f"recipes {sorted(DATASETS)}, so it was looked up in the "
            "multimodal registry -- which failed to import. Multimodal "
            "recipes need the optional dependencies torchvision and Pillow "
            "(and av for video): install them with `pip install torchvision "
            "pillow av`, or name a text recipe instead."
        ) from exc
    if dataset_name not in MULTIMODAL_DATASETS:
        raise ValueError(
            f"unknown dataset {dataset_name!r}. Expected 'random', "
            f"'local_jsonl', a text recipe {sorted(DATASETS)}, or a "
            f"multimodal recipe {sorted(MULTIMODAL_DATASETS)}"
        )
    return _ResolvedRecipe(
        recipe=MULTIMODAL_DATASETS[dataset_name],
        build_packing=partial(
            build_multimodal_packing, num_packing_bins=num_packing_bins
        ),
        collator=MultimodalCollator,
    )


def _resolve_text_recipe(
    dataloader_config: DataloaderConfig,
    *,
    dataset_name: str,
    tokenizer: BaseTokenizer,
) -> _ResolvedRecipe:
    """Resolve a text recipe with its configured packing and collation policy."""
    if dataset_name == "local_jsonl":
        recipe = build_local_jsonl(path=dataloader_config.dataset_path)
    elif dataset_name == "local_jsonl_sft":
        if dataloader_config.chat_renderer is not None:
            # Imported here, not at module scope: the renderers package is an
            # optional dependency, and a run that leaves chat_renderer unset
            # must not have to install it.
            from llmtuner.datasets.text.renderer import build_chat_renderer

            recipe = build_local_jsonl_sft_multiturn(
                path=dataloader_config.dataset_path,
                messages_field=dataloader_config.messages_field,
                renderer=build_chat_renderer(
                    tokenizer=tokenizer,
                    renderer_name=dataloader_config.chat_renderer,
                ),
            )
        else:
            recipe = build_local_jsonl_sft(
                path=dataloader_config.dataset_path,
                prompt_field=dataloader_config.prompt_field,
                response_field=dataloader_config.response_field,
            )
    else:
        recipe = DATASETS[dataset_name]
    build_packing: _PackingBuilder = build_concat_then_split_packing
    if dataloader_config.packing == "first_fit":
        # Bind recipe-specific options before the shared graph-building step.
        build_packing = partial(
            build_first_fit_packing,
            num_packing_bins=dataloader_config.num_packing_bins,
        )
    return _ResolvedRecipe(
        recipe=recipe, build_packing=build_packing, collator=TextCollator
    )


def build_dataloader(
    config: LLMTunerConfig,
    *,
    dp_rank: int,
    dp_world_size: int,
    num_tokens_per_batch: int,
    repeat: bool = True,
    dataset: str | None = None,
) -> BaseDataLoader:
    """Build a checkpointable loader for this rank's data ownership.

    Args:
        config: dataset, tokenization, packing, and run geometry.
        dp_rank: rank within the data-parallel group; PP stages in one pipeline
            use the same coordinates so they read the same batch.
        dp_world_size: number of independent data shards.
        num_tokens_per_batch: token budget per rank. Text PP divides this budget
            among independently packed microbatch rows before runtime splitting.
            Random data derives its row count from config.global_batch_size.
        repeat: repeat corpus data indefinitely when true. False builds a finite
            corpus loader only with dp_world_size=1; distributed exhaustion is
            not coordinated. Random data remains infinite regardless of repeat.
        dataset: optional recipe override, used for held-out validation data
            without changing config.dataloader.dataset.

    Synthetic data bypasses tokenizer and packing setup. Corpus loaders resolve
    a text or multimodal recipe, construct its tokenizer and packing graph, then
    bind the corresponding collator. Multimodal packing uses whole documents
    and does not use the text packing selector or PP row grouping.
    """
    validate_dp_coordinates(dp_rank=dp_rank, dp_world_size=dp_world_size)
    dataloader_config = config.dataloader
    max_context_length = config.max_seq_len
    dataset_name = dataloader_config.dataset if dataset is None else dataset
    if dataset_name == "random":
        return RandomTokenDataLoader(
            seed=config.seed,
            vocab_size=config.vocab_size,
            batch_size=config.global_batch_size,
            seq_len=max_context_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

    validate_loader_iteration(repeat=repeat, dp_world_size=dp_world_size)

    # Try names outside the text registry and local builders as media recipes.
    is_multimodal = (
        dataset_name not in {"local_jsonl", "local_jsonl_sft"}
        and dataset_name not in DATASETS
    )

    require_positive_integer("num_tokens_per_batch", num_tokens_per_batch)
    num_microbatches = (
        config.parallel.pp_config.num_pp_microbatches
        if not is_multimodal and config.parallel.pipeline_parallel_size > 1
        else 1
    )
    require_positive_integer("num_pp_microbatches", num_microbatches)
    if num_tokens_per_batch % num_microbatches:
        raise ValueError("token batch must be divisible by num_pp_microbatches")
    packing_tokens_per_batch = num_tokens_per_batch // num_microbatches

    data_identity = (
        corpus_data_identity(config, dataset=dataset_name, repeat=repeat)
        if dataset_name in {"local_jsonl", "local_jsonl_sft"}
        or dataloader_config.data_identity is not None
        else None
    )

    if is_multimodal:
        # Resolved before the tokenizer is constructed, so a bad name fails
        # fast without loading tokenizer assets.
        from llmtuner.components.tokenizer.multimodal import MultimodalTokenizer

        resolved = _resolve_multimodal_recipe(
            dataset_name, num_packing_bins=dataloader_config.num_packing_bins
        )
        tokenizer = MultimodalTokenizer(
            tokenizer_path=dataloader_config.tokenizer_path,
            image_token=dataloader_config.mm_image_token,
            video_token=dataloader_config.mm_video_token,
            vision_start_token=dataloader_config.mm_vision_start_token,
            vision_end_token=dataloader_config.mm_vision_end_token,
            pad_token=dataloader_config.mm_pad_token,
        )
        # Media lists require whole-document FirstFit packing; --packing applies
        # only to text. The media collator constructs patches after packing.
    else:
        # Keep tokenizer dependencies out of the synthetic-data path.
        from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer_path=dataloader_config.tokenizer_path
        )
        resolved = _resolve_text_recipe(
            dataloader_config, dataset_name=dataset_name, tokenizer=tokenizer
        )
    context = DatasetBuildContext(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=packing_tokens_per_batch,
        read_options=grain.ReadOptions(),
        max_num_documents=dataloader_config.max_num_documents,
    )
    collator = (
        partial(PipelineTextCollator, num_microbatches=num_microbatches)
        if num_microbatches > 1
        else resolved.collator
    )
    # Apply ordering and sharding to the graph before loader batching/prefetch.
    graph = resolved.build_packing(
        resolved.recipe,
        context=context,
        iteration_policy=DatasetIterationPolicy(
            seed=config.seed,
            shuffle=dataloader_config.shuffle,
            repeat=repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_buffer_size=dataloader_config.streaming_shuffle_buffer_size,
        ),
    )
    return GrainDataLoader(
        graph,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
        collator=collator,
        repeat=repeat,
        num_prefetch_batches=dataloader_config.num_prefetch_batches,
        max_num_documents=dataloader_config.max_num_documents,
        data_identity=data_identity,
    )
