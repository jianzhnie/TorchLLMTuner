"""Assemble dataloaders from run configuration and data-parallel ownership.

Resolve a text or multimodal recipe, build its tokenizer and packing graph,
then construct the loader. Synthetic data bypasses tokenizer and packing setup.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import grain.python as grain

from experiments.pipeline_parallel.collators import PipelineTextCollator
from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
from llmtuner.components.tokenizer.kimi import KimiK3Tokenizer
from llmtuner.components.tokenizer.multimodal import MultimodalTokenizer
from llmtuner.datasets.collators import (
    ContextRowsCollator,
)
from llmtuner.datasets.contracts import (
    BaseDataLoader,
    DatasetBuildContext,
    DatasetIterationPolicy,
    require_positive_integer,
    validate_dp_coordinates,
    validate_loader_iteration,
)
from llmtuner.datasets.corpus_identity import (
    corpus_data_identity as corpus_data_identity,
)
from llmtuner.datasets.dataloader import GrainDataLoader
from llmtuner.datasets.loader_builder import (
    _resolve_multimodal_recipe,
    _resolve_text_recipe,
)
from llmtuner.datasets.packing.global_rows import GlobalRowsIterDataset
from llmtuner.datasets.synthetic import RandomTokenDataLoader
from llmtuner.datasets.text.recipes import (
    DATASETS,
)

if TYPE_CHECKING:
    from experiments.pipeline_parallel.config import (
        LLMTunerConfig,
    )

__all__ = ["build_dataloader"]


def resolve_packing_geometry(
    config: LLMTunerConfig,
    *,
    is_multimodal: bool,
    num_tokens_per_batch: int,
    repeat: bool,
) -> tuple[int, int, int]:
    dataloader_config = config.dataloader
    max_context_length = config.max_seq_len
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
    global_rows = dataloader_config.global_context_rows
    if global_rows:
        if (
            is_multimodal
            or config.parallel.pipeline_parallel_size > 1
            or config.gradient_accumulation_steps != 1
            or not repeat
        ):
            raise ValueError(
                "global_context_rows currently requires repeated text, PP=1 "
                "and gradient_accumulation_steps=1"
            )
        if num_tokens_per_batch % max_context_length:
            raise ValueError("global_context_rows requires whole context rows")
        packing_tokens_per_batch = max_context_length
    return packing_tokens_per_batch, num_microbatches, global_rows


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
        if dataloader_config.global_context_rows:
            raise ValueError("global_context_rows requires a text corpus dataset")
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

    packing_tokens_per_batch, num_microbatches, global_rows = resolve_packing_geometry(
        config,
        is_multimodal=is_multimodal,
        num_tokens_per_batch=num_tokens_per_batch,
        repeat=repeat,
    )

    data_identity = (
        corpus_data_identity(config, dataset=dataset_name, repeat=repeat)
        if dataset_name in {"local_jsonl", "local_jsonl_sft"}
        or dataloader_config.data_identity is not None
        else None
    )

    if is_multimodal:
        # Resolved before the tokenizer is constructed, so a bad name fails
        # fast without loading tokenizer assets.
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
        if dataloader_config.tokenizer_backend == "kimi_k3":
            tokenizer = KimiK3Tokenizer(tokenizer_path=dataloader_config.tokenizer_path)
        else:
            tokenizer = HuggingFaceTokenizer(
                tokenizer_path=dataloader_config.tokenizer_path
            )
        if dataloader_config.chat_template is not None:
            tokenizer.set_chat_template(dataloader_config.chat_template)
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
    if global_rows:
        collator = partial(
            ContextRowsCollator,
            num_microbatches=num_tokens_per_batch // max_context_length,
        )
    # Apply ordering and sharding to the graph before loader batching/prefetch.
    graph = resolved.build_packing(
        resolved.recipe,
        context=context,
        iteration_policy=DatasetIterationPolicy(
            seed=config.seed,
            shuffle=dataloader_config.shuffle,
            repeat=repeat,
            dp_rank=0 if global_rows else dp_rank,
            dp_world_size=1 if global_rows else dp_world_size,
            streaming_shuffle_buffer_size=dataloader_config.streaming_shuffle_buffer_size,
        ),
    )
    if global_rows:
        graph = GlobalRowsIterDataset(
            graph,
            global_rows=num_tokens_per_batch * dp_world_size // max_context_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
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
