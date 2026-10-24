"""Parse flat configuration flags and run training or seed checkpoint export.

HfArgumentParser receives the scalar views defined in config/cli_config.py.
Each group validates its fields; LLMTunerConfig.from_groups assembles them into
one run configuration. This entry point accepts flags, with no implicit args
file or config-file channel.

The console owns the process-group lifetime. It closes the trainer and destroys
the group on exit, preserving the original error if cleanup also fails.

Examples:
    python -m llmtuner --steps 20
    torchrun --nproc_per_node=2 -m llmtuner --data_parallel_shard_size -1
    python -m llmtuner --steps 20 --enable --interval 10 --dump_folder ./outputs
    python -m llmtuner --steps 20 --enable_tensorboard
    python -m llmtuner --steps 20 --dataset local_jsonl \\
        --dataset_path ./corpus.jsonl --tokenizer_path ./tokenizer
    python -m llmtuner --steps 20 --enable_profiling --profile_freq 4
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import torch.distributed as dist
from transformers import HfArgumentParser

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.cli_config import PARSER_GROUPS, cli_view
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig, ProfilerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from llmtuner.config.training_config import TrainingConfig
from llmtuner.errors import ConfigError
from llmtuner.utils.logging import get_logger

logger = get_logger(__name__)


def parse_config(args: Sequence[str] | None = None) -> LLMTunerConfig:
    """Parse explicit flags, or the process argv when args is omitted."""
    # ``PARSER_GROUPS`` is the parser's group set and order (see
    # ``config/cli_config.py``, which also explains the views ``cli_view`` builds:
    # they hide the fields the CLI cannot carry). Parsing returns one instance
    # per view, in that order, so the two tuples zip; the instances are
    # subclasses of the groups, which is how they are keyed below.
    parser = HfArgumentParser([cli_view(group) for group in PARSER_GROUPS])
    # Each group is its own parser group, so every scalar field becomes a flag.
    parsed = dict(
        zip(
            PARSER_GROUPS,
            parser.parse_args_into_dataclasses(
                args=None if args is None else list(args), look_for_args_file=False
            ),
            strict=True,
        )
    )
    return LLMTunerConfig.from_groups(
        model=parsed[ModelConfig],
        parallel=parsed[ParallelConfig],
        fsdp_config=parsed[FSDPConfig],
        tp_config=parsed[TensorParallelConfig],
        pp_config=parsed[PipelineParallelConfig],
        cp_config=parsed[ContextParallelConfig],
        ep_config=parsed[ExpertParallelConfig],
        optimizer=parsed[OptimizerConfig],
        lr_scheduler=parsed[LRSchedulerConfig],
        training=parsed[TrainingConfig],
        checkpoint=parsed[CheckpointConfig],
        dataloader=parsed[DataloaderConfig],
        metrics=parsed[MetricsConfig],
        profiler=parsed[ProfilerConfig],
    )


def main() -> None:
    cfg = parse_config()
    if cfg.checkpoint.create_seed_checkpoint:
        world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        if world_size != 1:
            raise ConfigError(
                "Must create a seed checkpoint using a single device, to "
                "disable sharding."
            )
        if not cfg.checkpoint.enable:
            raise ConfigError(
                "Must enable checkpointing when creating a seed checkpoint."
            )
    trainer = None
    try:
        from .trainer import Trainer

        trainer = Trainer(cfg)
        if cfg.checkpoint.create_seed_checkpoint:
            # Mirrors torchtitan ``train.py``: a seed checkpoint is the unsharded
            # step-0 model, so it must be written from a single process (any
            # sharding would bake one rank's shard layout into the artifact), and
            # loading treats step-0 as model-only (see ``checkpointer/base.py``).
            if trainer.checkpointer.save(curr_step=0, last_step=True):
                print("Created seed checkpoint at step 0")
        else:
            trainer.train()
    finally:
        # Trainer.close releases model-side resources but deliberately leaves
        # the process group alive for callers that run another collective (for
        # example validation or post-run aggregation).  The console entrypoint
        # owns the process-group lifetime and tears it down here, matching
        # torchtitan's train.py lifecycle.
        training_error = sys.exc_info()[1]
        try:
            if trainer is not None:
                trainer.close()
        except BaseException:
            if training_error is None:
                raise
            try:
                logger.exception("Resource cleanup failed after an entry-point error")
            except BaseException:
                pass  # Keep the primary entry-point error when reporting fails.
        finally:
            cleanup_error = sys.exc_info()[1]
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except BaseException:
                if training_error is None and cleanup_error is None:
                    raise
                try:
                    logger.exception(
                        "Process-group cleanup failed after an entry-point error"
                    )
                except BaseException:
                    pass  # Preserve the active training or resource cleanup error.


if __name__ == "__main__":
    main()
