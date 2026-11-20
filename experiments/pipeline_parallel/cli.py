"""Parse flat configuration flags and run training or seed checkpoint export.

HfArgumentParser receives the scalar views defined in config/cli_config.py.
Each group validates its fields; LLMTunerConfig.from_groups assembles them into
one run configuration. This entry point accepts flags, with no implicit args
file or config-file channel.

The console owns the process-group lifetime. It closes the trainer and destroys
the group on exit, preserving the original error if cleanup also fails.

Examples:
    python -m experiments.pipeline_parallel.cli --steps 20
    torchrun --nproc_per_node=2 -m experiments.pipeline_parallel.cli \\
        --pipeline_parallel_size 2 --num_pp_microbatches 2
    python -m experiments.pipeline_parallel.cli --steps 20 --enable \\
        --interval 10 --dump_folder ./outputs
"""

from __future__ import annotations

from collections.abc import Sequence

from transformers import HfArgumentParser

from experiments.pipeline_parallel.cli_config import PARSER_GROUPS, cli_view
from experiments.pipeline_parallel.config import LLMTunerConfig
from experiments.pipeline_parallel.parallel_config import (
    ContextParallelConfig,
    ExpertParallelConfig,
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from experiments.pipeline_parallel.trainer import Trainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig, ProfilerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.cli import run_training


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
    run_training(parse_config(), trainer_factory=Trainer)


if __name__ == "__main__":
    main()
