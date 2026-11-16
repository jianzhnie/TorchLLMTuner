"""Pipeline schedule configuration, isolated from production training."""

from dataclasses import dataclass, field

from torch.distributed.pipelining.schedules import get_schedule_class

from llmtuner.config.parallel_config import (
    ContextParallelConfig as ContextParallelConfig,
)
from llmtuner.config.parallel_config import (
    ExpertParallelConfig as ExpertParallelConfig,
)
from llmtuner.config.parallel_config import (
    FSDPConfig as FSDPConfig,
)
from llmtuner.config.parallel_config import (
    ParallelConfig as BaseParallelConfig,
)
from llmtuner.config.parallel_config import (
    TensorParallelConfig as TensorParallelConfig,
)
from llmtuner.config.utils import require_integer_at_least
from llmtuner.errors import ConfigError


@dataclass(kw_only=True, slots=True)
class PipelineParallelConfig:
    """Execution policy for PP; parallel degrees live in ParallelConfig."""

    module_fqns_per_model_part: list[list[str]] | None = None
    """Explicit module FQNs per pipeline chunk; nested lists are programmatic-only."""

    pipeline_parallel_first_stage_less_layers: int = 1
    """Layers removed from the first stage to offset embedding overhead."""

    pipeline_parallel_last_stage_less_layers: int = 1
    """Layers removed from the last stage to offset output-layer overhead."""

    pipeline_parallel_layers_per_stage: int | None = None
    """Layers per virtual stage; None infers the split from model and schedule."""

    pipeline_parallel_schedule: str = "1F1B"
    """PyTorch pipeline schedule name; must match the stage layout."""

    pipeline_parallel_schedule_csv: str | None = ""
    """Reserved CSV schedule path; non-empty values are unsupported."""

    num_pp_microbatches: int = 1
    """Pipeline splits per DP rank and accumulation iteration; ignored when PP = 1."""

    def __post_init__(self) -> None:
        require_integer_at_least(self, "num_pp_microbatches", group="parallel")
        if self.pipeline_parallel_schedule_csv:
            raise ConfigError(
                "parallel.pipeline_parallel_schedule_csv is not supported; "
                "use pipeline_parallel_schedule instead."
            )
        try:
            get_schedule_class(self.pipeline_parallel_schedule)
        except ValueError as exc:
            raise ConfigError(
                "Invalid parallel.pipeline_parallel_schedule "
                f"{self.pipeline_parallel_schedule!r}: {exc}"
            ) from exc


@dataclass
class ParallelConfig(BaseParallelConfig):
    pipeline_parallel_size: int = 1
    pp_config: PipelineParallelConfig = field(default_factory=PipelineParallelConfig)

    @property
    def pp_size(self) -> int:
        return self.pipeline_parallel_size

    def non_dp_sizes(self) -> int:
        return super().non_dp_sizes() * self.pipeline_parallel_size

    def validate(self) -> None:
        super().validate()
        self.pp_config.validate()
