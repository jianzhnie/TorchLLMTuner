"""Observability configs: metrics reporting and profiling.

Both are nested configs grafted onto ``TrainingConfig``; neither constructs the
runtime object it describes (``components/metrics.py`` /
``components/profiler.py`` own those) -- configs here are only ever read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmtuner.config.validate import require_at_least
from llmtuner.errors import ConfigError


@dataclass(kw_only=True)
class MetricsConfig:
    """What the metrics processor reports, and where.

    ``log_freq`` and ``tag`` are this module's own: the first is a CLI concern
    (the processor reads its window length from here), the second names a logging
    key. Both live with the config rather than the processor because the config
    is what the command line describes.
    """

    log_freq: int = 1
    """Console log frequency, in steps. Also the TensorBoard/WandB frequency --
    one window feeds both."""

    enable_tensorboard: bool = False
    """Whether to write TensorBoard event files."""

    disable_color_printing: bool = False
    """Whether to drop colour from the console line."""

    save_tb_folder: str = "tb"
    """TensorBoard folder, relative to the dump folder."""

    save_for_all_ranks: bool = False
    """Whether every rank logs, rather than only the metrics rank."""

    log_ranks: list[int] = field(default_factory=lambda: [0])
    """Ranks whose below-ERROR lines reach the console. Under pipeline
    parallelism the loss lives on the last stage's first rank; add it here
    (``--log_ranks 0 7``) to see its lines. Default: rank 0 only."""

    enable_wandb: bool = False
    """Whether to stream metrics to Weights & Biases."""

    tag: str | None = None
    """Prefix applied to every recorded key in TensorBoard/WandB. The console
    line is not prefixed: it is read live, never merged."""

    def __post_init__(self) -> None:
        if self.log_freq < 1:
            raise ConfigError(f"metrics.log_freq must be >= 1, got {self.log_freq}")
        if any(r < 0 for r in self.log_ranks):
            raise ConfigError(
                f"metrics.log_ranks entries must be >= 0, got {self.log_ranks}"
            )


@dataclass(kw_only=True)
class ProfilerConfig:
    """What the profiler collects, and when."""

    enable_profiling: bool = False
    """Whether to collect Kineto traces."""

    save_traces_folder: str = "profiling/traces"
    """Trace location, relative to the base folder."""

    profile_freq: int = 10
    """How often to collect a trace, in iterations."""

    profiler_repeat: int | None = None
    """How many times to repeat the profiling cycle. ``None`` repeats forever,
    which is ``torch.profiler.schedule``'s own default."""

    profiler_skip_first: int | None = None
    """How many iterations to skip before the schedule starts."""

    profiler_skip_first_wait: int | None = None
    """How many waits to skip at the start of the first cycle."""

    profiler_active: int = 1
    """Iterations the profiler is active for, per cycle."""

    profiler_warmup: int = 3
    """Warmup iterations before the active ones in each cycle.

    Warmup discards its results, so it is what keeps the first active iteration
    from being dominated by lazy initialization.
    """

    enable_memory_snapshot: bool = False
    """Whether to write allocator memory snapshots."""

    memory_snapshot_freq: int | None = None
    """Snapshot frequency, in iterations. Defaults to ``profile_freq``."""

    save_memory_snapshot_folder: str = "profiling/memory_snapshot"
    """Snapshot location, relative to the base folder.

    Spelled out rather than derived from ``save_traces_folder``: the two are
    independent knobs, and a derived default would silently follow a custom
    trace folder into an unexpected place.
    """

    memory_snapshot_max_entries: int = 1_000_000
    """Alloc/free events kept per snapshot.

    The allocator history is a ring buffer, so this bounds how far back a
    snapshot can see, and with it the dump's size and cost.
    """

    def __post_init__(self) -> None:
        cycle = self.profiler_warmup + self.profiler_active
        if self.enable_profiling and self.profile_freq < cycle:
            raise ConfigError(
                "profiler.profile_freq must be >= profiler_warmup + "
                f"profiler_active ({cycle}), got {self.profile_freq}"
            )
        require_at_least(
            self, "profiler_active", "memory_snapshot_max_entries", group="profiler"
        )
        require_at_least(self, "profiler_warmup", group="profiler", minimum=0)


# HfArgumentParser cannot turn a nested dataclass into a set of flags -- it
# collapses it to one opaque `--<field>` argument and never consults the fields
# inside. The `<x>_config` field is that escape hatch: the config's own group is
# parsed separately in train.py and grafted back on here, which is also where
# its __post_init__ re-runs against the parsed values.
#
# That is what keeps every class in this package named `*Config`. Renaming one is
# not cosmetic -- HfArgumentParser takes the flag stem from the FIELD, so the
# field names below are the CLI: `metrics_config` is what makes this class's
# fields reach the user as `--log_freq` and `--enable_tensorboard`. A class
# named `MetricsArguments` with a field `metrics_arguments` would move every one
# of them behind a `--metrics_arguments` prefix, which is a breaking change.
