"""Training-loop config; the nested configs it grafts live in sibling modules
(``observability`` / ``activation_checkpoint`` / ``compile`` / ``validation``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from llmtuner.config.activation_checkpoint_config import (
    VALID_AC_MODES,
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.config.observability_config import MetricsConfig, ProfilerConfig
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.errors import ConfigError


def validate_activation_swap_options(
    *, activation_swap: bool, compile: bool, activation_checkpoint: str
) -> None:
    """Check saved-tensor hook compatibility before any model transformation.

    Shared by aggregate config validation and direct parallel assembly. Full
    checkpointing skips swap; selective and region policies own competing hooks.
    """
    for name, value in (("activation_swap", activation_swap), ("compile", compile)):
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be bool, got {value!r}")
    if activation_swap and compile:
        raise ConfigError(
            "activation_swap=True requires compile=False because saved-tensor "
            "copy hooks are Python runtime boundaries, not compile regions."
        )
    if activation_swap and activation_checkpoint in ("selective", "region"):
        raise ConfigError(
            f"activation_swap=True is incompatible with activation_checkpoint="
            f"{activation_checkpoint!r}; checkpoint saved-tensor hooks prevent "
            "activation swap from releasing and restoring tensors. "
            "Set training.activation_checkpoint_mode='none'; full mode skips swap."
        )


@dataclass
class TrainingConfig:
    """Training loop hyperparameters and reproducibility."""

    global_batch_size: int = field(
        default=8,
        metadata={
            "help": "Sequences per step across DP ranks. PP splits each rank batch "
            "into num_pp_microbatches."
        },
    )
    max_seq_len: int = field(default=64, metadata={"help": "Sequence length"})
    steps: int = field(default=20, metadata={"help": "Number of optimizer steps"})
    seed: int = field(default=42, metadata={"help": "Base RNG seed"})
    compile: bool = field(default=False, metadata={"help": "torch.compile the model"})
    compile_config: CompileConfig = field(
        default_factory=CompileConfig,
        metadata={
            "help": "How the model is compiled (per-block vs whole-model, "
            "backend, async TP). Read only under compile=True; its defaults "
            "reproduce the plain whole-model compile."
        },
    )
    activation_checkpoint_mode: str = field(
        default="none",
        metadata={
            "help": "Activation checkpointing: none, full, selective, memory_budget "
            "(requires compile), or region (requires torch_remat). Applied "
            "after TP/EP/CP and before compile/FSDP."
        },
    )
    activation_swap: bool = field(
        default=False,
        metadata={
            "help": "Swap saved decoder activations to pinned Host memory on NPU. "
            "Works with AC mode 'none'; full AC skips swapping, while selective "
            "and region AC are incompatible. Also incompatible with compile."
        },
    )
    activation_swap_sync_release: bool = field(
        default=False,
        metadata={
            "help": "Wait for activation D2H copies on the compute stream before "
            "continuing forward. Used only when activation_swap=True."
        },
    )
    activation_swap_max_host_bytes: int = field(
        default=0,
        metadata={
            "help": "Maximum live swapped activation Host bytes per model part "
            "and rank (0: unlimited). At capacity keep activations on device. "
            "Excludes allocator cache, virtual optimizer state and process RSS."
        },
    )
    selective_ac: SelectiveACConfig = field(
        default_factory=SelectiveACConfig,
        metadata={
            "help": "Selective activation checkpointing. Read only under "
            "activation_checkpoint_mode='selective'."
        },
    )
    memory_budget_ac: MemoryBudgetACConfig = field(
        default_factory=MemoryBudgetACConfig,
        metadata={
            "help": "Memory-budget activation checkpointing. Read only under "
            "activation_checkpoint_mode='memory_budget'."
        },
    )
    region_ac: RegionACConfig = field(
        default_factory=RegionACConfig,
        metadata={
            "help": "Region activation checkpointing (torch_remat). Read only "
            "under activation_checkpoint_mode='region', and the mode needs the "
            "optional torch_remat package (torch >= 2.10) when the policy is "
            "applied."
        },
    )
    deterministic: bool = field(
        default=True,
        metadata={
            "help": "Deterministic algorithms -- required for bit-exact comparison"
        },
    )
    detect_anomaly: bool = field(
        default=False,
        metadata={
            "help": "Enable autograd anomaly detection (debug only, significant "
            "overhead). NaN/Inf gradient checks stay off because they need "
            "aten._is_any_true, which has no DTensor sharding strategy."
        },
    )
    max_norm: float = field(
        default=1.0,
        metadata={
            "help": "Gradient-norm clip threshold. A non-positive value disables "
            "clipping but still reports grad_norm."
        },
    )
    grad_norm_precision: Literal["float32", "parameter"] = field(
        default="float32",
        metadata={
            "help": "Compute per-gradient norms in FP32 (default), or in the "
            "gradient dtype before FP32 power reduction to match TorchTitan-NPU."
        },
    )
    gc_freq: int = field(
        default=50,
        metadata={
            "help": "Run a cyclic garbage collection every this many steps. The "
            "training loop takes the collector over from CPython so it fires at a "
            "step boundary instead of mid-forward."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Full forward/backward microbatches per optimizer update. PP "
            "microbatches instead split each batch; loss uses the global "
            "valid-token count."
        },
    )
    chunked_loss_num_chunks: int = field(
        default=1,
        metadata={
            "help": "Sequence chunks for lm_head and cross-entropy; reduces peak "
            "logits memory. 1 disables chunking; unsupported with PP."
        },
    )
    dump_folder: str = field(
        default="./outputs",
        metadata={
            "help": "Root directory for this run's outputs. The checkpoint, "
            "TensorBoard and profiling folders are resolved against it."
        },
    )
    # A plain `checkpoint` field would be nicer to read, but a dataclass field
    # and the class it types cannot share a name.
    checkpoint_config: CheckpointConfig = field(
        default_factory=CheckpointConfig,
        metadata={"help": "Checkpointing (see components/checkpointer)."},
    )
    dataloader_config: DataloaderConfig = field(
        default_factory=DataloaderConfig,
        metadata={"help": "Micro-batch source (see datasets/)."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics reporting (see components/metrics)."},
    )
    profiler_config: ProfilerConfig = field(
        default_factory=ProfilerConfig,
        metadata={"help": "Profiling (see components/profiler)."},
    )
    ema_config: EMAConfig | None = field(
        default=None,
        metadata={
            "help": "Online EMA of model weights (see components/optimizer). "
            "None (the default) disables it. No CLI flag -- set it from code, "
            "like optimizer.param_groups."
        },
    )
    validation_config: ValidationConfig | None = field(
        default=None,
        metadata={
            "help": "Periodic validation pass (see Trainer.validate). None "
            "(the default) disables it. No CLI flag -- set it from code, like "
            "ema_config."
        },
    )

    @property
    def checkpoint(self) -> CheckpointConfig:
        """The checkpoint manager's config."""
        return self.checkpoint_config

    @property
    def metrics(self) -> MetricsConfig:
        """The metrics processor's config. See ``checkpoint`` for the shape."""
        return self.metrics_config

    @property
    def dataloader(self) -> DataloaderConfig:
        """The micro-batch source's config. See ``checkpoint`` for the shape."""
        return self.dataloader_config

    @property
    def profiler(self) -> ProfilerConfig:
        """The profiler's config. See ``checkpoint`` for the shape."""
        return self.profiler_config

    @property
    def ema(self) -> EMAConfig | None:
        """The weight EMA's config, or None when EMA is off."""
        return self.ema_config

    @property
    def validation(self) -> ValidationConfig | None:
        """The validation loop's config, or None when validation is off."""
        return self.validation_config

    def __post_init__(self) -> None:
        require_integer_at_least(
            self, "activation_swap_max_host_bytes", minimum=0, group="training"
        )
        if self.grad_norm_precision not in ("float32", "parameter"):
            raise ConfigError(
                "training.grad_norm_precision must be float32 or parameter"
            )
        for name in (
            "compile",
            "deterministic",
            "detect_anomaly",
            "activation_swap",
            "activation_swap_sync_release",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ConfigError(f"training.{name} must be bool, got {value!r}")
        if (
            not isinstance(self.max_norm, int | float)
            or isinstance(self.max_norm, bool)
            or not math.isfinite(self.max_norm)
        ):
            raise ConfigError(
                f"training.max_norm must be a finite number, got {self.max_norm!r}"
            )
        # One check, not five copies: these fields share the floor and the
        # wording, and the same shape guards ParallelConfig's degrees. The
        # text is the contract -- test_config.py pins
        # ``f"{field} must be >= 1"`` for each of these names.
        require_integer_at_least(
            self,
            "global_batch_size",
            "max_seq_len",
            "steps",
            "gradient_accumulation_steps",
            "gc_freq",
            group="training",
        )
        require_integer_at_least(self, "chunked_loss_num_chunks", group="training")
        if self.activation_checkpoint_mode not in VALID_AC_MODES:
            raise ConfigError(
                "training.activation_checkpoint_mode must be one of: "
                f"{VALID_AC_MODES} (got {self.activation_checkpoint_mode!r})"
            )
        if self.activation_checkpoint_mode == "memory_budget" and not self.compile:
            raise ConfigError(
                "training.activation_checkpoint_mode='memory_budget' requires "
                "training.compile=True: the budget is consumed by the compile "
                "partitioner, so without compile it would silently do nothing."
            )
