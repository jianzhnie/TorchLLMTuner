"""Parallel execution policies, mesh degrees, and combination validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from llmtuner.config.field_checks_config import require_integer_at_least
from llmtuner.errors import (
    ConfigError,
    EnvironmentUnsupportedError,
    UnsupportedCombinationError,
)

__all__ = [
    "CP_LOAD_BALANCERS",
    "CP_STRATEGIES",
    "EP_DISPATCHERS",
    "ContextParallelConfig",
    "ExpertParallelConfig",
    "FSDPConfig",
    "FSDPReshardPolicy",
    "FSDPSymmMemScope",
    "ParallelConfig",
    "PipelineParallelConfig",
    "TensorParallelConfig",
    "validate_expert_dispatcher",
]

FSDPReshardPolicy = Literal["default", "always", "never"]


FSDPSymmMemScope = Literal["all", "dense"]


@dataclass(kw_only=True, slots=True)
class FSDPConfig:
    """Execution policy for FSDP; parallel degrees live in ParallelConfig."""

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """FSDP2 parameter resharding policy after forward."""

    fsdp_cpu_offload: bool = False
    """Offload FSDP2 parameters and gradients to CPU between computations."""

    fsdp_cpu_offload_pin_memory: bool = True
    """Use pinned host memory for FSDP2 CPU offload transfers."""

    fsdp_forward_prefetch_count: int = 0
    """Number of later FSDP units explicitly prefetched during forward."""

    fsdp_backward_prefetch_count: int = 0
    """Number of earlier FSDP units explicitly prefetched during backward."""

    fsdp_apply_modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Programmatic-only module patterns and per-unit ``fully_shard`` overrides."""

    fsdp_ignored_modules: list[str] = field(default_factory=list)
    """Module patterns excluded from FSDP parameter management."""

    fsdp_implementation: Literal["native", "custom"] = "native"
    """FSDP engine. ``custom`` selects the fine-grained implementation."""

    enable_fsdp_symm_mem: bool = False
    """Enable FSDP2 symmetric-memory communication when supported."""

    fsdp_symm_mem_scope: Literal["all", "dense"] = "all"
    """Apply symmetric-memory communication to all or dense FSDP modules."""

    def __post_init__(self) -> None:
        if self.fsdp_reshard_after_forward not in {"default", "always", "never"}:
            raise ConfigError(
                "parallel.fsdp_reshard_after_forward must be one of "
                "'default'/'always'/'never', got "
                f"{self.fsdp_reshard_after_forward!r}"
            )
        if self.fsdp_symm_mem_scope not in ("all", "dense"):
            raise ConfigError(
                "parallel.fsdp_symm_mem_scope must be one of 'all'/'dense', got "
                f"{self.fsdp_symm_mem_scope!r}"
            )
        require_integer_at_least(
            self,
            "fsdp_forward_prefetch_count",
            "fsdp_backward_prefetch_count",
            group="parallel",
            minimum=0,
        )
        if self.fsdp_implementation not in ("native", "custom"):
            raise ConfigError(
                "parallel.fsdp_implementation must be 'native' or 'custom'"
            )
        if self.fsdp_ignored_modules and not self.fsdp_apply_modules:
            raise ConfigError(
                "parallel.fsdp_ignored_modules requires fsdp_apply_modules; "
                "ignored patterns only belong to the pattern-driven FSDP path"
            )
        if self.enable_fsdp_symm_mem and (
            not torch.cuda.is_available()
            or (
                torch.version.hip is None
                and torch.cuda.get_device_capability() < (9, 0)
            )
        ):
            raise ConfigError(
                "parallel.enable_fsdp_symm_mem is only supported on NVIDIA "
                "GPUs with compute capability 9.0 or newer."
            )


@dataclass(kw_only=True, slots=True)
class TensorParallelConfig:
    """Execution policy for TP; parallel degrees live in ParallelConfig."""

    enable_sequence_parallel: bool = True
    """Required for TP > 1; TP = 1 permits disabling it."""

    tp_colwise_modules: list[str] = field(default_factory=list)

    tp_rowwise_modules: list[str] = field(default_factory=list)

    tp_sequence_modules: list[str] = field(default_factory=list)
    """Optional pattern overrides layered on top of the model TP plan."""

    def __post_init__(self) -> None:
        if self.tp_sequence_modules:
            raise UnsupportedCombinationError(
                "parallel.tp_sequence_modules is not supported by the fused sequence "
                "parallel engine; normalization already consumes the sequence shard"
            )
        conflicting = set(self.tp_colwise_modules) & set(self.tp_rowwise_modules)
        if conflicting:
            raise ConfigError(
                f"TP patterns {sorted(conflicting)!r} have conflicting styles in "
                "parallel.tp_colwise_modules and parallel.tp_rowwise_modules"
            )


CP_STRATEGIES = frozenset({"kv_allgather", "ulysses", "kcp"})


CP_LOAD_BALANCERS = frozenset({None, "headtail", "ptrr"})


@dataclass(kw_only=True, slots=True)
class ContextParallelConfig:
    """Execution policy for CP; parallel degrees live in ParallelConfig."""

    context_parallel_strategy: str = "kv_allgather"
    """Attention redistribution: kv_allgather, ulysses, or kcp."""

    context_parallel_load_balancer: str | None = None
    """Sequence balancing: None or headtail; ptrr is unsupported. Ulysses/KCP require
    None.
    """

    def __post_init__(self) -> None:
        if self.context_parallel_load_balancer == "":
            raise ConfigError(
                "parallel.context_parallel_load_balancer cannot be an empty "
                "string; use None to disable load balancing."
            )
        if self.context_parallel_load_balancer not in CP_LOAD_BALANCERS:
            raise ConfigError(
                "parallel.context_parallel_load_balancer must be one of "
                "None/'headtail'/'ptrr', got "
                f"{self.context_parallel_load_balancer!r}"
            )
        if self.context_parallel_load_balancer == "ptrr":
            raise UnsupportedCombinationError(
                "parallel.context_parallel_load_balancer='ptrr' is not "
                "implemented in llmtuner; use 'headtail' or None."
            )
        if self.context_parallel_strategy not in CP_STRATEGIES:
            raise ConfigError(
                "parallel.context_parallel_strategy must be one of "
                "'kv_allgather'/'ulysses'/'kcp', got "
                f"{self.context_parallel_strategy!r}"
            )
        if self.context_parallel_strategy == "kcp":
            if self.context_parallel_load_balancer is not None:
                raise UnsupportedCombinationError(
                    "KCP requires contiguous sequence shards"
                )
        if (
            self.context_parallel_strategy == "ulysses"
            and self.context_parallel_load_balancer is not None
        ):
            raise UnsupportedCombinationError(
                "parallel.context_parallel_strategy='ulysses' requires "
                "context_parallel_load_balancer=None."
            )


EP_DISPATCHERS = ("alltoall", "torchao", "domino", "mc2", "deepep", "hybridep")


def validate_expert_dispatcher(backend: str) -> None:
    """Validate backend availability without requiring a parallel config."""
    if backend not in EP_DISPATCHERS:
        raise ConfigError(
            "parallel.ep_token_dispatcher must be one of "
            f"{'/'.join(EP_DISPATCHERS)}, got {backend!r}"
        )
    if backend in ("deepep", "hybridep"):
        raise EnvironmentUnsupportedError(
            f"parallel.ep_token_dispatcher={backend!r} is "
            "a registered gap, not a supported backend: its external kernels "
            "and distributed wrappers are not included in this build. Use "
            "'alltoall'."
        )
    if backend == "mc2":
        raise EnvironmentUnsupportedError(
            "parallel.ep_token_dispatcher='mc2' requires the Ascend fused "
            "all-to-all/grouped-matmul operators. This build has no compatible "
            "MC2 operator registration; refusing an unfused silent fallback."
        )


@dataclass(kw_only=True, slots=True)
class ExpertParallelConfig:
    """Execution policy for EP; parallel degrees live in ParallelConfig."""

    router_aux_loss_coef: float | None = None
    """Override the HF MoE auxiliary-loss coefficient; None preserves the model
    setting.
    """

    moe_quantile_balancing: bool = False
    """Quantile-balanced routing; requires sigmoid scores, unrestricted groups, and EP
    > 1.
    """

    fixed_router: bool = False
    """Debug-only deterministic round-robin routing."""

    ep_token_dispatcher: str = "alltoall"
    """EP backend: alltoall, torchao, or domino. Other registered backends are
    unavailable.
    """

    ep_torchao_pad_multiple: int = 16
    """Token padding for torchao grouped GEMMs: typically 16 (FP8) or 32 (MXFP8)."""

    def __post_init__(self) -> None:
        validate_expert_dispatcher(self.ep_token_dispatcher)
        require_integer_at_least(
            self, "ep_torchao_pad_multiple", group="parallel.ep_config"
        )
        if self.fixed_router and self.moe_quantile_balancing:
            raise UnsupportedCombinationError(
                "parallel.fixed_router cannot be combined with moe_quantile_balancing"
            )


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
        # Keep schedule discovery lazy to avoid importing pipelining at import time.
        from torch.distributed.pipelining.schedules import get_schedule_class

        try:
            get_schedule_class(self.pipeline_parallel_schedule)
        except ValueError as exc:
            raise ConfigError(
                "Invalid parallel.pipeline_parallel_schedule "
                f"{self.pipeline_parallel_schedule!r}: {exc}"
            ) from exc


@dataclass(kw_only=True, slots=True)
class ParallelConfig:
    """Parallel degrees composed with TP/FSDP/EP/CP/PP execution plans."""

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    tp_config: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    pp_config: PipelineParallelConfig = field(default_factory=PipelineParallelConfig)
    cp_config: ContextParallelConfig = field(default_factory=ContextParallelConfig)
    ep_config: ExpertParallelConfig = field(default_factory=ExpertParallelConfig)

    data_parallel_replicate_size: int = 1
    """Replication degree; > 1 enables DDP, or HSDP with DP sharding > 1."""

    data_parallel_shard_size: int = -1
    """Sharding degree; -1 derives leftover ranks, 1 disables sharding."""

    tensor_parallel_size: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    pipeline_parallel_size: int = 1
    """Physical pipeline ranks; virtual stages are inferred from the split and
    schedule.
    """

    context_parallel_kv_allgather_size: int = 1
    """Independent KV AllGather CP axis degree."""

    context_parallel_ulysses_size: int = 1
    """Independent Ulysses CP axis degree."""

    context_parallel_kcp_size: int = 1
    """KCP degree for pure KDA models; requires strategy='kcp'."""

    expert_parallel_size: int = 1
    """EP degree for MoE. Tiles the dense DP-shard × CP × TP region; EP = 1 permits
    pure TP.
    """

    train_timeout_seconds: int = 100
    """Process-group timeout applied after the first training step, in seconds."""

    def non_dp_sizes(self) -> int:
        """Product of fixed world-mesh degrees: dp_replicate*tp*pp*cp."""
        return (
            self.data_parallel_replicate_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.cp_size
        )

    def derive_dp(self, world_size: int) -> int:
        """Resolve ``data_parallel_shard_size`` against world_size."""
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size < 1
        ):
            raise ConfigError(
                f"world_size must be a positive integer, got {world_size!r}"
            )
        fixed = self.non_dp_sizes()
        if self.data_parallel_shard_size == -1:
            if world_size % fixed != 0:
                raise ConfigError(
                    f"world_size={world_size} not divisible by "
                    f"dp_replicate*tp*pp*cp={fixed}"
                )
            return world_size // fixed
        dp_shard = self.data_parallel_shard_size
        if dp_shard * fixed != world_size:
            raise ConfigError(
                f"dp_shard*dp_replicate*tp*pp*cp = {dp_shard * fixed} "
                f"!= world_size={world_size}"
            )
        return dp_shard

    def __post_init__(self):
        for name, expected in (
            ("fsdp_config", FSDPConfig),
            ("tp_config", TensorParallelConfig),
            ("pp_config", PipelineParallelConfig),
            ("cp_config", ContextParallelConfig),
            ("ep_config", ExpertParallelConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ConfigError(f"parallel.{name} must be {expected.__name__}")
        require_integer_at_least(
            self,
            "train_timeout_seconds",
            "data_parallel_replicate_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_kv_allgather_size",
            "context_parallel_ulysses_size",
            "context_parallel_kcp_size",
            "expert_parallel_size",
            group="parallel",
        )
        if (
            not isinstance(self.data_parallel_shard_size, int)
            or isinstance(self.data_parallel_shard_size, bool)
            or (
                self.data_parallel_shard_size < 1
                and self.data_parallel_shard_size != -1
            )
        ):
            raise ConfigError(
                "parallel.data_parallel_shard_size must be >= 1 or -1 "
                f"(derive), got {self.data_parallel_shard_size}"
            )
        for name in ("fsdp_config", "tp_config", "pp_config", "cp_config", "ep_config"):
            getattr(self, name).__post_init__()
        if (
            self.tensor_parallel_size > 1
            and not self.tp_config.enable_sequence_parallel
        ):
            raise UnsupportedCombinationError(
                "parallel.enable_sequence_parallel=False is not supported: "
                "llmtuner's tensor parallelism is sequence-parallel by "
                "construction. Leave it true, or set tensor_parallel_size=1."
            )
        if (
            self.context_parallel_kcp_size > 1
            and self.cp_config.context_parallel_strategy != "kcp"
        ):
            raise ConfigError("context_parallel_kcp_size > 1 requires strategy='kcp'")
        if self.cp_config.context_parallel_strategy == "kcp":
            if self.tensor_parallel_size != 1:
                raise UnsupportedCombinationError("KCP requires tensor_parallel_size=1")
            if (
                self.context_parallel_kv_allgather_size != 1
                or self.context_parallel_ulysses_size != 1
            ):
                raise UnsupportedCombinationError(
                    "KCP cannot be combined with KV-AllGather or Ulysses CP axes"
                )
        if (
            self.context_parallel_ulysses_size > 1
            and self.cp_config.context_parallel_load_balancer is not None
        ):
            raise UnsupportedCombinationError(
                "the Ulysses CP axis requires context_parallel_load_balancer=None"
            )
        if (
            self.ep_config.ep_token_dispatcher != "alltoall"
            and self.expert_parallel_size == 1
        ):
            raise UnsupportedCombinationError(
                f"parallel.ep_token_dispatcher={self.ep_config.ep_token_dispatcher!r} "
                "has no effect at expert_parallel_size=1; set "
                "expert_parallel_size > 1 or use 'alltoall'."
            )

    # Short aliases for the torchtitan-spelled degree fields.
    @property
    def dp_size(self) -> int:
        return self.data_parallel_shard_size

    @property
    def tp_size(self) -> int:
        return self.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        return self.pipeline_parallel_size

    @property
    def cp_size(self) -> int:
        return (
            self.context_parallel_kv_allgather_size
            * self.context_parallel_ulysses_size
            * self.context_parallel_kcp_size
        )

    @property
    def ep_size(self) -> int:
        return self.expert_parallel_size
