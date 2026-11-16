"""TP declarations and pattern normalization without model mutation.

Pattern order is significant: a replicated first match prevents later broad
rules from sharding that target. Structural resolution consumes these choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from torch import nn

from ...config.parallel_config import ParallelConfig
from ..module_patterns import module_name_matches, resolve_module_patterns
from ..plan import unique_module_bindings
from .linear import ColumnParallelLinear, RowParallelLinear

ShardKind = Literal["colwise", "rowwise"]
MOE_PLAN_SPECS = frozenset({"packed_colwise", "packed_rowwise", "moe_tp_experts"})


@dataclass(frozen=True)
class ShardingConfig:
    """How one projection is sharded on the TP axis.

    Attributes:
        kind: ``"colwise"`` or ``"rowwise"`` -- the direction the weight is cut.
        implementation: the module class that realizes this declaration. Paired
            here rather than looked up from ``kind`` so a caller can swap in a
            different realizer (e.g. a fused-QKV variant) without touching the
            engine. Every realizer declares shard_dim and sequence_transform on
            its class and preserves them on the instance. It must register exactly
            one parameter, weight, with the declared shard shape and unchanged
            dtype/device/trainability.
        sequence_transform: whether the projection gathers, scatters or retains
            its input token rows. Weight shard direction alone does not specify
            the activation layout, especially for attention-local projections.
    """

    kind: ShardKind
    implementation: type[nn.Module]
    sequence_transform: Literal["gather", "scatter", "preserve"]

    def __post_init__(self) -> None:
        if self.kind not in ("colwise", "rowwise"):
            raise ValueError(f"Unknown shard kind {self.kind!r}")
        if self.sequence_transform not in ("gather", "scatter", "preserve"):
            raise ValueError("TP projection must declare its sequence transformation")
        if not isinstance(self.implementation, type) or not issubclass(
            self.implementation, nn.Module
        ):
            raise TypeError("TP projection implementation must be an nn.Module class")
        self.validate_implementation(self.implementation)

    def validate_implementation(
        self, implementation: type[nn.Module] | nn.Module
    ) -> None:
        """Require the same explicit layout declaration for every realizer.

        Class declarations are checked before construction; instance declarations
        are checked before installing replacements. This verifies the contract,
        not the numerical correctness of a custom forward implementation.
        """
        if implementation is not self.implementation and not isinstance(
            implementation, self.implementation
        ):
            raise TypeError("TP constructor returned a different projection type")
        dim = getattr(implementation, "shard_dim", None)
        if type(dim) is not int or dim != (0 if self.kind == "colwise" else 1):
            raise ValueError("TP shard kind disagrees with projection implementation")
        if (
            getattr(implementation, "sequence_transform", None)
            != self.sequence_transform
        ):
            raise ValueError(
                "TP sequence transform disagrees with projection implementation"
            )


def colwise() -> ShardingConfig:
    """Output features split; activations stay feature-sharded after the GEMM."""
    return ShardingConfig(
        kind="colwise",
        implementation=ColumnParallelLinear,
        sequence_transform="gather",
    )


def rowwise() -> ShardingConfig:
    """Input features split; activations reduce-scatter back to a sequence shard."""
    return ShardingConfig(
        kind="rowwise",
        implementation=RowParallelLinear,
        sequence_transform="scatter",
    )


def _require_tp_plan(plan: object) -> Mapping[str, str | ShardingConfig]:
    """Reject malformed declarations instead of treating false values as absent."""
    if not isinstance(plan, Mapping):
        raise TypeError(
            "TP plan must be a mapping of module patterns to sharding specs"
        )
    if any(not isinstance(pattern, str) for pattern in plan):
        raise TypeError("TP plan module patterns must be strings")
    return plan


def model_tp_config(model: nn.Module) -> Mapping[str, str | ShardingConfig]:
    """Read the optional model-owned declaration without truth-value fallback."""
    plan = getattr(model, "tp_config", None)
    return {} if plan is None else _require_tp_plan(plan)


def resolve_tp_sharding_plan(
    model: nn.Module, *, plan: Mapping[str, str | ShardingConfig] | None = None
) -> dict[str, ShardingConfig | None]:
    """Normalize an explicit or model-declared TP plan for dense projections.

    Patterns retain insertion order. Colwise/rowwise specs become ShardingConfig;
    replicated specs become None so matching leaves their modules intact.
    Replicated projection gradients are reduced by the trainer.

    MoE specs also become None: they name stacked expert parameters rather than
    nn.Linear modules. resolve_tp_targets reads the raw plan to select the
    structural MoE path. EP specs and unsupported replicated-activation layouts
    raise instead of silently applying sequence-parallel realizers.
    """
    plan = model_tp_config(model) if plan is None else _require_tp_plan(plan)
    resolved: dict[str, ShardingConfig | None] = {}
    for pattern, spec in plan.items():
        if isinstance(spec, ShardingConfig):
            resolved[pattern] = spec
        elif spec == "colwise":
            resolved[pattern] = colwise()
        elif spec == "rowwise":
            resolved[pattern] = rowwise()
        elif spec == "embedding_rowwise":
            # Realized by VocabParallelEmbedding; None prevents a broad dense
            # pattern from overriding the first matching vocabulary entry.
            resolved[pattern] = None
        elif spec in {"replicated_with_grad_allreduce", "colwise_gather_output"}:
            # Nothing for apply_tp to do: the projection stays whole on every
            # rank. The ``_with_grad_allreduce`` half is already implemented --
            # allreduce_replicated_tp_grads sums exactly these parameters'
            # gradients -- so this entry only has to be understood, not acted
            # on. Recorded as None rather than dropped so match still stops
            # here instead of falling through to a broader later pattern.
            resolved[pattern] = None
        elif spec in MOE_PLAN_SPECS:
            # MoE-under-TP, realized structurally in apply_tp (weight sharding
            # in shard_experts_for_tp, activation collectives in
            # TPMoeSequenceBoundary). None for
            # the same first-match-wins reason as above, and so a pattern that
            # happens to match an nn.Linear is left whole rather than wrongly
            # swapped for a dense realizer.
            resolved[pattern] = None
        else:
            raise ValueError(
                f"Unsupported TP plan entry for {pattern!r}: {spec!r}. This "
                "engine realizes 'colwise' / 'rowwise' (plus the replicated "
                "'replicated_with_grad_allreduce' / 'colwise_gather_output' and "
                "the MoE specs, which it leaves whole), and refuses HF's "
                "replicated-activation styles: 'colwise_rep' / 'rowwise_rep' "
                "keep the activation full on every rank, which is not the "
                "layout the surrounding SP realizers produce or consume. "
                "Rewrite the plan to 'colwise' / 'rowwise' for a model whose "
                "ops tolerate sequence-parallel activations, or run it without "
                "TP."
            )
    return resolved


def match_tp_sharding_spec(
    module_path: str, *, plan: dict[str, ShardingConfig | None]
) -> ShardingConfig | None:
    """Return the first matching TP specification, preserving plan order.

    A matching ``None`` entry leaves the module replicated and stops lookup;
    later sharding patterns cannot override it. No match also returns ``None``.
    """
    for pattern, spec in plan.items():
        if module_name_matches(module_path, pattern=pattern):
            return spec
    return None


def resolve_vocab_parallel_targets(
    model: nn.Module,
    plan: Mapping[str, str | ShardingConfig],
    *,
    module_bindings: tuple[tuple[str, nn.Module], ...] | None = None,
) -> list[tuple[str, nn.Embedding]]:
    """Resolve HF ``embedding_rowwise`` entries in first-match plan order."""
    modules = (
        module_bindings if module_bindings is not None else tuple(model.named_modules())
    )
    return [
        (path, module)
        for path, module in unique_module_bindings(modules)
        if isinstance(module, nn.Embedding)
        and next(
            (
                spec
                for pattern, spec in plan.items()
                if module_name_matches(path, pattern=pattern)
            ),
            None,
        )
        == "embedding_rowwise"
    ]


def resolve_tp_overrides(model: nn.Module, *, cfg: ParallelConfig) -> dict[str, str]:
    """Validate explicit overrides before splitting or transforming the model."""
    configured_plan = dict.fromkeys(cfg.tp_config.tp_colwise_modules, "colwise")
    for pattern in cfg.tp_config.tp_rowwise_modules:
        if pattern in configured_plan:
            raise ValueError(f"TP pattern {pattern!r} has conflicting styles")
        configured_plan[pattern] = "rowwise"
    if cfg.tp_config.tp_sequence_modules:
        raise ValueError(
            "tp_sequence_modules is not valid for llmtuner's fused sequence "
            "parallel engine; normalization modules already consume the "
            "sequence shard without a separate transformation"
        )
    modules = dict(model.named_modules())
    names = list(modules)
    # Model defaults may be overridden; explicit overrides must agree.
    for name in names:
        matches = [
            (pattern, style)
            for pattern, style in configured_plan.items()
            if module_name_matches(name, pattern=pattern)
        ]
        if len({style for _, style in matches}) > 1:
            raise ValueError(
                f"TP override patterns have conflicting styles for module "
                f"{name!r}: {matches}"
            )
    resolve_module_patterns(
        names, patterns=list(configured_plan), plan_name="TP override"
    )
    for pattern in configured_plan:
        matches = [
            module
            for name, module in modules.items()
            if module_name_matches(name, pattern=pattern)
        ]
        if matches and not any(isinstance(module, nn.Linear) for module in matches):
            raise ValueError(
                f"TP override {pattern!r} matches no nn.Linear projection; "
                "colwise/rowwise overrides require dense projection targets"
            )
    return configured_plan


def resolve_effective_tp_plan(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    plan: Mapping[str, str | ShardingConfig] | None = None,
) -> dict[str, str | ShardingConfig]:
    """Merge overrides once for dense, expert and vocabulary target resolution."""
    overrides = resolve_tp_overrides(model, cfg=cfg)
    declared = model_tp_config(model) if plan is None else _require_tp_plan(plan)
    # First matching pattern wins, so configuration overrides precede defaults.
    return {
        **overrides,
        **{path: spec for path, spec in declared.items() if path not in overrides},
    }
