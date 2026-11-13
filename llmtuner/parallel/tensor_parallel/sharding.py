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
            engine.
    """

    kind: ShardKind
    implementation: type[nn.Module]

    def __post_init__(self) -> None:
        if self.kind not in ("colwise", "rowwise"):
            raise ValueError(f"Unknown shard kind {self.kind!r}")


def colwise() -> ShardingConfig:
    """Output features split; activations stay feature-sharded after the GEMM."""
    return ShardingConfig(kind="colwise", implementation=ColumnParallelLinear)


def rowwise() -> ShardingConfig:
    """Input features split; activations reduce-scatter back to a sequence shard."""
    return ShardingConfig(kind="rowwise", implementation=RowParallelLinear)


def model_tp_config(model: nn.Module) -> Mapping[str, str | ShardingConfig]:
    """The TP plan ``model`` declares, or ``{}``.

    Prefers the explicit model's ``tp_config`` property and keeps legacy plan
    attributes only at the external compatibility boundary.
    """
    declared = getattr(model, "tp_config", None)
    if declared is not None:
        return declared
    # Explicit model contracts must not inherit private plans from a reference
    # base class. Those plans are often relative to another module tree and an
    # empty explicit declaration must fail before any sharding occurs.
    if getattr(model, "decoder_structure", None) is not None:
        return {}
    return (
        getattr(model, "_tp_config", None)
        or getattr(model, "_tp_plan", None)
        or getattr(model, "tp_plan", None)
        or {}
    )


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
    if plan is None:
        plan = model_tp_config(model)
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
    model: nn.Module, plan: Mapping[str, str | ShardingConfig]
) -> list[tuple[str, nn.Embedding]]:
    """Resolve HF ``embedding_rowwise`` entries in first-match plan order."""
    return [
        (path, module)
        for path, module in model.named_modules()
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


def resolve_tp_overrides(
    model: nn.Module, *, cfg: ParallelConfig, allow_unmatched: bool = False
) -> dict[str, str]:
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
    names = [name for name, _ in model.named_modules()]
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
    if not allow_unmatched:
        resolve_module_patterns(
            names, patterns=list(configured_plan), plan_name="TP override"
        )
    return configured_plan
