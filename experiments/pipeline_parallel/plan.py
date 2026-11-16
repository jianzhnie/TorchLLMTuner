"""Project validated global plans onto experimental pipeline ownership."""

from torch import nn

from llmtuner.parallel.expert_parallel.model_replacement import MoEConversionTarget
from llmtuner.parallel.plan import ParallelPlan, PlanTarget


def validate_pipeline_targets(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> tuple[tuple[PlanTarget, PlanTarget], ...]:
    """Validate retained PP ownership and return global/local metadata pairs.

    Missing modules and removed-stage Identity placeholders are nonlocal targets.
    The returned global bindings are only references for semantic comparison;
    execution must consume the independently resolved local plan.
    """
    local_plan.validate_binding(model)
    label = f"Pipeline {global_plan.stage.upper()}"
    # validate_binding checked the complete module table when supplied.
    modules = dict(
        model.named_modules(remove_duplicate=False)
        if local_plan.module_bindings is None
        else local_plan.module_bindings
    )
    expected = {}
    for target in global_plan.targets:
        module = modules.get(target.path)
        if module is None or (
            isinstance(module, nn.Identity)
            and not isinstance(target.module, nn.Identity)
        ):
            continue
        key = (target.path, target.kind)
        if key in expected:
            raise ValueError(f"{label} global plan contains duplicate targets")
        expected[key] = target
    actual = {(target.path, target.kind): target for target in local_plan.targets}
    if len(actual) != len(local_plan.targets):
        raise ValueError(f"{label} local plan contains duplicate targets")
    if expected.keys() != actual.keys():
        raise ValueError(
            f"{label} target ownership changed: "
            f"missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    return tuple((expected[key], target) for key, target in actual.items())


def validate_pipeline_cp_plan(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> None:
    """Check PP-local CP targets and layouts before other stages mutate them."""
    if (
        global_plan.stage != "cp"
        or local_plan.stage != "cp"
        or global_plan.degree != local_plan.degree
        or global_plan.implementation != local_plan.implementation
    ):
        raise ValueError("Pipeline CP plans must describe the same execution")
    for original, target in validate_pipeline_targets(global_plan, local_plan, model):
        if target.layout != original.layout:
            raise ValueError(f"Pipeline CP layout changed: {target.path}")


def pipeline_ep_semantics(target: PlanTarget) -> tuple[object, ...]:
    """Describe conversion mathematics without instance-specific references.

    PP deepcopy changes module/parameter identity. Routing values and packed
    geometry must remain unchanged until TP starts slicing those parameters.
    """
    binding = target.spec
    if target.kind != "moe" or not isinstance(binding, MoEConversionTarget):
        raise ValueError("Pipeline EP requires resolved MoE conversion targets")
    routing = binding.routing
    return (
        target.layout,
        binding.attribute,
        binding.router_logits,
        binding.contract,
        binding.model_math,
        binding.model_conversion,
        binding.declared_slot,
        None if binding.packed_experts is None else binding.packed_experts.shape,
        None
        if routing is None
        else (
            routing.top_k,
            routing.score_func,
            routing.route_norm,
            routing.route_scale,
            routing.num_expert_groups,
            routing.num_limited_groups,
            routing.shared is not None,
            routing.shared_gate is not None,
        ),
    )


def validate_pipeline_ep_plan(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> None:
    """Validate freshly bound PP-local EP ownership before any TP mutation.

    Removed stages are absent or Identity placeholders. Retained MoE paths must
    keep their globally checked layout and routing; global references are never
    reused for local execution. Resolve again after TP changes parameter layout.
    """
    if (
        global_plan.stage != "ep"
        or local_plan.stage != "ep"
        or global_plan.degree != local_plan.degree
        or global_plan.implementation != local_plan.implementation
    ):
        raise ValueError("Pipeline EP plans must describe the same execution")
    for original, target in validate_pipeline_targets(global_plan, local_plan, model):
        if pipeline_ep_semantics(target) != pipeline_ep_semantics(original):
            raise ValueError(f"Pipeline EP conversion semantics changed: {target.path}")


def validate_pipeline_tp_plan(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> None:
    """Check a freshly bound PP-local plan against the unmodified global plan.

    PP preserves registered FQNs and replaces unowned simple modules with
    Identity. Keep those removals distinct from a retained target losing its TP
    transformation. Live bindings and conversion metadata always come from the
    local resolver; global module references must never reach execution.
    """
    if (
        global_plan.stage != "tp"
        or local_plan.stage != "tp"
        or global_plan.degree != local_plan.degree
    ):
        raise ValueError("Pipeline TP plans must describe the same stage and degree")
    for original, target in validate_pipeline_targets(global_plan, local_plan, model):
        key = (target.path, target.kind)
        if target.layout != original.layout or (
            target.kind == "dense" and target.spec != original.spec
        ):
            raise ValueError(
                f"Pipeline TP layout or projection strategy changed: {key}"
            )
    modules = dict(
        model.named_modules(remove_duplicate=False)
        if local_plan.module_bindings is None
        else local_plan.module_bindings
    )
    local_history = dict(local_plan.token_history_requirements)
    for path, requirement in global_plan.token_history_requirements:
        module = modules.get(path)
        if module is None or isinstance(module, nn.Identity):
            continue
        if path not in local_history or (
            type(local_history[path]) is not type(requirement)
            or local_history[path] != requirement
        ):
            raise ValueError(f"Pipeline TP token history requirement changed: {path}")
