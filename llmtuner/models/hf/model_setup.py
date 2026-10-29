"""Construction-time validation and adaptation for Hugging Face models."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from ...config.model_config import LM_HEAD_COMPUTE_DTYPES
from ...utils.dtypes import get_dtype
from ...utils.logging import get_logger
from ..common.cast_linear import build_cast_linear
from ..common.moe.gate import RouterGateLinearFunction
from ..common.scatter_add import deterministic_scatter_add

logger = get_logger(__name__)

# Config attributes that mean "this model routes tokens to a pool of experts".
# Spelled differently per family (Mixtral/GraniteMoe, Qwen2/3-MoE, DeepSeek), so
# the probe accepts any of them rather than guessing one family's name.
MOE_CONFIG_ATTRIBUTES = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)


def is_moe_config(config: PretrainedConfig) -> bool:
    """Whether ``config`` declares a mixture-of-experts model.

    Structurally probed, like the rest of llmtuner's HF adaptation, so a new
    family only has to carry one of the usual attribute names.
    """
    for name in MOE_CONFIG_ATTRIBUTES:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _isolate_norm_input(module, args, kwargs):
    if args:
        value = args[0]
        return (value.view_as(value), *args[1:]), kwargs
    value = kwargs["hidden_states"]
    return args, {**kwargs, "hidden_states": value.view_as(value)}


def forward_torch_rms_norm(module: nn.Module, hidden_states: torch.Tensor):
    return torch.nn.functional.rms_norm(
        hidden_states,
        (module.weight.numel(),),
        module.weight,
        module.variance_epsilon,
    )


def stabilize_residual_norm_gradients(model: nn.Module) -> None:
    """Select normalization arithmetic and known residual gradient boundaries."""
    model_type = getattr(model.config, "model_type", None)
    backend = getattr(model.config, "norm_backend", "native")
    if backend not in ("native", "torch", "aten", "joint"):
        raise ValueError(
            "model.norm_backend must be 'native', 'torch', 'aten', or 'joint'"
        )
    if backend == "joint" and model_type != "qwen3":
        raise ValueError("model.norm_backend='joint' requires dense Qwen3")
    if backend in ("aten", "torch") and model_type not in ("qwen3", "qwen3_moe"):
        raise ValueError(f"model.norm_backend='{backend}' requires qwen3 or qwen3_moe")
    if model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        norm_type = Qwen3RMSNorm
    elif model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

        norm_type = Qwen3MoeRMSNorm
    else:
        return
    # Remove only adaptations owned here when reapplying construction setup.
    # User hooks and instance-specific forward implementations remain intact.
    for module in model.modules():
        if type(module) is not norm_type:
            continue
        for key, hook in list(module._forward_pre_hooks.items()):
            if hook is _isolate_norm_input:
                del module._forward_pre_hooks[key]
                module._forward_pre_hooks_with_kwargs.pop(key, None)
        if (
            backend != "aten"
            and "forward" in module.__dict__
            and getattr(module.forward, "__module__", None)
            == f"{__package__}.normalization"
        ):
            from .normalization import forward_aten_rms_norm

            if getattr(module.forward, "__func__", None) is forward_aten_rms_norm:
                del module.forward
        if backend != "torch" and getattr(module.forward, "__func__", None) is (
            forward_torch_rms_norm
        ):
            del module.forward
        if backend != "joint" and getattr(module.forward, "__module__", None) == (
            f"{__package__}.residual_normalization"
        ):
            from .residual_normalization import forward_joint_residual_norm

            if getattr(module.forward, "__func__", None) is forward_joint_residual_norm:
                del module.forward
    if backend != "joint":
        for layer in model.model.layers:
            if getattr(layer.forward, "__module__", None) == (
                f"{__package__}.residual_normalization"
            ):
                from .residual_normalization import forward_joint_residual_block

                if (
                    getattr(layer.forward, "__func__", None)
                    is forward_joint_residual_block
                ):
                    del layer.forward
    if backend in ("aten", "torch"):
        from types import MethodType

        if backend == "aten":
            from .normalization import forward_aten_rms_norm as norm_forward
        else:
            norm_forward = forward_torch_rms_norm
        for module in model.modules():
            if type(module) is norm_type:
                module.forward = MethodType(norm_forward, module)
        return
    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            if type(norm) is norm_type and _isolate_norm_input not in (
                norm._forward_pre_hooks.values()
            ):
                norm.register_forward_pre_hook(_isolate_norm_input, with_kwargs=True)
    if backend == "joint":
        from .residual_normalization import apply_joint_residual_norm

        apply_joint_residual_norm(model)


def forward_fp32_topk_router(module: nn.Module, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, module.hidden_dim)
    router_logits = RouterGateLinearFunction.apply(hidden_states, module.weight)
    probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
    scores, indices = torch.topk(probabilities, module.top_k, dim=-1, sorted=False)
    if module.norm_topk_prob:
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return router_logits, scores, indices


def stabilize_moe_router(model: nn.Module) -> None:
    if getattr(model.config, "model_type", None) != "qwen3_moe":
        return
    from types import MethodType

    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

    for module in model.modules():
        if type(module) is Qwen3MoeTopKRouter:
            module.forward = MethodType(forward_fp32_topk_router, module)


def resolve_submodule_name(
    module: nn.Module, names: tuple[str, ...], *, component: str
) -> str:
    """Return the first candidate attribute holding an nn.Module.

    HF names the same submodule differently across model families
    (``embed_tokens``/``wte``, ``norm``/``final_layernorm``/``ln_f``), so the
    wrapper probes instead of hardcoding one family's spelling. Resolved once at
    construction: a missing part should fail at build time, not mid-forward.
    """
    for name in names:
        if isinstance(getattr(module, name, None), nn.Module):
            return name
    raise AttributeError(
        f"{type(module).__name__} has no {component} module under any of {names}. "
        "Add the model's spelling to the probe in HFTransformerModel.__init__."
    )


def validate_head_config(config: PretrainedConfig) -> None:
    """Reject GQA head counts that no attention kernel can honor."""
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    for name, value in (
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv_heads),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if (
        num_heads is not None
        and num_kv_heads is not None
        and num_heads % num_kv_heads != 0
    ):
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"num_key_value_heads ({num_kv_heads})"
        )


def forward_torchtitan_moe(module: nn.Module, hidden_states: torch.Tensor):
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("experts_implementation='torchtitan' requires BF16 inputs")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    _, scores, expert_ids = module.gate(hidden_states)
    counts = torch.nn.functional.one_hot(
        expert_ids, num_classes=module.experts.num_experts
    ).sum(dim=(0, 1))
    dispatcher = module._llmtuner_local_dispatcher
    routed, counts, metadata = dispatcher.dispatch(
        hidden_states, scores, expert_ids, counts
    )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    gate, up = module.experts.gate_up_proj.chunk(2, dim=1)
    gate = gate.contiguous()
    up = up.contiguous()
    gate_values = torch._grouped_mm(
        routed.bfloat16(), gate.bfloat16().transpose(-2, -1), offs=offsets
    )
    up_values = torch._grouped_mm(
        routed.bfloat16(), up.bfloat16().transpose(-2, -1), offs=offsets
    )
    hidden = module.experts.act_fn(gate_values) * up_values
    hidden = (hidden.float() * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)).to(
        hidden.dtype
    )
    routed_output = torch._grouped_mm(
        hidden,
        module.experts.down_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(routed)
    output = deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, hidden_dim),
        routed_output,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)


def stabilize_moe_experts(model: nn.Module) -> None:
    if getattr(model.config, "experts_implementation", "native") != "torchtitan":
        return
    from types import MethodType

    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    from ...parallel.expert_parallel.token_dispatch import LocalTokenDispatcher

    modules = [
        module for module in model.modules() if type(module) is Qwen3MoeSparseMoeBlock
    ]
    if not modules:
        raise ValueError("Qwen3-MoE torchtitan experts require routed MoE blocks")
    for module in modules:
        module._llmtuner_local_dispatcher = LocalTokenDispatcher(
            module.experts.num_experts, module.gate.top_k
        )
        module.forward = MethodType(forward_torchtitan_moe, module)


def configure_experts_implementation(
    config: PretrainedConfig, *, model_cls: type
) -> None:
    """Set the requested HF experts kernel on the construction config.

    ``native`` preserves the model's built-in kernel. Other choices require
    Transformers' settable experts implementation API. Under EP, llmtuner
    replaces the MoE block after construction.

    Transformers before 5.x has no such API: its MoE blocks always run HF's
    per-expert ``F.linear`` loop, which is exactly the kernel ``"eager"`` names.
    That request is therefore already satisfied there and is accepted as a
    no-op; ``"grouped_mm"`` / ``"batched_mm"`` name kernels this build does not
    carry, so they still fail rather than silently run something else.
    """
    impl = getattr(config, "experts_implementation", "native")
    if impl == "native":
        return
    if impl == "torchtitan":
        if config.model_type != "qwen3_moe":
            raise ValueError("experts_implementation='torchtitan' requires Qwen3-MoE")
        config._experts_implementation = "eager"
        return
    if impl not in ("grouped_mm", "batched_mm", "eager"):
        raise ValueError(
            f"experts_implementation must be one of 'native', "
            f"'grouped_mm', 'batched_mm', 'eager', 'torchtitan', got '{impl}'"
        )
    can_set = getattr(model_cls, "_can_set_experts_implementation", None)
    if can_set is None:
        # No settable-experts API in this transformers. Only the request the
        # built-in kernel already satisfies can be honored.
        if impl == "eager" and is_moe_config(config):
            logger.info(
                "%s has no settable experts implementation (transformers < 5); "
                "its built-in experts path is the per-expert loop, so "
                "experts_implementation='eager' is already in effect.",
                model_cls.__name__,
            )
            return
        raise ValueError(
            f"{model_cls.__name__} does not support a settable experts "
            f"implementation, so experts_implementation='{impl}' cannot be "
            "honored. Set experts_implementation='native' to use the HF "
            "model's built-in experts kernel. This transformers build has no "
            "_can_set_experts_implementation at all -- that API first ships "
            "with transformers 5, and below it a MoE block always runs the "
            "per-expert loop, which is what 'eager' names."
        )
    if not can_set():
        raise ValueError(
            f"{model_cls.__name__} does not support a settable experts "
            f"implementation, so experts_implementation='{impl}' cannot "
            "be honored. Set experts_implementation='native' to use "
            "the HF model's built-in experts kernel."
        )
    config._experts_implementation = impl


def apply_cast_lm_head(model: nn.Module, config: PretrainedConfig) -> None:
    """Optional fixed-dtype lm_head (torchtitan's CastLinear semantics):
    score the vocabulary logits in the requested dtype while the stored
    weight keeps its own. Swapped in place rather than wrapped so the
    state-dict FQNs -- and a weight tie with the embedding -- are
    untouched (see models/common/cast_linear.py for why subclassing
    beats wrapping)."""
    compute_dtype = getattr(config, "compute_dtype", None)
    if compute_dtype is None:
        return
    if compute_dtype not in LM_HEAD_COMPUTE_DTYPES:
        raise ValueError(
            f"compute_dtype must be one of {sorted(LM_HEAD_COMPUTE_DTYPES)}, "
            f"got {compute_dtype!r}"
        )
    if model.lm_head is not None:
        model.lm_head = build_cast_linear(
            model.lm_head, compute_dtype=get_dtype(compute_dtype)
        )
