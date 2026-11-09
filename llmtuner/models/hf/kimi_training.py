"""Opt-in differentiable combine for official HF Kimi individual experts.

The caller must validate the source class before binding this forward. This
does not load remote code, enable EP, or establish full-model NPU support.
"""

from types import MethodType

import torch
from torch import nn

from ...errors import UnsupportedCombinationError


def install_kimi_moe_training(model: nn.Module, *, moe_type: type[nn.Module]) -> int:
    """Bind reviewed official MoE instances without replacing their parameters.

    The caller supplies the class from the explicitly loaded official module;
    this function neither executes remote code nor identifies it by class name.
    All matching instances are validated before any forward is changed. Exact
    types deliberately exclude subclasses whose forward contracts may differ.
    Call before distributed wrapping or compilation. This installs only MoE
    training, not attention or full-model NPU support.
    """
    if not isinstance(moe_type, type) or not issubclass(moe_type, nn.Module):
        raise TypeError("moe_type must be the reviewed official nn.Module class")
    modules = [module for module in model.modules() if type(module) is moe_type]
    if not modules:
        raise UnsupportedCombinationError("No instances of the supplied Kimi MoE class")
    for module in modules:
        if (
            getattr(module, "ep_size", 1) != 1
            or not isinstance(getattr(module, "experts", None), nn.ModuleList)
            or len(module.experts) != getattr(module, "num_experts", None)
            or any(expert is None for expert in module.experts)
        ):
            raise UnsupportedCombinationError(
                "Official Kimi training requires all individual experts locally"
            )
        gate = getattr(module, "gate", None)
        if gate is None or not all(
            hasattr(gate, name)
            for name in (
                "weight",
                "e_score_correction_bias",
                "moe_router_activation_func",
                "num_expert_group",
                "topk_group",
                "num_experts",
                "top_k",
                "moe_renormalize",
                "routed_scaling_factor",
            )
        ):
            raise UnsupportedCombinationError("Unsupported official Kimi gate contract")
        if gate.moe_router_activation_func not in {"sigmoid", "softmax"}:
            raise UnsupportedCombinationError(
                "Unsupported official Kimi router activation"
            )
    for module in modules:
        module.forward = MethodType(forward_kimi_training, module)
    return len(modules)


def forward_kimi_gate_training(
    module: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official selection arithmetic with gradients through selected scores."""
    inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = torch.nn.functional.linear(inputs.float(), module.weight.float())
    if module.moe_router_activation_func == "sigmoid":
        scores = logits.sigmoid()
    elif module.moe_router_activation_func == "softmax":
        scores = logits.softmax(-1)
    else:
        raise UnsupportedCombinationError("Unsupported official Kimi router activation")
    choices = scores + module.e_score_correction_bias.unsqueeze(0)
    groups = module.num_expert_group
    if groups > module.topk_group:
        group_scores = choices.reshape(
            inputs.shape[0], groups, module.num_experts // groups
        )
        group_scores = group_scores.topk(2, dim=-1).values.sum(-1)
        group_ids = group_scores.topk(module.topk_group, sorted=False).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_ids, 1)
        mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, -1, module.num_experts // groups)
            .reshape_as(choices)
        )
        choices = choices.masked_fill(~mask.bool(), float("-inf"))
    ids = choices.topk(module.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if module.top_k > 1 and module.moe_renormalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return ids, weights * module.routed_scaling_factor


def forward_kimi_training(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Retain official modules and weighted top-k order, including empty experts."""
    if getattr(module, "ep_size", 1) != 1:
        raise UnsupportedCombinationError(
            "Official Kimi training adapter requires unsharded individual experts"
        )
    identity = hidden_states
    ids, scores = forward_kimi_gate_training(module.gate, hidden_states)
    inputs = hidden_states.reshape(-1, hidden_states.shape[-1])
    if module.use_latent_moe:
        inputs = module.routed_expert_down_proj(inputs)
    # Preserve assignment slots so summation follows official top-k ordering,
    # rather than accumulating in expert-index order.
    selected = inputs.new_zeros((*ids.shape, inputs.shape[-1]))
    for index, expert in enumerate(module.experts):
        tokens, slots = (ids == index).nonzero(as_tuple=True)
        selected[tokens, slots] = expert(inputs.index_select(0, tokens))
    output = (selected.to(scores.dtype) * scores.unsqueeze(-1)).sum(1)
    output = output.to(inputs.dtype)
    if module.use_latent_moe:
        if module.latent_moe_use_norm:
            output = module.routed_expert_norm(output)
        output = module.routed_expert_up_proj(output)
    output = output.reshape_as(identity)
    if module.config.num_shared_experts is not None:
        output = output + module.shared_experts(identity)
    return output
