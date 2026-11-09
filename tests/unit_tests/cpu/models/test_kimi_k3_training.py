"""Actual pinned official HF classes; no numerical stubs or upstream edits.

Explicit optional source/dependency environment is needed for this opt-in test:
KIMI_K3_OFFICIAL_SOURCE=/path/to/reviewed/source pytest -q <this file>
An explicitly requested source with incompatible dependencies fails, not skips.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

from llmtuner.models.hf.kimi_k3_training import (
    OFFICIAL_REVISION,
    enable_official_k3_moe_training,
)


@pytest.fixture
def official():
    source = os.environ.get("KIMI_K3_OFFICIAL_SOURCE")
    if not source:
        pytest.skip("Set KIMI_K3_OFFICIAL_SOURCE for real official HF class tests")
    root = Path(source).resolve()
    sys.path.insert(0, str(root.parent))
    try:
        yield importlib.import_module(f"{root.name}.modeling_kimi_linear")
    finally:
        sys.path.remove(str(root.parent))


@pytest.mark.parametrize("empty_experts", [False, True])
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("activation", ["sigmoid", "softmax"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_official_moe_training_vjp(official, latent, activation, dtype, empty_experts):
    torch.manual_seed(123)
    config = official.KimiLinearConfig(
        hidden_size=8,
        intermediate_size=12,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        num_expert_group=2,
        topk_group=1,
        moe_intermediate_size=6,
        moe_router_activation_func=activation,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
        hidden_act="situ",
        activation_situ_beta=1.7,
        activation_situ_linear_beta=2.3,
        routed_scaling_factor=1.4,
    )
    module = official.KimiSparseMoeBlock(config).to(dtype).eval()
    # Explicit test-only initialization policy, not asserted to be official init.
    with torch.no_grad():
        module.gate.e_score_correction_bias.copy_(
            torch.tensor([0.11, -0.07, 0.03, -0.13], dtype=dtype)
        )
    if empty_experts:
        # Choose only the first group; keep its two scores unequal and live.
        with torch.no_grad():
            module.gate.e_score_correction_bias.copy_(
                torch.tensor([10.0, 9.0, -10.0, -9.0], dtype=dtype)
            )
    parameters = dict(module.named_parameters())
    state = {k: v.clone() for k, v in module.state_dict().items()}
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    module.train()
    with pytest.raises(AssertionError):
        module.gate(x)
    module.eval()
    ids, scores = module.gate(x)
    selected = ids.unique().tolist()
    assert selected
    if empty_experts:
        assert selected == [0, 1]
    flat = x.reshape(-1, 8)
    if latent:
        flat = module.routed_expert_down_proj(flat)
    y = type(module).moe_infer.__wrapped__(module, flat, ids, scores)
    if latent:
        y = module.routed_expert_up_proj(module.routed_expert_norm(y))
    expected = y.view_as(x) + module.shared_experts(x)
    expected_grads = torch.autograd.grad(
        expected, (x, *parameters.values()), dy, allow_unused=True
    )
    evidence = enable_official_k3_moe_training(
        module,
        revision=OFFICIAL_REVISION,
        correction_bias_provenance="test asymmetric constants; not official init",
    )
    module.train()
    actual_ids, actual_scores = module.gate(x)
    assert torch.equal(ids, actual_ids)
    assert torch.equal(scores, actual_scores)
    assert actual_scores.dtype == torch.float32
    actual = module(x)
    actual_grads = torch.autograd.grad(
        actual, (x, *parameters.values()), dy, allow_unused=True
    )
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
    assert torch.equal(expected, actual)
    for left, right in zip(expected_grads, actual_grads, strict=True):
        if left is None:
            assert right is None
        else:
            assert torch.isfinite(right).all() and right.numel() > 0
            assert torch.equal(left, right)
    assert torch.count_nonzero(actual_grads[0]) > 0
    grads = dict(zip(parameters, actual_grads[1:], strict=True))
    for name in ("gate.weight", "shared_experts.gate_proj.weight"):
        assert grads[name] is not None and torch.count_nonzero(grads[name]) > 0
    for index in selected:
        for projection in ("w1", "w2", "w3"):
            grad = grads[f"experts.{index}.{projection}.weight"]
            assert grad is not None and torch.count_nonzero(grad) > 0
    for index in set(range(config.num_experts)) - set(selected):
        for projection in ("w1", "w2", "w3"):
            grad = grads[f"experts.{index}.{projection}.weight"]
            assert grad is None or torch.count_nonzero(grad) == 0
    assert grads["gate.e_score_correction_bias"] is None
    if latent:
        for name in ("routed_expert_down_proj.weight", "routed_expert_up_proj.weight"):
            assert grads[name] is not None and torch.count_nonzero(grads[name]) > 0
    assert dict(module.named_parameters()).keys() == parameters.keys()
    for name, parameter in module.named_parameters():
        assert parameter is parameters[name]
        assert torch.equal(parameter, state[name])
    assert evidence["adapted_moe_modules"] == 1
    with pytest.raises(TypeError):
        module(x, past_key_values=object())
    with pytest.raises(ValueError, match="contiguous"):
        module(x.transpose(0, 1))
    module.ep_size = 2
    with pytest.raises(ValueError, match="EP"):
        module(x)


def test_adapter_rejects_unpinned_source():
    with pytest.raises(ValueError, match="Pinned revision"):
        enable_official_k3_moe_training(
            torch.nn.Linear(2, 2), revision="unknown", correction_bias_provenance="x"
        )
    with pytest.raises(ValueError, match="No official"):
        enable_official_k3_moe_training(
            torch.nn.Linear(2, 2),
            revision=OFFICIAL_REVISION,
            correction_bias_provenance="explicit",
        )
