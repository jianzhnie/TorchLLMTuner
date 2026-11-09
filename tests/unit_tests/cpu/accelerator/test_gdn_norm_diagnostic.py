"""Pure report/validation checks without importing reference or NPU backends."""

import ast
from pathlib import Path
from types import MethodType

import pytest
import torch

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeRMSNormGated,
    )
except ImportError:
    Qwen3_5MoeRMSNormGated = None

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "integration_tests/qwen35_gdn_norm_alignment.py"
)


@pytest.fixture(scope="module")
def helpers():
    # The executable intentionally depends on the dedicated reference env.
    # Extract only its pure helpers, never substituting a numerical backend.
    tree = ast.parse(SCRIPT.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("channel_errors", "verify_tensor", "forward_reference_cast_order")
    ]
    namespace = {"torch": torch, "ATOL": 0.005, "RTOL": 0.02}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(SCRIPT), "exec"),
        namespace,
    )
    return namespace


def test_small_cancellation_channel_failure_is_retained(helpers):
    hf = torch.ones(128, dtype=torch.bfloat16)
    reference = hf.clone()
    hf[35], reference[35] = 0.01, 0.025625
    rows = helpers["channel_errors"](hf, reference)
    assert len(rows) == 128
    assert [row["channel"] for row in rows if not row["passed"]] == [35]
    assert rows[35]["absolute_error"] > rows[35]["limit"]


def test_negative_channel_uses_absolute_relative_scale(helpers):
    hf = torch.full((128,), -2.671875, dtype=torch.bfloat16)
    reference = torch.full_like(hf, -2.65625)
    rows = helpers["channel_errors"](hf, reference)
    assert all(row["passed"] for row in rows)
    assert rows[0]["absolute_error"] == 0.015625


@pytest.mark.parametrize(
    "value", [torch.empty(0), torch.zeros(128), torch.tensor([float("nan")])]
)
def test_invalid_evidence_rejected(helpers, value):
    with pytest.raises(AssertionError):
        helpers["verify_tensor"](value)


@pytest.mark.skipif(Qwen3_5MoeRMSNormGated is None, reason="HF Qwen3.5 unavailable")
@pytest.mark.parametrize("valid_tokens", [93, 128])
def test_cast_order_uses_unrounded_activation_for_weight_vjp(helpers, valid_tokens):
    generator = torch.Generator().manual_seed(42)
    shape = (4096, 128)
    x = torch.randn(shape, generator=generator).bfloat16().requires_grad_()
    z = torch.randn(shape, generator=generator).bfloat16().requires_grad_()
    dy = torch.randn(shape, generator=generator).bfloat16()
    dy.reshape(1, 128, 32, 128)[:, valid_tokens:] = 0
    native = Qwen3_5MoeRMSNormGated(128).bfloat16()
    candidate = Qwen3_5MoeRMSNormGated(128).bfloat16()
    weight = candidate.weight
    candidate.forward = MethodType(helpers["forward_reference_cast_order"], candidate)
    assert candidate.weight is weight and weight.dtype == torch.bfloat16
    assert set(candidate.state_dict()) == {"weight"}
    a, b = native(x, z), candidate(x, z)
    assert torch.equal(a, b)  # Ones weight hides the forward cast-order difference.
    an = torch.autograd.grad(a, (x, z, native.weight), dy)
    bc = torch.autograd.grad(b, (x, z, candidate.weight), dy)
    assert torch.equal(an[0], bc[0]) and torch.equal(an[1], bc[1])
    assert not torch.equal(an[2], bc[2])
    # Independent chain-rule witness: the cotangent crosses BF16 before the
    # weight multiply, but the candidate's weight VJP uses unrounded norm(x).
    normalized = x.detach().float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + 1e-6
    )
    cotangent = dy.float() * torch.nn.functional.silu(z.detach().float())
    cotangent = cotangent.bfloat16().float()
    expected_dw = (cotangent * normalized).sum(0).bfloat16()
    assert torch.equal(bc[2], expected_dw)
