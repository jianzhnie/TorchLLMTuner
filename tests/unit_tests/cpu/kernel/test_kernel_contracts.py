"""Input-driven dispatch, numerical references, and device-independent contracts."""

import logging

import pytest
import torch

from llmtuner.kernel import dispatch_kernel, get_kernel, register_kernel, registry
from llmtuner.kernel.npu import basic as npu_basic
from llmtuner.kernel.ops.attention import fusion_attention
from llmtuner.kernel.ops.moe import moe_token_permute
from llmtuner.kernel.ops.rms_norm import rms_norm
from llmtuner.kernel.ops.rotary import rotary_mul


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_KERNEL_REGISTRY",
        {name: dict(backends) for name, backends in registry._KERNEL_REGISTRY.items()},
    )
    monkeypatch.setattr(registry, "_warned_fallbacks", set())


def test_dispatch_follows_tensor_instead_of_current_accelerator(monkeypatch):
    def reject_hardware_lookup():
        raise AssertionError("Tensor dispatch must not inspect the current accelerator")

    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", reject_hardware_lookup
    )
    values = torch.randn(3, 4)
    assert rms_norm(values, torch.ones(4), 1e-6).device.type == "cpu"
    actual = dispatch_kernel(
        "rotary_mul", hidden_states=values, cosine=torch.ones(4), sine=torch.zeros(4)
    )
    torch.testing.assert_close(actual, values, atol=0, rtol=0)


def test_registry_fallback_is_explicit_and_warns_once(isolated_registry, caplog):
    with pytest.raises(KeyError, match="not available"):
        get_kernel("rms_norm", "unknown")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            assert get_kernel(
                "rms_norm", "unknown", fallback_to_cpu=True
            ) is get_kernel("rms_norm", "cpu")
    assert len(caplog.records) == 1
    with pytest.raises(KeyError, match="not registered"):
        get_kernel("missing_kernel", "cpu")


@pytest.mark.parametrize("name,device", [("", "cpu"), ("a", ""), (None, "cpu")])
def test_registry_rejects_invalid_identifiers(name, device):
    with pytest.raises(ValueError, match="must not be empty"):
        register_kernel(name, device)


def test_registry_rejects_noncallables_and_accepts_idempotent_registration(
    isolated_registry,
):
    with pytest.raises(TypeError, match="callable"):
        register_kernel("custom", "cpu")(None)

    def kernel():
        return 42

    for _ in range(2):
        assert register_kernel("custom", "cpu")(kernel) is kernel
    assert dispatch_kernel("custom", device_type="cpu") == 42


@pytest.mark.parametrize("mode", [None, "half", "interleave"])
def test_rotary_uses_each_coordinate_factor_and_preserves_gradients(mode):
    torch.manual_seed(17)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    cosine = torch.randn(2, 1, 8, dtype=torch.float64, requires_grad=True)
    sine = torch.randn(2, 1, 8, dtype=torch.float64, requires_grad=True)
    # Independent coordinate permutation, with deliberately unequal even/odd factors.
    indices = (
        [1, 0, 3, 2, 5, 4, 7, 6] if mode == "interleave" else [4, 5, 6, 7, 0, 1, 2, 3]
    )
    signs = [-1, 1] * 4 if mode == "interleave" else [-1] * 4 + [1] * 4
    expected = x * cosine + x[..., indices] * x.new_tensor(signs) * sine
    actual = rotary_mul(x, cosine, sine, rotary_mode=mode)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.autograd.gradcheck(
        lambda a, b, c: rotary_mul(a, b, c, rotary_mode=mode), (x, cosine, sine)
    )


@pytest.mark.parametrize("width,mode", [(7, "half"), (7, "interleave"), (8, "bad")])
def test_rotary_rejects_invalid_geometry_or_mode(width, mode):
    x = torch.randn(2, width)
    with pytest.raises(ValueError):
        rotary_mul(x, torch.ones_like(x), torch.zeros_like(x), rotary_mode=mode)


def test_rms_norm_matches_formula_and_gradients():
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float64, requires_grad=True)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(rms_norm(x, weight, 1e-5), expected)
    assert torch.autograd.gradcheck(lambda a, b: rms_norm(a, b, 1e-5), (x, weight))


@pytest.mark.parametrize("epsilon", [-1.0, float("nan"), float("inf")])
def test_rms_norm_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        rms_norm(torch.ones(2, 4), torch.ones(4), epsilon)


def test_moe_stable_permutation_and_duplicate_assignment_backward():
    x = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    experts = torch.tensor([[2, 0], [0, 0], [2, 1]], dtype=torch.int32)
    # Sorting (expert, flattened assignment order) yields these source rows.
    rows = [0, 1, 1, 2, 0, 2]
    actual = moe_token_permute(x, experts)
    torch.testing.assert_close(actual, x[rows], atol=0, rtol=0)
    weights = torch.arange(1, 25, dtype=x.dtype).reshape(6, 4)
    (actual * weights).sum().backward()
    expected_grad = torch.stack(
        [weights[0] + weights[4], weights[1] + weights[2], weights[3] + weights[5]]
    )
    torch.testing.assert_close(x.grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape,dtype", [((2,), torch.int64), ((2, 0), torch.int64), ((2, 1), torch.float32)]
)
def test_moe_rejects_invalid_expert_ids(shape, dtype):
    with pytest.raises(ValueError):
        moe_token_permute(torch.ones(2, 4), torch.zeros(shape, dtype=dtype))


def test_empty_moe_permutation_retains_autograd():
    x = torch.empty(0, 4, requires_grad=True)
    result = moe_token_permute(x, torch.empty(0, 2, dtype=torch.int64))
    assert result.shape == (0, 4)
    result.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape


def test_packed_attention_cross_lengths_output_and_backward():
    torch.manual_seed(19)
    tensors = [torch.randn(n, 2, 4, requires_grad=True) for n in (5, 7, 7)]
    reference = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    q, k, v = reference
    mask = torch.zeros(5, 7, dtype=torch.bool)
    mask[:2, :4] = True
    mask[2:, 4:] = True
    scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2) * 0.5
    expected = (
        scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.transpose(0, 1)
    ).transpose(0, 1)
    actual = fusion_attention(
        *tensors,
        head_num=2,
        scale=0.5,
        q_cu_seqlens=[0, 2, 5],
        k_cu_seqlens=torch.tensor([0, 4, 7]),
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    gradient = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, tensors, gradient)
    expected_grad = torch.autograd.grad(expected, reference, gradient)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "offsets",
    [
        [0, 4, 2, 5],
        [0, -1, 5],
        [1, 5],
        [0, 4],
        [0, True, 5],
        [0, 2.5, 5],
        torch.tensor([0.0, 5.0]),
        torch.tensor([[0, 5]]),
    ],
)
def test_packed_attention_rejects_bad_offsets(offsets):
    x = torch.ones(5, 2, 4)
    with pytest.raises(ValueError):
        fusion_attention(
            x, x, x, head_num=2, scale=0.5, q_cu_seqlens=offsets, k_cu_seqlens=offsets
        )


@pytest.mark.parametrize("tokens", [0, 3])
def test_empty_attention_queries_keep_zero_gradients(tokens):
    q = torch.empty(0, 2, 4, requires_grad=True)
    k = torch.randn(tokens, 2, 4, requires_grad=True)
    v = torch.randn(tokens, 2, 4, requires_grad=True)
    result = fusion_attention(
        q, k, v, head_num=2, scale=0.5, q_cu_seqlens=None, k_cu_seqlens=None
    )
    assert result.shape == q.shape
    result.sum().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert tensor.grad.count_nonzero() == 0


def test_attention_rejects_nonempty_queries_without_keys():
    with pytest.raises(ValueError, match="nonempty keys"):
        fusion_attention(
            torch.empty(2, 1, 4),
            torch.empty(0, 1, 4),
            torch.empty(0, 1, 4),
            head_num=1,
            scale=0.5,
            q_cu_seqlens=None,
            k_cu_seqlens=None,
        )


def test_npu_dependency_error_keeps_original_cause(monkeypatch):
    cause = ImportError("torch-npu is unavailable")
    monkeypatch.setattr(npu_basic, "torch_npu", None)
    monkeypatch.setattr(npu_basic, "TORCH_NPU_IMPORT_ERROR", cause)
    with pytest.raises(ImportError, match="require torch-npu") as result:
        npu_basic.rms_norm_npu(torch.ones(2, 4), torch.ones(4), 1e-6)
    assert result.value.__cause__ is cause


def test_tensor_free_lookup_reuses_accelerator_device_selection(monkeypatch):
    monkeypatch.setattr(registry, "default_device_type", "cpu")
    assert get_kernel("rms_norm") is get_kernel("rms_norm", "cpu")


def test_rotary_rejects_factors_that_expand_or_do_not_broadcast():
    for shape in ((3, 4), (3, 2, 4)):
        with pytest.raises(ValueError, match="broadcast"):
            rotary_mul(torch.ones(2, 4), torch.ones(shape), torch.zeros(shape))


def test_rms_norm_rejects_multiaxis_weights_and_mismatched_width():
    for weight in (torch.ones(2, 2), torch.ones(3)):
        with pytest.raises(ValueError):
            rms_norm(torch.ones(2, 4), weight, 1e-6)


def test_attention_empty_middle_document_preserves_other_documents():
    x = torch.randn(5, 2, 4)
    options = dict(head_num=2, scale=0.5)
    expected = fusion_attention(
        x, x, x, **options, q_cu_seqlens=[0, 2, 5], k_cu_seqlens=[0, 2, 5]
    )
    actual = fusion_attention(
        x, x, x, **options, q_cu_seqlens=[0, 2, 2, 5], k_cu_seqlens=[0, 2, 2, 5]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
