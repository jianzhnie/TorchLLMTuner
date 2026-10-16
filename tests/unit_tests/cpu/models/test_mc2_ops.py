"""CPU emulation checks adapter algebra only, not Ascend fused kernels."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.parallel.expert_parallel.mc2_ops import (
    AllToAllGroupedMatmul,
    GroupedMatmulAllToAll,
    load_mc2_ops,
)


class EmulatedOps:
    def __init__(self):
        self.calls = []

    def project(self, inputs, weights, counts, transpose):
        pieces = inputs.split(counts)
        return torch.cat(
            [
                piece @ (weight.T if transpose else weight)
                for piece, weight in zip(pieces, weights, strict=True)
            ]
        )

    def npu_alltoallv_gmm(self, inputs, weights, comm, size, send, recv, **kwargs):
        assert size == 1 and send == recv and comm == "test"
        self.calls.append(("dispatch", kwargs.get("trans_gmm_weight", False)))
        return (
            self.project(inputs, weights, recv, kwargs.get("trans_gmm_weight", False)),
            None,
            inputs,
        )

    def npu_gmm_alltoallv(self, inputs, weights, comm, size, send, recv, **kwargs):
        assert size == 1 and send == recv and comm == "test"
        self.calls.append(("combine", kwargs.get("trans_gmm_weight", False)))
        return (
            self.project(inputs, weights, send, kwargs.get("trans_gmm_weight", False)),
            None,
        )

    def npu_grouped_matmul(self, left, right, *, group_list, **kwargs):
        counts = group_list.tolist()
        return [
            torch.stack(
                [
                    x @ y
                    for x, y in zip(
                        left[0].split(counts, dim=1),
                        right[0].split(counts),
                        strict=True,
                    )
                ]
            )
        ]


@pytest.mark.parametrize("counts", [[2, 3], [0, 5], [0, 0]])
def test_mc2_fused_adapter_gradients_match_independent_linear_reference(counts):
    torch.manual_seed(13)
    inputs = torch.randn(sum(counts), 4, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    down = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    reference_input = inputs.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    reference_down = down.detach().clone().requires_grad_()
    expected = []
    for i, piece in enumerate(reference_input.split(counts)):
        gate, up = F.linear(piece, reference_weights[i]).chunk(2, dim=-1)
        expected.append(F.linear(F.silu(gate) * up, reference_down[i]))
    expected = torch.cat(expected)
    ops = EmulatedOps()
    gate, up = AllToAllGroupedMatmul.apply(
        inputs, weights, ops, "test", 1, counts, counts
    ).chunk(2, dim=-1)
    actual = GroupedMatmulAllToAll.apply(
        F.silu(gate) * up, down, ops, "test", 1, counts, counts
    )
    cotangent = torch.randn_like(expected)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        (inputs, weights, down),
        (reference_input, reference_weights, reference_down),
        strict=True,
    ):
        torch.testing.assert_close(got.grad, want.grad)
    assert ops.calls == [
        ("dispatch", True),
        ("combine", True),
        ("dispatch", False),
        ("combine", False),
    ]


def test_mc2_missing_package_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(ImportError, match="matching your PyTorch"):
        load_mc2_ops()


def test_mc2_missing_fused_ops_are_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    with pytest.raises(EnvironmentUnsupportedError, match="npu_alltoallv_gmm"):
        load_mc2_ops()


def test_mc2_routed_experts_preserve_separate_weights_and_router_gradients(monkeypatch):
    from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
    from llmtuner.parallel.expert_parallel import mc2_dispatch
    from llmtuner.parallel.expert_parallel.token_dispatch import (
        AllToAllTokenDispatcher,
        DispatchCounts,
        LocalTokenDispatcher,
    )

    torch.manual_seed(21)
    ops = EmulatedOps()
    monkeypatch.setattr(mc2_dispatch, "load_mc2_ops", lambda: ops)
    monkeypatch.setattr(
        mc2_dispatch, "mc2_communication_context", lambda *args: (ops, "test", 1)
    )
    experts = mc2_dispatch.MC2GroupedExperts(4, 3, 2, use_grouped_mm=False).half()
    parameter_ids = [id(parameter) for parameter in experts.parameters()]
    hooks = []
    experts.register_forward_pre_hook(lambda module, args: hooks.append("pre"))
    experts.register_forward_hook(lambda module, args, output: hooks.append("post"))
    reference_experts = GroupedExperts(4, 3, 2, use_grouped_mm=False).half()
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    reference_experts.load_state_dict(experts.state_dict())
    dispatcher = AllToAllTokenDispatcher(2, 2)
    dispatcher.ep_group = object()
    monkeypatch.setattr(
        dispatcher,
        "prepare_dispatch_counts",
        lambda counts: [
            DispatchCounts(
                counts=counts[0],
                input_splits=[int(counts.sum())],
                output_splits=[int(counts.sum())],
            )
        ],
    )
    actual_model = mc2_dispatch.MC2RoutedExperts(experts, dispatcher)
    reference_model = RoutedExperts(reference_experts, LocalTokenDispatcher(2, 2))
    inputs = torch.randn(3, 4, dtype=torch.float16, requires_grad=True)
    scores = torch.rand(3, 2, dtype=torch.float16, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_scores = scores.detach().clone().requires_grad_()
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    counts = torch.bincount(ids.flatten(), minlength=2)
    actual = actual_model(inputs, scores, ids, counts)
    expected = reference_model(ref_inputs, ref_scores, ids, counts)
    assert hooks == ["pre", "post"]
    assert [id(parameter) for parameter in experts.parameters()] == parameter_ids
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
    for got, want in zip(
        [inputs, scores, *experts.parameters()],
        [ref_inputs, ref_scores, *reference_experts.parameters()],
        strict=True,
    ):
        torch.testing.assert_close(got.grad, want.grad, atol=1e-4, rtol=1e-3)
    assert set(experts.state_dict()) == {"w1_EFD", "w3_EFD", "w2_EDF"}


@pytest.mark.parametrize(
    "counts, world_size, expected",
    [
        ([2, 0, 3, 1, 4, 0], 2, [3, 4, 3]),
        ([0, 0, 0, 0, 0, 0], 3, [0, 0]),
        ([5, 0, 0, 2, 1, 7], 3, [6, 9]),
    ],
)
def test_mc2_weight_gradient_counts_sum_source_ranks(counts, world_size, expected):
    from llmtuner.parallel.expert_parallel.mc2_ops import _expert_counts

    actual = _expert_counts(counts, world_size, torch.device("cpu"))
    assert actual.dtype == torch.int64
    assert actual.tolist() == expected
