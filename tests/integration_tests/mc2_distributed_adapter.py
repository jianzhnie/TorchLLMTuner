"""CPU/Gloo emulation of MC2 wire ordering, not Ascend kernel validation.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. Real all-to-all transport
and production MC2 autograd are compared with independent dense experts.
"""

from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.models.common.moe.experts import GroupedExperts, RoutedExperts
from llmtuner.parallel.expert_parallel import mc2_dispatch
from llmtuner.parallel.expert_parallel.token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)
from tests.unit_tests.cpu.models.test_mc2_ops import EmulatedOps


class DistributedOps(EmulatedOps):
    def npu_alltoallv_gmm(self, inputs, weights, comm, size, send, recv, **kwargs):
        local = weights.shape[0]
        send_totals = torch.tensor(send).reshape(size, local).sum(1).tolist()
        recv_totals = torch.tensor(recv).reshape(size, local).sum(1).tolist()
        received = inputs.new_empty(sum(recv), inputs.shape[1])
        dist.all_to_all_single(received, inputs.contiguous(), recv_totals, send_totals)
        pieces = received.split(recv)
        permuted = torch.cat(
            [
                pieces[source * local + expert]
                for expert in range(local)
                for source in range(size)
            ]
        )
        totals = torch.tensor(recv).reshape(size, local).sum(0).tolist()
        return (
            self.project(
                permuted, weights, totals, kwargs.get("trans_gmm_weight", False)
            ),
            None,
            permuted,
        )

    def npu_gmm_alltoallv(self, inputs, weights, comm, size, send, recv, **kwargs):
        local = weights.shape[0]
        counts = torch.tensor(send).reshape(size, local)
        projected = self.project(
            inputs,
            weights,
            counts.sum(0).tolist(),
            kwargs.get("trans_gmm_weight", False),
        )
        pieces = projected.split(counts.T.flatten().tolist())
        source_order = torch.cat(
            [
                pieces[expert * size + source]
                for source in range(size)
                for expert in range(local)
            ]
        )
        output = inputs.new_empty(sum(recv), projected.shape[1])
        dist.all_to_all_single(
            output,
            source_order.contiguous(),
            torch.tensor(recv).reshape(size, local).sum(1).tolist(),
            counts.sum(1).tolist(),
        )
        return output, None


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        for empty_rank, zero_receives, all_empty in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            torch.manual_seed(5)
            reference_experts = GroupedExperts(4, 3, 4, use_grouped_mm=False).half()
            for parameter in reference_experts.parameters():
                torch.nn.init.normal_(parameter, std=0.1)
            experts = mc2_dispatch.MC2GroupedExperts(
                4, 3, 2, use_grouped_mm=False
            ).half()
            with torch.no_grad():
                for local, full in zip(
                    experts.parameters(), reference_experts.parameters(), strict=True
                ):
                    local.copy_(full[rank * 2 : (rank + 1) * 2])
            inputs = torch.randn(5, 4, dtype=torch.float16)
            scores = torch.rand(5, 2, dtype=torch.float16)
            ids = torch.tensor([[0, 1], [2, 3], [0, 3], [1, 2], [2, 0]])
            if zero_receives:
                ids[:] = torch.tensor([0, 1])
            boundary = 0 if empty_rank else 2
            sl = slice(0, boundary) if rank == 0 else slice(boundary, 5)
            if all_empty:
                inputs, scores, ids = inputs[:0], scores[:0], ids[:0]
                sl = slice(0, 0)
            x = inputs[sl].clone().requires_grad_()
            weights = scores[sl].clone().requires_grad_()
            ref_x = inputs.clone().requires_grad_()
            ref_weights = scores.clone().requires_grad_()
            reference = RoutedExperts(reference_experts, LocalTokenDispatcher(4, 2))
            dispatcher = AllToAllTokenDispatcher(4, 2)
            dispatcher.set_ep_group(ep_group=dist.group.WORLD)
            ops = DistributedOps()
            with patch.object(mc2_dispatch, "load_mc2_ops", lambda ops=ops: ops):
                model = mc2_dispatch.MC2RoutedExperts(experts, dispatcher)
            optimizer = torch.optim.SGD(experts.parameters(), lr=0.01, momentum=0.9)
            reference_optimizer = torch.optim.SGD(
                reference_experts.parameters(), lr=0.01, momentum=0.9
            )
            original_ids = ids.clone()
            for step in range(3):
                # Exercise loss of all local routes after momentum exists,
                # then restore the original destinations on the next update.
                ids.copy_(original_ids)
                if step == 1 and not (empty_rank or zero_receives or all_empty):
                    ids[:] = torch.tensor([0, 1])
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                for leaf in (x, weights, ref_x, ref_weights):
                    leaf.grad = None
                expected = reference(
                    ref_x, ref_weights, ids, torch.bincount(ids.flatten(), minlength=4)
                )
                with (
                    patch.object(mc2_dispatch, "load_mc2_ops", lambda ops=ops: ops),
                    patch.object(
                        mc2_dispatch,
                        "mc2_communication_context",
                        lambda *args, ops=ops: (ops, "test", 2),
                    ),
                ):
                    actual = model(
                        x,
                        weights,
                        ids[sl],
                        torch.bincount(ids[sl].flatten(), minlength=4),
                    )
                    actual.float().sum().backward()
                expected.float().sum().backward()
                for got, wanted in (
                    (actual, expected[sl]),
                    (x.grad, ref_x.grad[sl]),
                    (weights.grad, ref_weights.grad[sl]),
                ):
                    torch.testing.assert_close(got, wanted, atol=5e-4, rtol=1e-2)
                for got, wanted in zip(
                    experts.parameters(), reference_experts.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        got.grad,
                        wanted.grad[rank * 2 : (rank + 1) * 2],
                        atol=5e-4,
                        rtol=1e-2,
                    )
                optimizer.step()
                reference_optimizer.step()
                for got, wanted in zip(
                    experts.parameters(), reference_experts.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        got,
                        wanted[rank * 2 : (rank + 1) * 2],
                        atol=5e-4,
                        rtol=1e-2,
                    )
                    torch.testing.assert_close(
                        optimizer.state[got]["momentum_buffer"],
                        reference_optimizer.state[wanted]["momentum_buffer"][
                            rank * 2 : (rank + 1) * 2
                        ],
                        atol=5e-4,
                        rtol=1e-2,
                    )
            if rank == 0:
                print(
                    f"MC2 adapter {empty_rank=} {zero_receives=} {all_empty=}: "
                    "three momentum updates, outputs and gradients passed"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
