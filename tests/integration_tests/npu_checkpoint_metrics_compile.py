"""NPU aot_eager: checkpointed metric mutations and optimizer updates.

Single-device real-op regression for auxiliary injection and quantile statistics;
this does not exercise distributed expert dispatch or Inductor.
"""

from copy import deepcopy
from itertools import product

import torch
from torch import nn

from llmtuner.config.activation_checkpoint_config import SelectiveACConfig
from llmtuner.memory.activation_checkpoint import wrap_full, wrap_selective
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.common.moe.routing import QuantileBalancer


class SquaredAuxLoss(AuxLoss):
    def forward(self, scores):
        return self.inject(scores.square().sum(), carrier=scores)


class MetricBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.aux = SquaredAuxLoss(coeff=0.3)
        self.balancer = QuantileBalancer(4, 2, 16)
        self.register_buffer("bias", torch.tensor([-0.2, -0.1, 0.1, 0.2]))

    def forward(self, inputs, padding):
        scores = self.linear(inputs).sigmoid()
        self.balancer.observe(
            scores, scores.mean(-1, keepdim=True), self.bias, padding_mask_T=padding
        )
        return self.aux(scores).square()


def main():
    if not torch.npu.is_available():
        raise RuntimeError("checkpoint metric compilation requires a visible NPU")
    torch.npu.set_device(0)
    for mode, padding_kind, dtype in product(
        ("full", "selective"), ("none", "mixed", "all"), (torch.float32, torch.bfloat16)
    ):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        torch.manual_seed(71)
        reference = MetricBlock().to(device="npu", dtype=dtype)
        candidate = deepcopy(reference)
        wrapped = (
            wrap_full(candidate)
            if mode == "full"
            else wrap_selective(candidate, SelectiveACConfig())
        )
        compiled = torch.compile(wrapped, backend="aot_eager", fullgraph=True)
        optimizers = [
            torch.optim.SGD(m.parameters(), lr=0.01) for m in (reference, candidate)
        ]
        padding = {
            "none": None,
            "mixed": torch.tensor([False, True, False], device="npu"),
            "all": torch.ones(3, dtype=torch.bool, device="npu"),
        }[padding_kind]
        for _ in range(3):
            AuxLoss.set_step_denominator(
                torch.tensor(3.0, device="npu"), model_parts=[reference, candidate]
            )
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(3, 4, device="npu", dtype=dtype)
            expected = reference(inputs, padding)
            actual = compiled(inputs, padding)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for retain in (True, False):
                expected.sum().backward(retain_graph=retain)
                actual.sum().backward(retain_graph=retain)
            assert reference.aux.instance_acc.item() > 0
            torch.testing.assert_close(
                candidate.aux.instance_acc, reference.aux.instance_acc, rtol=0, atol=0
            )
            torch.testing.assert_close(
                candidate.balancer.required_bias_histogram_EB,
                reference.balancer.required_bias_histogram_EB,
                rtol=0,
                atol=0,
            )
            for got, want in zip(
                candidate.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(got.grad, want.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
            for name, value in candidate.state_dict().items():
                torch.testing.assert_close(
                    value, reference.state_dict()[name], rtol=0, atol=0
                )
        valid = {"none": 3, "mixed": 2, "all": 0}[padding_kind]
        assert (
            candidate.balancer.required_bias_histogram_EB.sum().item() == 3 * 4 * valid
        )
        assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
        torch.npu.synchronize()
        print(
            f"PASS {mode=} {padding_kind=} {dtype=}: metrics, gradients "
            "and three SGD steps exact",
            flush=True,
        )


if __name__ == "__main__":
    main()
