"""Coordinate a backend registered on only one NPU rank before model mutation."""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.exc import InvalidBackend
from torch._dynamo.utils import counters

from llmtuner.config import CompileConfig, ParallelConfig
from llmtuner.parallel.parallelize import parallelize_model
from tests.integration_tests.harness import init_distributed


class ProbeLayer(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs).sin()


def main():
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(19 + rank)
        backend = "llmtuner_rank_local_preflight"
        if rank == 1:
            register_backend(lambda graph, inputs: graph.forward, name=backend)
        model = nn.Module()
        model.layers = nn.ModuleList([ProbeLayer(4, 4, device=device)])
        original = model.layers[0]
        error = None
        try:
            parallelize_model(
                model,
                cfg=ParallelConfig(),
                mesh=None,
                parallel_dims=None,
                compile=True,
                compile_config=CompileConfig(backend=backend, per_block=True),
            )
        except (InvalidBackend, RuntimeError) as caught:
            error = caught
        outcomes = [None, None]
        dist.all_gather_object(
            outcomes,
            (
                type(error).__name__,
                str(error),
                model.layers[0] is original and original._compiled_call_impl is None,
            ),
        )
        assert outcomes[0][0] == "InvalidBackend", outcomes
        assert outcomes[1][0] == "RuntimeError", outcomes
        assert "failed on another rank" in outcomes[1][1], outcomes
        assert all(item[2] for item in outcomes), outcomes
        reference = deepcopy(original)
        graphs_before = counters["stats"]["unique_graphs"]
        parallelize_model(
            model,
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=None,
            compile=True,
            compile_config=CompileConfig(backend="aot_eager", per_block=True),
        )
        assert original._compiled_call_impl is not None
        inputs = torch.linspace(-1, 1, 16, device=device).reshape(4, 4)
        actual, expected = model.layers[0](inputs), reference(inputs)
        assert counters["stats"]["unique_graphs"] > graphs_before, counters["stats"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for parameter, other in zip(
            original.parameters(), reference.parameters(), strict=True
        ):
            assert parameter.grad.count_nonzero().item() > 0
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        torch.optim.SGD(original.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for parameter, other in zip(
            original.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        print(f"PASS coordinated compile preflight rank={rank}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
