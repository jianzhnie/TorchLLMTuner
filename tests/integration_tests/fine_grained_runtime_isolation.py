"""Two live relocated FSDP runtimes must keep independent iteration state.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 and this script.
CPU/Gloo; interleaved forwards, reverse backwards, a failed peer forward,
and independent distributed EMA updates checked against dense averages.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.optimizer.ema import EMA
from tests.integration_tests.fine_grained_fsdp_equivalence import (
    Model,
    assert_clean,
    full,
    runtime,
)


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("cpu", (2,))
        replicated = init_device_mesh(
            "cpu", (2, 1), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        for recompute in (False, True):
            models, optimizers, references, reference_optimizers = [], [], [], []
            for index in range(2):
                seed = 42 + index
                torch.manual_seed(seed)
                reference = Model()
                model, optimizer = runtime(
                    torch.device("cpu"),
                    mesh,
                    replicated,
                    reshard=bool(index),
                    recompute=recompute,
                    multiple_groups=not bool(index),
                    seed=seed,
                )
                models.append(model)
                optimizers.append(optimizer)
                references.append(reference)
                reference_optimizers.append(
                    torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
                )
            averages = [EMA(model_parts=[model], decay=0.5) for model in models]
            dense_averages = [
                [p.detach().clone() for p in reference.parameters()]
                for reference in references
            ]
            generator = torch.Generator().manual_seed(100 + dist.get_rank())
            for step in range(3):
                inputs = [torch.randn(2, 4, generator=generator) for _ in models]
                # Keep B's live forward while A fails and resets its contexts.
                outputs = [None, models[1](inputs[1])]
                if step == 1:
                    models[0].blocks[0].fail = True
                    try:
                        models[0](inputs[0])
                    except RuntimeError as error:
                        assert str(error) == "injected forward failure"
                    else:
                        raise AssertionError("expected forward failure")
                    models[0].reset_iter_state()
                    assert_clean(models[0], multiple_groups=True)
                    models[0].blocks[0].fail = False
                outputs[0] = models[0](inputs[0])
                roots = [model._get_fsdp_state() for model in models]
                assert roots[0]._state_ctx is not roots[1]._state_ctx
                contexts = [
                    {id(c) for c in root.communication_contexts} for root in roots
                ]
                assert contexts[0].isdisjoint(contexts[1])
                for index in (1, 0) if step % 2 == 0 else (0, 1):
                    expected = references[index](inputs[index])
                    torch.testing.assert_close(outputs[index], expected)
                    outputs[index].square().sum().backward()
                    expected.square().sum().backward()
                    for got, want in zip(
                        models[index].parameters(),
                        references[index].parameters(),
                        strict=True,
                    ):
                        dist.all_reduce(want.grad)
                        torch.testing.assert_close(full(got.grad), want.grad)
                    assert_clean(models[index], multiple_groups=index == 0)
                    optimizers[index].step()
                    reference_optimizers[index].step()
                    averages[index].step(step + 1)
                    for average, parameter in zip(
                        dense_averages[index],
                        references[index].parameters(),
                        strict=True,
                    ):
                        average.lerp_(parameter.detach(), 0.5)
                    actual_averages = [
                        state["ema_params"]
                        for optimizer in averages[index].optimizers
                        for state in optimizer.state.values()
                    ]
                    for actual, expected_average in zip(
                        actual_averages, dense_averages[index], strict=True
                    ):
                        torch.testing.assert_close(full(actual), expected_average)
                    optimizers[index].zero_grad(set_to_none=True)
                    reference_optimizers[index].zero_grad(set_to_none=True)
                    for got, want in zip(
                        models[index].parameters(),
                        references[index].parameters(),
                        strict=True,
                    ):
                        torch.testing.assert_close(full(got), want)
            print("PASS runtime isolation", dist.get_rank(), recompute, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
