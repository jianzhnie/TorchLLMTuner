"""Check CPU-offloaded HSDP clipping against a full accelerator reference.

Run with four accelerator ranks using torchrun from the repository root.
This covers FP32 clipping, not full Trainer, optimizer or recovery behavior.
"""

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.fsdp_mesh_equivalence import (
    SEQ,
    _build_model,
    _data,
    _loss_sum,
)
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    try:
        assert device.type != "cpu", "CPU offload requires an accelerator"
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert world == 4
        config = ParallelConfig(
            data_parallel_replicate_size=2, data_parallel_shard_size=2
        )
        config.fsdp_config.fsdp_cpu_offload = True
        config.fsdp_config.fsdp_cpu_offload_pin_memory = False
        topology = ParallelTopology.from_config(config, world)
        topology.build_device_mesh()
        model = apply_fsdp(_build_model(), config, topology)
        (_loss_sum(model, *_data(0, rank)) / (world * SEQ)).backward()
        reference = _build_model()
        reference_loss = sum(
            _loss_sum(reference, *_data(0, index)) for index in range(world)
        ) / (world * SEQ)
        reference_loss.backward()
        parameters = list(model.parameters())
        references = list(reference.parameters())
        original = [parameter.grad.to_local().clone() for parameter in parameters]
        expected_original = [parameter.grad.clone() for parameter in references]
        for order in (1.0, 2.0, float("inf")):
            for foreach in (False, True):
                for parameter, gradient in zip(parameters, original, strict=True):
                    assert parameter.grad.device.type == "cpu"
                    parameter.grad.to_local().copy_(gradient)
                for parameter, gradient in zip(
                    references, expected_original, strict=True
                ):
                    parameter.grad.copy_(gradient)
                expected = torch.nn.utils.clip_grad_norm_(
                    references, 0.01, norm_type=order, foreach=foreach
                )
                actual = clip_grad_norm_(
                    parameters,
                    0.01,
                    norm_type=order,
                    foreach=foreach,
                    error_if_nonfinite=True,
                )
                assert actual.device == device
                assert expected.item() > 0.01
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                for parameter, expected_parameter in zip(
                    parameters, references, strict=True
                ):
                    assert parameter.grad.device.type == "cpu"
                    full = parameter.grad.to(device).full_tensor()
                    torch.testing.assert_close(
                        full, expected_parameter.grad, atol=1e-6, rtol=1e-6
                    )
                if rank == 0:
                    print(f"order={order} {foreach=} norm and clipping passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
