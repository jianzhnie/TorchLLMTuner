"""Two NPU ranks: TP assembly preserves an instance attention kernel and VJP.

Exercises both sharded and replicated attention boundaries with real HCCL.
The small synthetic kernel isolates composition; it does not certify Qwen RoPE
arithmetic or target-model training precision.
"""

import os
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from tests.caps import require_env


def kernel(self, hidden_states):
    return torch.tanh(hidden_states * self.gain), None


def main():
    require_env("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
        rank = dist.get_rank()
        for replicated in (False, True):
            model = nn.Module()
            attention = nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                attention.add_module(name, nn.Linear(8, 8, bias=False, device="npu"))
            attention.gain = nn.Parameter(torch.tensor(0.5, device="npu"))
            attention.forward = MethodType(kernel, attention)
            model.add_module("self_attn", attention)
            model.add_module("mlp", nn.Linear(8, 8, bias=False, device="npu"))
            plan = (
                {"mlp": "colwise"}
                if replicated
                else {
                    "self_attn.q_proj": "colwise",
                    "self_attn.k_proj": "colwise",
                    "self_attn.v_proj": "colwise",
                    "self_attn.o_proj": "rowwise",
                }
            )
            apply_tp(model, mesh, ParallelConfig(tensor_parallel_size=2), plan=plan)
            inputs = torch.linspace(-1, 1, 128, device="npu").reshape(1, 16, 8)
            reference_input = inputs.detach().clone().requires_grad_()
            reference, _ = attention._tp_attention_original_forward(reference_input)
            if replicated:
                reference = reference.chunk(2, dim=1)[rank]
            probe = torch.full_like(reference, rank + 1)
            reference_gradients = torch.autograd.grad(
                reference, (reference_input, attention.gain), probe
            )
            input_gradient = reference_gradients[0].clone()
            dist.all_reduce(input_gradient)
            local = inputs.chunk(2, dim=1)[rank].detach().clone().requires_grad_()
            observed, auxiliary = attention(local)
            assert auxiliary is None
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
            gradients = torch.autograd.grad(observed, (local, attention.gain), probe)
            torch.testing.assert_close(
                gradients[0], input_gradient.chunk(2, dim=1)[rank], atol=0, rtol=0
            )
            torch.testing.assert_close(
                gradients[1], reference_gradients[1], atol=0, rtol=0
            )
            print(f"PASS rank={rank} replicated={replicated} attention kernel VJP")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
