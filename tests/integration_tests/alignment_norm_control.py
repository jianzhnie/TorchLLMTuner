"""Diagnostic RMSNorm weight-gradient ownership control, not a training backend."""

from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def native_norm(value, weight, eps):
    if value.device.type == "npu":
        import torch_npu

        return torch_npu.npu_rms_norm(value, weight, eps)[0]
    return torch.nn.functional.rms_norm(value, (weight.numel(),), weight, eps)


class PartitionedNormWeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, eps, group, geometry):
        batch, sequence = geometry
        size = dist.get_world_size(group)
        if (
            value.numel() // value.shape[-1] * size != batch * sequence
            or sequence % size
            or batch < 1
        ):
            raise ValueError("Norm control requires complete equal-length documents")
        ctx.save_for_backward(value, weight)
        ctx.eps, ctx.group, ctx.geometry = eps, group, geometry
        return native_norm(value, weight, eps)

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        with torch.enable_grad():
            local = value.detach().requires_grad_()
            local_weight = weight.detach().requires_grad_()
            output = native_norm(local, local_weight, ctx.eps)
            input_gradient = torch.autograd.grad(output, local, gradient)[0]

        size, rank = dist.get_world_size(ctx.group), dist.get_rank(ctx.group)
        # Repartition only this diagnostic's parameter-gradient calculation.
        # The ordinary native input derivative and residual graph stay intact.
        gathered = []
        for tensor in (value, gradient):
            local = tensor.detach().reshape(-1, tensor.shape[-1]).contiguous()
            pieces = [torch.empty_like(local) for _ in range(size)]
            dist.all_gather(pieces, local, group=ctx.group)
            full = torch.cat(pieces).reshape(*ctx.geometry, tensor.shape[-1])
            gathered.append(full.chunk(size, dim=1)[rank].contiguous())
        reference_input, reference_gradient = gathered
        with torch.enable_grad():
            reference_weight = weight.detach().requires_grad_()
            output = native_norm(reference_input, reference_weight, ctx.eps)
            weight_gradient = torch.autograd.grad(
                output, reference_weight, reference_gradient
            )[0]
        return input_gradient, weight_gradient, None, None, None


def install_norm_partition_control(trainer, geometry):
    dims = trainer.parallel_dims
    if dims.tp <= 1 or dims.cp != 1 or dims.pp != 1:
        raise ValueError("Norm partition control requires TP without CP/PP")
    model = trainer.model_parts[0]
    norms = [model.norm]
    for layer in model.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    group = dims.get_mesh("tp").get_group()

    def forward(module, value):
        weight = module.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        return PartitionedNormWeightGradient.apply(
            value, weight, module.variance_epsilon, group, geometry()
        )

    for module in norms:
        module.forward = MethodType(forward, module)
