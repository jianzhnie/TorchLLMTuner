"""Optional split-expert communication layout for packed parameter storage."""

import math

import torch


def reorder_expert_region(source, destination, *, experts, width, hidden, inverse):
    """Translate equal-sized gate/up/down regions without changing their values."""
    projection = experts * width * hidden
    if source.numel() != 3 * projection or destination.numel() != 3 * projection:
        raise ValueError("expert communication region has incompatible size")
    if inverse:
        gate = source[:projection].view(experts, width, hidden)
        down = source[projection : 2 * projection]
        up = source[2 * projection :].view(experts, width, hidden)
        packed = destination[: 2 * projection].view(experts, 2, width, hidden)
        packed[:, 0].copy_(gate)
        packed[:, 1].copy_(up)
        destination[2 * projection :].copy_(down)
    else:
        packed = source[: 2 * projection].view(experts, 2, width, hidden)
        destination[:projection].view(experts, width, hidden).copy_(packed[:, 0])
        destination[projection : 2 * projection].copy_(source[2 * projection :])
        destination[2 * projection :].view(experts, width, hidden).copy_(packed[:, 1])


class SplitExpertReduceScatter:
    """Permute only communication buffers; retain FSDP parameters and gradients.

    Extra input/output buffers live on the calling reduction stream. The FSDP
    synchronous enqueue contract is required; an asynchronous Work would not
    cover the subsequent inverse permutation.
    """

    def __init__(
        self,
        original,
        *,
        world_size,
        offset,
        experts,
        width,
        hidden,
        workspace_limit_bytes,
        output_numel=None,
    ):
        dimensions = (world_size, offset, experts, width, hidden, workspace_limit_bytes)
        if any(type(value) is not int for value in dimensions):
            raise ValueError("split expert reduction geometry must use integers")
        if output_numel is not None and (
            type(output_numel) is not int or output_numel < 0
        ):
            raise ValueError("invalid expected reduction size")
        if world_size < 1 or offset < 0 or experts < 0 or width < 1 or hidden < 1:
            raise ValueError("invalid split expert reduction geometry")
        if workspace_limit_bytes <= 0:
            raise ValueError("invalid split expert reduction geometry")
        self.validate_parameters = None
        self.workspace_limit_bytes = workspace_limit_bytes
        self.output_numel = output_numel
        self.original = original
        self.world_size = world_size
        self.offset = offset
        self.experts = experts
        self.width = width
        self.hidden = hidden
        self.region = 3 * experts * width * hidden

    def allocate(self, *args, **kwargs):
        return self.original.allocate(*args, **kwargs)

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        from llmtuner.accelerator.collectives import synchronize_local_errors

        with synchronize_local_errors(
            "Split expert reduction preparation", group=group
        ):
            if self.validate_parameters is not None:
                self.validate_parameters()
            if async_op:
                raise ValueError("split expert reduction requires synchronous enqueue")
            if group.size() != self.world_size:
                raise ValueError("split expert reduction process group changed")
            if (
                not input_tensor.is_contiguous()
                or not output_tensor.is_contiguous()
                or input_tensor.ndim != 1
                or output_tensor.ndim != 1
                or input_tensor.dtype != output_tensor.dtype
                or input_tensor.device != output_tensor.device
                or input_tensor.untyped_storage().data_ptr()
                == output_tensor.untyped_storage().data_ptr()
            ):
                raise ValueError(
                    "split expert reduction requires distinct flat buffers"
                )
            workspace = (
                input_tensor.numel() + output_tensor.numel()
            ) * input_tensor.element_size()
            if workspace > self.workspace_limit_bytes:
                raise ValueError(
                    "split expert reduction workspace exceeds explicit budget"
                )
            if (
                self.output_numel is not None
                and output_tensor.numel() != self.output_numel
            ):
                raise ValueError(
                    "split expert reduction parameter participation changed"
                )
            inputs = input_tensor.view(self.world_size, -1)
            if inputs.shape[1] != output_tensor.numel():
                raise ValueError("split expert reduction input/output geometry changed")
            if self.offset + self.region > output_tensor.numel():
                raise ValueError("split expert reduction region exceeds buffer")
            reordered = input_tensor.clone()
            targets = reordered.view_as(inputs)
            for source, destination in zip(inputs, targets, strict=True):
                reorder_expert_region(
                    source[self.offset : self.offset + self.region],
                    destination[self.offset : self.offset + self.region],
                    experts=self.experts,
                    width=self.width,
                    hidden=self.hidden,
                    inverse=False,
                )
            reduced = torch.empty_like(output_tensor)
        self.original(reduced, reordered, group, op, async_op=False)
        output_tensor.copy_(reduced)
        reorder_expert_region(
            reduced[self.offset : self.offset + self.region],
            output_tensor[self.offset : self.offset + self.region],
            experts=self.experts,
            width=self.width,
            hidden=self.hidden,
            inverse=True,
        )
        return None


def _split_expert_reduction_plan(model, workspace_limit_bytes):
    """Install on native single-mesh, evenly sharded packed-expert FSDP units."""
    from .policies import iter_fsdp_modules

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen = set()
    pending = []
    for module in iter_fsdp_modules(model):
        state = module._get_fsdp_state()
        if id(state) in seen:
            continue
        seen.add(id(state))
        groups = state._fsdp_param_groups
        for group in groups:
            parameters = group.fsdp_params
            packed = [
                i
                for i, parameter in enumerate(parameters)
                if names.get(id(parameter.sharded_param), "").endswith(
                    ".mlp.experts.gate_up_proj"
                )
            ]
            if not packed:
                continue
            if len(groups) != 1 or len(packed) != 1:
                raise ValueError("split expert reduction requires one FSDP mesh")
            index = packed[0]
            if index + 1 >= len(parameters):
                raise ValueError("packed expert reduction requires adjacent down")
            gate, down = parameters[index : index + 2]
            if not names[id(down.sharded_param)].endswith(".mlp.experts.down_proj"):
                raise ValueError("packed expert reduction requires adjacent down")
            experts, twice_width, hidden = gate._orig_size
            width = twice_width // 2
            world = gate.mesh_info.shard_mesh_size
            if (
                twice_width % 2
                or experts % world
                or tuple(down._orig_size) != (experts, hidden, width)
                or any(p.fsdp_placement.dim != 0 for p in parameters)
            ):
                raise ValueError("split expert reduction requires even Shard(0)")
            offset = sum(
                math.ceil(p._orig_size[0] / world) * math.prod(p._orig_size[1:])
                for p in parameters[:index]
            )
            for parameter in parameters:
                expected_shape = (
                    math.ceil(parameter._orig_size[0] / world),
                    *parameter._orig_size[1:],
                )
                if tuple(parameter.padded_sharded_param_size) != expected_shape:
                    raise ValueError(
                        "split expert reduction per-parameter padding mismatch"
                    )
            output_numel = sum(
                math.prod(p.padded_sharded_param_size) for p in parameters
            )
            expected_numel = sum(
                math.ceil(p._orig_size[0] / world) * math.prod(p._orig_size[1:])
                for p in parameters
            )
            if output_numel != expected_numel or any(
                not p.sharded_param.requires_grad for p in parameters
            ):
                raise ValueError(
                    "split expert reduction FSDP padding or participation mismatch"
                )
            if gate.mesh_info.replicate_mesh_dim is not None:
                raise ValueError("split expert reduction does not support HSDP")
            if isinstance(group._reduce_scatter_comm, SplitExpertReduceScatter):
                raise ValueError("split expert reduction is already installed")
            pending.append(
                (
                    module,
                    SplitExpertReduceScatter(
                        group._reduce_scatter_comm,
                        world_size=world,
                        offset=offset,
                        experts=experts // world,
                        width=width,
                        hidden=hidden,
                        workspace_limit_bytes=workspace_limit_bytes,
                        output_numel=output_numel,
                    ),
                )
            )
    if not pending:
        raise ValueError("split expert reduction found no packed expert units")
    return pending


def install_split_expert_reduction(model, *, workspace_limit_bytes):
    """Validate every local unit and coordinate before installing any hooks."""
    from llmtuner.accelerator.collectives import synchronize_local_errors

    with synchronize_local_errors("Split expert reduction installation"):
        pending = _split_expert_reduction_plan(model, workspace_limit_bytes)
    for module, communication in pending:
        group = module._get_fsdp_state()._fsdp_param_groups[0]
        signature = tuple(
            (
                id(p),
                id(p.sharded_param),
                tuple(p._orig_size),
                tuple(p.padded_sharded_param_size),
            )
            for p in group.fsdp_params
        )

        def validate_parameters(group=group, signature=signature):
            current = tuple(
                (
                    id(p),
                    id(p.sharded_param),
                    tuple(p._orig_size),
                    tuple(p.padded_sharded_param_size),
                )
                for p in group.fsdp_params
            )
            if current != signature or not all(
                p.sharded_param.requires_grad for p in group.fsdp_params
            ):
                raise ValueError("split expert reduction parameter set changed")

        communication.validate_parameters = validate_parameters
        module.set_reduce_scatter_unused_params(True)
        module.set_custom_reduce_scatter(communication)
