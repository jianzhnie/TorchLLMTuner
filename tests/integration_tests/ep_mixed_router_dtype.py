"""Two-rank EP regression for FP32 routers with BF16 expert parameters.

Run explicitly with torchrun --standalone --nproc_per_node=2. Compares native
local routing over both ranks' tokens with EP routing over disjoint inputs.
Checks async checkpoint cold resume and the public mixed-dtype FSDP rejection.
This is a small-model correctness check, not target-model acceptance.
--restore-local also checks EP2 to local routing with a full-token update.
"""

import argparse
import copy
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from transformers import AutoConfig

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.core.model_state import ModelCheckpointState
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import (
    CheckpointConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    ParallelConfig,
)
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.harness import init_distributed


def verify_checkpoint_resume(
    candidate,
    source,
    inputs,
    upstream,
    *,
    async_mode,
    restore_local=False,
    compile_experts=False,
):
    def services(module, folder):
        optimizer = OptimizersContainer(
            OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
            model_parts=[module],
        )
        scheduler = build_lr_scheduler(
            LRSchedulerConfig(decay_ratio=1.0),
            optimizers=optimizer.optimizers,
            training_steps=4,
        )
        manager = DistributedCheckpointManager(
            CheckpointConfig(enable=True, interval=1, async_mode=async_mode),
            model_parts=[module],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=folder,
        )
        return optimizer, scheduler, manager

    def advance(module, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        output = module(inputs.detach() * 1.25)
        output.backward(upstream)
        dist.all_reduce(module.router.gate.weight.grad)
        optimizer.step()
        scheduler.step()
        return output.detach()

    rank = dist.get_rank()
    context = TemporaryDirectory() if rank == 0 else nullcontext(None)
    with context as root:
        paths = [root]
        dist.broadcast_object_list(paths, src=0)
        optimizer, scheduler, manager = services(candidate, paths[0])
        restored_manager = None
        try:
            initial = [p.detach().clone() for p in candidate.parameters()]
            # The caller supplied checked gradients from the first backward.
            optimizer.step()
            scheduler.step()
            for parameter, previous in zip(
                candidate.parameters(), initial, strict=True
            ):
                assert not torch.equal(parameter, previous)
                state = optimizer.optimizers[0].state[parameter]
                for key in ("exp_avg", "exp_avg_sq"):
                    assert state[key].dtype == parameter.dtype
                    assert state[key].count_nonzero() > 0
                    assert torch.isfinite(state[key]).all()
            saved = copy.deepcopy(candidate.state_dict())
            full_saved = (
                copy.deepcopy(ModelCheckpointState(candidate).state_dict())
                if restore_local
                else None
            )
            full_saved_optimizer = (
                copy.deepcopy(optimizer.state_dict()) if restore_local else None
            )
            saved_scheduler = copy.deepcopy(scheduler.state_dict())
            assert manager.save(1)
            manager.maybe_wait_for_staging()
            continuous = advance(candidate, optimizer, scheduler)
            manager.close()
            restored_model = copy.deepcopy(source)
            assert (
                swap_hf_moe_blocks(
                    restored_model,
                    ep_group=dist.group.WORLD,
                    router_aux_loss_coef=0.0,
                )
                == 1
            )
            restored = restored_model.layers[0].mlp
            if compile_experts:
                restored.routed_experts.inner_experts.compile(
                    backend="aot_eager", fullgraph=True
                )
            restored_optimizer, restored_scheduler, restored_manager = services(
                restored, paths[0]
            )
            assert all(not inner.state for inner in restored_optimizer.optimizers)
            assert restored_manager.load(1)
            torch.testing.assert_close(restored.state_dict(), saved, rtol=0, atol=0)
            resumed = advance(restored, restored_optimizer, restored_scheduler)
            torch.testing.assert_close(resumed, continuous, rtol=0, atol=0)
            torch.testing.assert_close(
                restored.state_dict(), candidate.state_dict(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0
            )
            assert restored_scheduler.state_dict() == scheduler.state_dict()
            dist.barrier()
            print(
                f"rank {rank}: mixed-dtype EP {async_mode} "
                "checkpoint cold resume passed"
            )
            if restore_local:
                # Both ranks restore all experts, then run the complete token
                # batch independently. No EP or router gradient reduction is
                # permitted in this reference update.
                restored_manager.close()
                local_model = copy.deepcopy(source)
                assert swap_hf_moe_blocks(local_model, router_aux_loss_coef=0.0) == 1
                local = local_model.layers[0].mlp
                local_optimizer, local_scheduler, restored_manager = services(
                    local, paths[0]
                )
                assert all(not inner.state for inner in local_optimizer.optimizers)
                assert restored_manager.load(1)
                torch.testing.assert_close(
                    local.state_dict(), full_saved, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    local_optimizer.state_dict(), full_saved_optimizer, rtol=0, atol=0
                )
                assert local_scheduler.state_dict() == saved_scheduler
                batches = [
                    torch.empty_like(inputs) for _ in range(dist.get_world_size())
                ]
                gradients = [torch.empty_like(upstream) for _ in batches]
                outputs = [torch.empty_like(continuous) for _ in batches]
                dist.all_gather(batches, inputs.detach())
                dist.all_gather(gradients, upstream)
                dist.all_gather(outputs, continuous)
                local_optimizer.zero_grad(set_to_none=True)
                local_output = local(torch.cat(batches) * 1.25)
                local_output.backward(torch.cat(gradients))
                local_optimizer.step()
                local_scheduler.step()
                # EP changes GEMM batching and BF16 accumulation. Restore is
                # exact above; forward/update comparisons use the declared
                # arithmetic tolerance of this regression.
                torch.testing.assert_close(
                    local_output, torch.cat(outputs), rtol=5e-3, atol=5e-3
                )
                torch.testing.assert_close(
                    local.state_dict(),
                    ModelCheckpointState(candidate).state_dict(),
                    rtol=5e-3,
                    atol=5e-3,
                )
                torch.testing.assert_close(
                    local_optimizer.state_dict(),
                    optimizer.state_dict(),
                    rtol=5e-3,
                    atol=5e-3,
                )
                assert local_scheduler.state_dict() == scheduler.state_dict()
                print(f"rank {rank}: EP2 to local MoE cold resume and update passed")
        finally:
            manager.close()
            if restored_manager is not None:
                restored_manager.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--async-mode",
        choices=("disabled", "async", "async_with_pinned_mem"),
        default="async",
    )
    parser.add_argument(
        "--restore-local",
        action="store_true",
        help="Also restore EP2 into native local routing and compare the next update",
    )
    parser.add_argument("--compile-experts", action="store_true")
    args = parser.parse_args()
    device = init_distributed()
    try:
        assert device.type == "npu", "This regression requires real NPU execution"
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(52)
        config = AutoConfig.for_model(
            "qwen3_moe",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        source = HFTransformerModel(config).to(device=device, dtype=torch.bfloat16)
        source.layers[0].mlp.gate.float()
        local = copy.deepcopy(source)
        sharded = copy.deepcopy(source)
        original_block = sharded.layers[0].mlp
        original_parameters = tuple(sharded.parameters())
        original_state = copy.deepcopy(sharded.state_dict())
        original_counts = dict(AuxLoss._group_counts)
        failure = RuntimeError("intentional EP replacement observer failure")

        def reject_replacement(modules, parameters):
            assert original_block in modules
            assert parameters
            raise failure

        try:
            swap_hf_moe_blocks(
                sharded,
                ep_group=dist.group.WORLD,
                on_replacement=reject_replacement,
            )
        except RuntimeError as error:
            assert error is failure
        else:
            raise AssertionError("EP replacement observer did not reject")
        assert sharded.layers[0].mlp is original_block
        assert all(
            before is after
            for before, after in zip(
                original_parameters, sharded.parameters(), strict=True
            )
        )
        torch.testing.assert_close(sharded.state_dict(), original_state, rtol=0, atol=0)
        assert AuxLoss._group_counts == original_counts
        print(f"rank {rank}: EP observer failure preserved model and aux counts")
        assert swap_hf_moe_blocks(local, router_aux_loss_coef=0.0) == 1
        assert (
            swap_hf_moe_blocks(
                sharded, ep_group=dist.group.WORLD, router_aux_loss_coef=0.0
            )
            == 1
        )
        reference = local.layers[0].mlp
        candidate = sharded.layers[0].mlp
        if args.compile_experts:
            candidate.routed_experts.inner_experts.compile(
                backend="aot_eager", fullgraph=True
            )
        # The public assembly path rejects mixed parameter dtypes instead of
        # casting the FP32 router under a uniform FSDP policy.
        fsdp_config = ParallelConfig(data_parallel_shard_size=2)
        topology = ParallelTopology.from_config(fsdp_config, world_size=2)
        topology.build_device_mesh()
        original_parameters = tuple(local.parameters())
        try:
            apply_fsdp(local, fsdp_config, topology)
        except ValueError as error:
            assert "single parameter dtype" in str(error), str(error)
        else:
            raise AssertionError("Mixed-dtype FSDP must reject before wrapping")
        assert local.layers[0].mlp is reference
        assert all(
            before is after
            for before, after in zip(
                original_parameters, local.parameters(), strict=True
            )
        )
        for module in (reference, candidate):
            assert module.router.gate.weight.dtype == torch.float32
            assert all(
                parameter.dtype == torch.bfloat16
                for parameter in module.routed_experts.inner_experts.parameters()
            )
        full_input = torch.randn(12, 16, device=device, dtype=torch.bfloat16)
        upstream = torch.randn_like(full_input)
        full_input.requires_grad_()
        shard_input = full_input.detach().chunk(2)[rank].clone().requires_grad_()
        expected = reference(full_input)
        actual = candidate(shard_input)
        # BF16 grouped matmul accumulation can vary with dispatch batching.
        torch.testing.assert_close(
            actual, expected.chunk(2)[rank], rtol=5e-3, atol=5e-3
        )
        expected.backward(upstream)
        actual.backward(upstream.chunk(2)[rank])
        torch.testing.assert_close(
            shard_input.grad, full_input.grad.chunk(2)[rank], rtol=5e-3, atol=5e-3
        )
        actual_router_grad = candidate.router.gate.weight.grad
        dist.all_reduce(actual_router_grad)
        torch.testing.assert_close(
            actual_router_grad,
            reference.router.gate.weight.grad,
            rtol=5e-3,
            atol=5e-3,
        )
        expected_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0, name
            target = expected_parameters[name]
            gradient = target.grad
            if name.startswith("routed_experts."):
                gradient = gradient.chunk(2)[rank]
                torch.testing.assert_close(
                    parameter, target.chunk(2)[rank], rtol=0, atol=0
                )
            torch.testing.assert_close(parameter.grad, gradient, rtol=5e-3, atol=5e-3)
        print(f"rank {rank}: FP32 router/BF16 expert EP forward and backward passed")
        verify_checkpoint_resume(
            candidate,
            source,
            shard_input,
            upstream.chunk(2)[rank],
            async_mode=args.async_mode,
            restore_local=args.restore_local,
            compile_experts=args.compile_experts,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
