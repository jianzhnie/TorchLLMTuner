"""Two-rank NPU DCP resume with an externally shared router bias buffer.

Run with torchrun --standalone --nproc-per-node=2 -m
 tests.integration_tests.ep_router_bias_alias_checkpoint --async-mode async.
Checks real EP training, alias identity, cold/repeated load and the next update.
Use --phase save --directory PATH followed by --phase resume --directory PATH
in separate torchrun processes to verify process-independent recovery.
This small-model regression does not establish target-model acceptance.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from transformers import AutoConfig

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import CheckpointConfig, LRSchedulerConfig, OptimizerConfig
from llmtuner.models.common.moe.bias_update import update_expert_bias
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.integration_tests.harness import init_distributed


def build_model(device):
    torch.manual_seed(52)
    config = AutoConfig.for_model(
        "deepseek_v3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        experts_implementation="eager",
    )
    source = HFTransformerModel(config).to(device).float()
    source.register_buffer(
        "external_bias", source.layers[0].mlp.gate.e_score_correction_bias
    )
    assert (
        swap_hf_moe_blocks(source, ep_group=dist.group.WORLD, router_aux_loss_coef=0.0)
        == 1
    )
    model = torch.nn.Module()
    model.block = source.layers[0].mlp
    model.register_buffer("external_bias", source.external_bias)
    assert model.external_bias is model.block.expert_bias_E
    return model


def services(model, folder, async_mode):
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0),
        optimizers=optimizer.optimizers,
        training_steps=4,
    )
    manager = DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, async_mode=async_mode),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=folder,
    )
    return optimizer, scheduler, manager


def advance(model, optimizer, scheduler, inputs):
    optimizer.zero_grad(set_to_none=True)
    before_weights = [parameter.detach().clone() for parameter in model.parameters()]
    output = model.block(inputs)
    assert output.count_nonzero() > 0
    output.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if not name.startswith("block.routed_experts."):
            dist.all_reduce(parameter.grad)
    # Deterministic, nonuniform global token statistics exercise a real bias
    # update independent of whether this tiny random batch balances by chance.
    model.block.tokens_per_expert_E.copy_(
        torch.arange(4, device=inputs.device, dtype=torch.float32)
    )
    assert any(parameter.grad.count_nonzero() > 0 for parameter in model.parameters())
    previous = model.external_bias.clone()
    update_expert_bias([model.block])
    assert not torch.equal(previous, model.external_bias)
    assert model.external_bias is model.block.expert_bias_E
    optimizer.step()
    scheduler.step()
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(model.parameters(), before_weights, strict=True)
    )
    return output.detach()


def capture_reference(device, folder, async_mode):
    model = build_model(device)
    optimizer, scheduler, manager = services(model, folder, async_mode)
    try:
        torch.manual_seed(80 + dist.get_rank())
        inputs = torch.randn(3, 5, 32, device=device)
        advance(model, optimizer, scheduler, inputs)
        reference = {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "async_mode": async_mode,
            "inputs": inputs,
            "saved_model": deepcopy(model.state_dict()),
            "saved_optimizer": deepcopy(optimizer.state_dict()),
            "saved_scheduler": deepcopy(scheduler.state_dict()),
        }
        assert manager.save(1)
        manager.maybe_wait_for_staging()
        reference["output"] = advance(model, optimizer, scheduler, inputs * 1.25)
        reference["next_model"] = deepcopy(model.state_dict())
        reference["next_optimizer"] = deepcopy(optimizer.state_dict())
        reference["next_scheduler"] = deepcopy(scheduler.state_dict())
        return reference
    finally:
        manager.close()


def verify_resume(device, folder, async_mode, reference):
    assert reference["rank"] == dist.get_rank()
    assert reference["world_size"] == dist.get_world_size()
    assert reference["async_mode"] == async_mode
    model = build_model(device)
    optimizer, scheduler, manager = services(model, folder, async_mode)
    try:
        assert all(not inner.state for inner in optimizer.optimizers)
        for _ in range(2):
            assert manager.load(1)
            assert model.external_bias is model.block.expert_bias_E
            torch.testing.assert_close(
                model.state_dict(), reference["saved_model"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state_dict(), reference["saved_optimizer"], rtol=0, atol=0
            )
            assert scheduler.state_dict() == reference["saved_scheduler"]
            actual = advance(model, optimizer, scheduler, reference["inputs"] * 1.25)
            torch.testing.assert_close(actual, reference["output"], rtol=0, atol=0)
            torch.testing.assert_close(
                model.state_dict(), reference["next_model"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state_dict(), reference["next_optimizer"], rtol=0, atol=0
            )
            assert scheduler.state_dict() == reference["next_scheduler"]
        print(f"PASS external router bias DCP resume rank={dist.get_rank()}")
    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--async-mode", choices=("disabled", "async"), default="async")
    parser.add_argument("--phase", choices=("all", "save", "resume"), default="all")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.phase != "all" and args.directory is None:
        parser.error("--directory is required for separate save/resume processes")
    device = init_distributed()
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        context = (
            TemporaryDirectory()
            if args.directory is None
            else nullcontext(str(args.directory))
        )
        with context as folder:
            folders = [folder]
            dist.broadcast_object_list(folders, src=0)
            folder = Path(folders[0])
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / f"reference-rank{dist.get_rank()}.pt"
            if args.phase == "resume":
                # Keep CPU counters on CPU and NPU tensors on their saved rank device.
                reference = torch.load(artifact, weights_only=True)
            else:
                reference = capture_reference(device, str(folder), args.async_mode)
                if args.phase == "save":
                    torch.save(reference, artifact)
            if args.phase != "save":
                verify_resume(device, str(folder), args.async_mode, reference)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
