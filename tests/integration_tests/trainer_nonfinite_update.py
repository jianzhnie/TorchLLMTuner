"""CPU DP2 FSDP: rank-local nonfinite values stop every parameter update.

Run with torchrun --nproc_per_node=2; append custom, compile and/or ac.
Add ep2 for fixed-routing MoE, or tp2 with four ranks for DP2 x TP2 MoE.
Add cp2 with four ranks for real DeepSeek V4 HCA CP2, without attention mocks.
Add pp2 with four ranks for DP2 x PP2 using the real 1F1B schedule.
Uses the real Trainer, synthetic loader, model forward/backward and optimizers.
Only the returned loss or one post-backward local gradient is corrupted.
"""

import sys
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.config.ema_config import EMAConfig
from llmtuner.models.common.moe.layer import MoE
from llmtuner.trainer.cli import parse_config
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.trainer_ema_process_resume import snapshot


def update_state(trainer):
    state = snapshot(trainer)
    return {
        key: state[key]
        for key in ("model", "optimizer", "lr_scheduler", "ema", "local_shards")
    }


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    trainer = None
    try:
        tp_size = 2 if "tp2" in sys.argv else 1
        ep_size = 2 if "ep2" in sys.argv else 1
        cp_size = 2 if "cp2" in sys.argv else 1
        pp_size = 2 if "pp2" in sys.argv else 1
        if cp_size > 1 and (tp_size > 1 or ep_size > 1 or pp_size > 1):
            raise ValueError("The HCA CP scenario requires TP1, EP1 and PP1")
        moe_enabled = tp_size > 1 or ep_size > 1 or cp_size > 1
        assert dist.get_world_size() == 2 * tp_size * cp_size * pp_size
        with TemporaryDirectory() as root:
            config = parse_config(
                [
                    "--model_name_or_path",
                    "qwen3",
                    "--vocab_size",
                    "32",
                    "--hidden_size",
                    "16",
                    "--intermediate_size",
                    "32",
                    "--num_hidden_layers",
                    "2",
                    "--num_attention_heads",
                    "2",
                    "--num_key_value_heads",
                    "2",
                    "--max_seq_len",
                    "8",
                    "--global_batch_size",
                    str(8 if pp_size > 1 else 2),
                    "--seed",
                    "42",
                    "--deterministic",
                    "true",
                    "--enable",
                    "true",
                    "--last_save_model_only",
                    "false",
                    "--total_steps",
                    "20",
                    "--steps",
                    "20",
                    "--dump_folder",
                    root,
                ]
            )
            config.parallel.data_parallel_shard_size = 2
            config.parallel.tensor_parallel_size = tp_size
            config.parallel.expert_parallel_size = ep_size
            config.parallel.pipeline_parallel_size = pp_size
            if pp_size > 1:
                config.parallel.pp_config.num_pp_microbatches = 4
            if moe_enabled and cp_size == 1:
                config.model.model_name_or_path = "qwen3_moe"
                config.model.experts_implementation = "eager"
                config.model.arch_overrides.update(
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=16,
                    router_aux_loss_coef=0.03,
                )
                config.parallel.ep_config.fixed_router = True
                config.parallel.ep_config.router_aux_loss_coef = 0.03
            if cp_size > 1:
                from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
                    DeepseekV4Experts,
                )

                from tests.integration_tests.hca_trainer_checkpoint import (
                    config as hca_config,
                )

                config.model = hca_config(root).model
                config.parallel.context_parallel_ulysses_size = cp_size
                config.parallel.cp_config.context_parallel_strategy = "ulysses"
            config.parallel.fsdp_config.fsdp_implementation = (
                "custom" if "custom" in sys.argv else "native"
            )
            config.training.compile = "compile" in sys.argv
            config.training.compile_config.backend = "aot_eager"
            config.training.activation_checkpoint_mode = (
                "full" if "ac" in sys.argv else "none"
            )
            config.training.ema_config = EMAConfig(decay=0.75)
            trainer = Trainer(config)
            # Initialize real Adam moments and give EMA nontrivial history.
            trainer.step = 1
            trainer.train_step(trainer.batch_generator(trainer.data_iterator()))
            assert trainer.lr_scheduler.state_dict()["last_epoch"] == 1
            warmed = update_state(trainer)
            assert any(
                key.startswith("state.")
                and isinstance(value, torch.Tensor)
                and bool(torch.count_nonzero(value))
                for key, value in warmed["optimizer"].items()
            ), "Adam moments were not exercised"
            assert any(
                not torch.equal(
                    value,
                    warmed["model"][
                        key.removeprefix("state.").removesuffix(".ema_params")
                    ],
                )
                for key, value in warmed["ema"].items()
            ), "EMA did not retain history distinct from the live model"
            for invalid in ("loss", "gradient"):
                for failing_rank in range(dist.get_world_size()):
                    trainer.step += 1
                    before = update_state(trainer)
                    forward_backward = trainer.forward_backward_step
                    injected = []

                    def corrupt(
                        *args,
                        forward_backward=forward_backward,
                        failing_rank=failing_rank,
                        invalid=invalid,
                        injected=injected,
                        **kwargs,
                    ):
                        loss = forward_backward(*args, **kwargs)
                        if dist.get_rank() == failing_rank:
                            if invalid == "loss":
                                injected.append(True)
                                return loss.new_tensor(float("nan"))
                            if cp_size > 1:
                                groups = [
                                    module
                                    for part in trainer.model_parts
                                    for module in part.modules()
                                    if isinstance(module, DeepseekV4Experts)
                                ]
                                assert groups, "The test did not reach HCA experts"
                            elif moe_enabled:
                                groups = [
                                    module.routed_experts.inner_experts
                                    for part in trainer.model_parts
                                    for module in part.modules()
                                    if isinstance(module, MoE)
                                ]
                                assert groups, "The test did not reach MoE experts"
                            else:
                                groups = trainer.model_parts
                            for group in groups:
                                for parameter in group.parameters():
                                    grad = parameter.grad
                                    if grad is not None:
                                        local = (
                                            grad.to_local()
                                            if isinstance(grad, DTensor)
                                            else grad
                                        )
                                        if local.numel():
                                            local.fill_(float("inf"))
                                            injected.append(True)
                                            return loss
                            raise AssertionError("No local gradient was available")
                        return loss

                    with patch.object(trainer, "forward_backward_step", corrupt):
                        try:
                            trainer.train_step(
                                trainer.batch_generator(trainer.data_iterator())
                            )
                        except RuntimeError as error:
                            assert "not finite" in str(error), str(error)
                        else:
                            raise AssertionError("Nonfinite step updated parameters")
                    assert bool(injected) == (dist.get_rank() == failing_rank)
                    assert_state_equal(update_state(trainer), before)
                    dist.barrier()
                    # CPU assertions are recoverable. New gradients must replace
                    # the failed window's values; this does not roll back data/RNG.
                    trainer.train_step(trainer.batch_generator(trainer.data_iterator()))
                    after = update_state(trainer)
                    assert after["lr_scheduler"]["last_epoch"] == trainer.step
                    assert any(
                        not torch.equal(before["model"][key], after["model"][key])
                        for key in before["model"]
                    )
                    dist.barrier()
            if config.training.compile:
                assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
            if dist.get_rank() == 0:
                print("Rank-local nonfinite steps preserve every update state.")
    finally:
        if trainer is not None:
            trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
