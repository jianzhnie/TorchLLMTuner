"""Independent-process synchronous DCP recovery for explicit models on CPU.

Run --phase write then --phase resume with the same --folder. This checks the
production checkpoint manager, moments, scheduler, default RNG and a synthetic
cursor; it is not a real dataset, distributed or target-scale Trainer test.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.core.random_state import RNGCheckpointState
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.models.deepseek.deepseek_v4.modeling import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
)
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM


class Cursor:
    """Synthetic input position saved through the production stateful interface."""

    def __init__(self) -> None:
        self.step = 0

    def state_dict(self) -> dict[str, int]:
        return {"step": self.step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step = state["step"]


def assert_equal(actual: Any, expected: Any, path: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_equal(left, right, f"{path}.{i}")
    else:
        assert actual == expected, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen3", "deepseek_v4"), default="qwen3")
    parser.add_argument(
        "--qkv-projection", choices=("separate", "fused"), default="separate"
    )
    parser.add_argument("--freeze-projection", choices=("q_proj", "k_proj", "v_proj"))
    args = parser.parse_args()
    if args.model != "qwen3" and (
        args.qkv_projection != "separate" or args.freeze_projection is not None
    ):
        parser.error("QKV projection options apply only to Qwen3")
    seed = 71 if args.phase == "write" else 93
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        router_aux_loss_coef=0.0,
    )
    if args.model == "deepseek_v4":
        config = DeepseekV4Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            q_lora_rank=16,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=12,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            compress_rates={
                "compressed_sparse_attention": 2,
                "heavily_compressed_attention": 4,
            },
            layer_types=["compressed_sparse_attention", "sliding_attention"],
            mlp_layer_types=["hash_moe", "moe"],
            max_position_embeddings=64,
        )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.qkv_projection = args.qkv_projection
    model = (
        Qwen3MoeForCausalLM(config)
        if args.model == "qwen3"
        else DeepseekV4ForCausalLM(config)
    )
    if args.model == "deepseek_v4" and args.phase == "write":
        model.initialize_training_buffers()
    if args.freeze_projection is not None:
        getattr(model.layers[0].self_attn, args.freeze_projection).requires_grad_(False)
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=3e-4, implementation="for-loop"),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0, min_lr_factor=0.1),
        optimizers=optimizer.optimizers,
        training_steps=4,
    )
    cursor = Cursor()
    manager = DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=2, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={"cursor": cursor, "rng_state": RNGCheckpointState(torch.device("cpu"))},
        folder=str(args.folder),
    )
    losses = []
    try:
        if args.phase == "resume":
            assert manager.load(2)
            assert cursor.step == 2, "Input cursor was not restored"
            assert all(inner.state for inner in optimizer.optimizers), (
                "Adam moments were not restored"
            )
            for group in optimizer.param_groups:
                group["foreach"] = False
        frozen = tuple(
            parameter for parameter in model.parameters() if not parameter.requires_grad
        )
        frozen_values = tuple(parameter.detach().clone() for parameter in frozen)
        while cursor.step < 4:
            tokens = torch.randint(32, (2, 5))
            targets = torch.randint(32, (2, 5))
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens).logits
            scale = 1.0 + random.random() + float(np.random.random())
            loss = (
                torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten()
                )
                * scale
            )
            loss.backward()
            assert model.tok_embeddings.weight.grad.count_nonzero() > 0
            assert all(parameter.grad is None for parameter in frozen)
            optimizer.step()
            for parameter, original in zip(frozen, frozen_values, strict=True):
                torch.testing.assert_close(parameter, original, atol=0, rtol=0)
                assert all(
                    parameter not in inner.state for inner in optimizer.optimizers
                )
            scheduler.step()
            cursor.step += 1
            losses.append(float(loss.detach()))
            if args.phase == "write" and cursor.step == 2:
                assert manager.save(2)
        result = {
            "execution": (args.model, args.qkv_projection, args.freeze_projection),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cursor": cursor.state_dict(),
            "losses": losses[-2:],
            "next_random": (torch.rand(4), random.random(), float(np.random.random())),
        }
        artifact = args.folder / "continuous.pt"
        if args.phase == "write":
            torch.save(result, artifact)
        else:
            expected = torch.load(artifact, map_location="cpu", weights_only=True)
            assert_equal(result, expected)
            print(
                "Independent-process DCP: state, moments, scheduler, RNG, "
                "cursor and next steps exact"
            )
    finally:
        manager.close()


if __name__ == "__main__":
    main()
