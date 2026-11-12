"""Four CPU/Gloo ranks: CP/TP shape failures stop production preprocessing.

Run with torchrun --nproc_per_node=4. Uses the explicit Qwen3 model and input
sharders; does not execute attention, model parallel transforms or an optimizer.
"""

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen3MoeConfig

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.trainer.trainer import Trainer


def batch(length: int) -> dict[str, torch.Tensor]:
    tokens = torch.arange(length)
    return {
        "input": tokens,
        "labels": tokens + 1,
        "padding_mask": tokens % 3 == 0,
    }


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        assert dist.get_world_size() == 4
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                mlp_only_layers=[0],
            )
        )
        model.set_cp_mesh(mesh["cp"], load_balancer=None)
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.model_parts = [model]
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(max_seq_len=16)
        trainer.parallel_dims = SimpleNamespace(
            get_optional_mesh=lambda name: mesh[name] if name in ("cp", "tp") else None
        )
        for axis, invalid_length in (("cp", 15), ("tp", 14)):
            for failing_rank in range(4):
                source = batch(
                    invalid_length if dist.get_rank() == failing_rank else 16
                )
                try:
                    trainer.preprocess(source)
                except ValueError as error:
                    assert dist.get_rank() == failing_rank
                    assert f"not divisible by {axis}=2" in str(error)
                except RuntimeError as error:
                    assert dist.get_rank() != failing_rank
                    assert str(error) == (
                        "Training batch preprocessing failed on another rank"
                    )
                else:
                    raise AssertionError("A rank continued after invalid CP/TP input")
                dist.barrier()
                healthy = batch(16)
                inputs, labels, kwargs = trainer.preprocess(healthy)
                cp_rank = mesh["cp"].get_local_rank()
                tp_rank = mesh["tp"].get_local_rank()
                start = cp_rank * 8 + tp_rank * 4
                indices = torch.arange(start, start + 4)
                torch.testing.assert_close(inputs, indices, rtol=0, atol=0)
                torch.testing.assert_close(labels, indices + 1, rtol=0, atol=0)
                # Attention gathers TP sequence shards before QKV; mask and
                # positions therefore retain the complete CP-local sequence.
                cp_indices = torch.arange(cp_rank * 8, (cp_rank + 1) * 8)
                torch.testing.assert_close(
                    kwargs["padding_mask"], cp_indices % 3 == 0, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    kwargs["positions"],
                    cp_indices,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(healthy["input"], torch.arange(16))
                dist.barrier()
        if dist.get_rank() == 0:
            print("CP/TP shape failures coordinated; healthy retry aligned.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
