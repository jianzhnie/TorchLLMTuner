"""EP can need expert FSDP even when dense DP and CP have degree one.

Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=4
tests/integration_tests/ep_fsdp_without_dense_dp.py``.

Compare actual TP4/EP2/efsdp2 outputs and expert gradients with an unsharded
HF model on CPU/Gloo. Dense DP and CP both have degree one.
"""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.spmd import spmd_context
from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.ep_fsdp_equivalence import _ref_expert_grads
from tests.integration_tests.harness import init_distributed, moved


def main() -> None:
    device = init_distributed()
    assert dist.get_world_size() == 4
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=8,
            arch_overrides={
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "norm_topk_prob": True,
                "router_aux_loss_coef": 0.0,
            },
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=1,
            tensor_parallel_size=4,
            expert_parallel_size=2,
        ),
        training=TrainingConfig(max_seq_len=32, steps=1),
    )
    dims = ParallelTopology.from_config(cfg.parallel, 4)
    dims.build_device_mesh()
    torch.manual_seed(0)
    model = moved(HFTransformerModel(build_model_config_for(cfg)).to(torch.float32))
    reference = copy.deepcopy(model)
    model = parallelize_hf_transformers(
        model,
        cfg=cfg.parallel,
        mesh=dims.spmd_dense_mesh(),
        parallel_dims=dims,
        device=device,
    )
    expert = model.layers[0].mlp.routed_experts.inner_experts.w1_EFD
    assert isinstance(expert, DTensor)
    assert expert.device_mesh.size() == 2

    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 128, (32,), generator=generator)
    labels = torch.randint(0, 128, (32,), generator=generator)
    batch = {"input": moved(ids), "labels": moved(labels)}
    with spmd_context(None):
        ref_inputs, ref_labels, ref_kwargs = reference.preprocess_inputs(
            batch, parallel_dims=None
        )
        ref_logits = reference(ref_inputs, **ref_kwargs)
        (F.cross_entropy(ref_logits, ref_labels, reduction="sum") / 32).backward()
    inputs, targets, kwargs = model.preprocess_inputs(batch, parallel_dims=dims)
    with spmd_context(dims):
        logits = model(inputs, **kwargs)
        (F.cross_entropy(logits, targets, reduction="sum") / 32).backward()
    tp_rank = dims.get_mesh("tp").get_local_rank()
    token_slice = slice(tp_rank * 8, (tp_rank + 1) * 8)
    torch.testing.assert_close(logits, ref_logits[token_slice], rtol=1e-4, atol=1e-5)
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    ep_rank = dims.get_mesh("ep").get_local_rank()
    expert_slice = slice(ep_rank * 4, (ep_rank + 1) * 4)
    # Read the reference gradients in HF's own terms: 5.x fuses the experts into
    # ``gate_up_proj``/``down_proj``, 4.x keeps one module per expert, and this
    # script is otherwise anchored on what HF computed.
    ref_gate_up, ref_down = _ref_expert_grads(reference.layers[0].mlp)
    max_grad_diff = 0.0
    for parameter, expected in (
        (grouped.w1_EFD, ref_gate_up[expert_slice, :48]),
        (grouped.w3_EFD, ref_gate_up[expert_slice, 48:]),
        (grouped.w2_EDF, ref_down[expert_slice]),
    ):
        actual = parameter.grad.full_tensor()
        assert actual.abs().max() > 0
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        max_grad_diff = max(max_grad_diff, float((actual - expected).abs().max()))
    differences = torch.tensor(
        [
            float((logits.detach() - ref_logits.detach()[token_slice]).abs().max()),
            max_grad_diff,
        ],
        device=device,
    )
    dist.all_reduce(differences, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"logits max abs diff = {differences[0].item():.3e}")
        print(f"expert grads max abs diff = {differences[1].item():.3e}")
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
