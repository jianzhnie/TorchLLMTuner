"""EP can need expert FSDP even when dense DP and CP have degree one.

Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=4
tests/integration_tests/ep_fsdp_without_dense_dp.py``.

Compare actual TP4/EP2/efsdp2 outputs and all parameter gradients with an
unsharded HF model. Dense DP and CP both have degree one.
Set LLMTUNER_TEST_DP_MODE=replicate or shard with eight ranks to add DP2
and compare all gradients with the mean over distinct DP inputs.
Set LLMTUNER_TEST_OPTIMIZER_STEPS=3 to compare three SGD updates, including
all expert and dense weights, with the evolving independent HF reference.
Set LLMTUNER_TEST_EXPERT_COMPILE=1 for NPU BF16 expert aot_eager/fullgraph
coverage; this opt-in uses declared rtol=atol=5e-3 for BF16 kernel differences.
Also set LLMTUNER_TEST_COMPILE_STAGE=1 to compile decoder blocks through the
production compile stage before FSDP instead of compiling only experts afterward.
Set LLMTUNER_TEST_AC_MODE=full or selective to exercise production AC ordering.
Set LLMTUNER_TEST_BF16=1 without compile flags for an eager BF16 control.
Set LLMTUNER_TEST_CHECK_AC_REPLAY=1 with eager full AC to require identical
post-attention norm inputs in forward and replay (diagnostic hooks, no timing).
Set LLMTUNER_TEST_DETERMINISTIC=1 with HCCL_DETERMINISTIC=true for the
combined diagnostic control, or use 'torch' to change only PyTorch's policy.
Neither option changes the production seeding policy or the default fixture.
"""

import copy
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from torch.distributed.tensor import DTensor, Replicate

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.spmd import spmd_context
from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig
from llmtuner.config.compile_config import CompileConfig
from llmtuner.models.common.aux_loss import in_checkpoint_recompute
from llmtuner.parallel.parallelize import parallelize_model
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.runtime.gradients import allreduce_replicated_tp_grads
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.ep_fsdp_equivalence import _ref_expert_grads
from tests.integration_tests.harness import init_distributed, moved


def _expert_weights(block: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    experts = block.experts
    if hasattr(experts, "gate_up_proj"):
        return experts.gate_up_proj, experts.down_proj
    names = (
        ("gate_proj", "up_proj", "down_proj")
        if hasattr(experts[0], "gate_proj")
        else ("w1", "w3", "w2")
    )
    gate, up, down = names
    return (
        torch.stack(
            [
                torch.cat(
                    [getattr(expert, gate).weight, getattr(expert, up).weight], dim=0
                )
                for expert in experts
            ]
        ),
        torch.stack([getattr(expert, down).weight for expert in experts]),
    )


def main() -> None:
    device = init_distributed()
    deterministic = os.environ.get("LLMTUNER_TEST_DETERMINISTIC", "0")
    assert deterministic in {"0", "1", "torch"}
    if deterministic != "0":
        # Diagnostic control only: do not silently change the default baseline.
        if deterministic == "1":
            assert os.environ.get("HCCL_DETERMINISTIC", "").lower() == "true"
        torch.use_deterministic_algorithms(True)
    if dist.get_rank() == 0:
        print(
            "execution controls: "
            f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()} "
            f"HCCL_DETERMINISTIC={os.environ.get('HCCL_DETERMINISTIC', '<unset>')} "
            f"TASK_QUEUE_ENABLE={os.environ.get('TASK_QUEUE_ENABLE', '<unset>')}",
            flush=True,
        )
    compiled = os.environ.get("LLMTUNER_TEST_EXPERT_COMPILE", "0") == "1"
    compile_stage = os.environ.get("LLMTUNER_TEST_COMPILE_STAGE", "0") == "1"
    assert not compile_stage or compiled
    assert not compiled or device.type == "npu"
    bf16 = compiled or os.environ.get("LLMTUNER_TEST_BF16", "0") == "1"
    dtype = torch.bfloat16 if bf16 else torch.float32
    rtol, atol = (5e-3, 5e-3) if bf16 else (1e-4, 1e-5)
    ac_mode = os.environ.get("LLMTUNER_TEST_AC_MODE", "none")
    assert ac_mode in {"none", "full", "selective"}
    dp_mode = os.environ.get("LLMTUNER_TEST_DP_MODE", "none")
    assert dp_mode in {"none", "replicate", "shard"}
    dp_size = 1 if dp_mode == "none" else 2
    assert dist.get_world_size() == 4 * dp_size
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
            data_parallel_shard_size=2 if dp_mode == "shard" else 1,
            data_parallel_replicate_size=2 if dp_mode == "replicate" else 1,
            tensor_parallel_size=4,
            expert_parallel_size=2,
        ),
        training=TrainingConfig(max_seq_len=32, steps=1),
    )
    dims = ParallelTopology.from_config(cfg.parallel, 4 * dp_size)
    dims.build_device_mesh()
    torch.manual_seed(0)
    model = moved(HFTransformerModel(build_model_config_for(cfg)).to(dtype))
    reference = copy.deepcopy(model)
    model = parallelize_model(
        model,
        cfg=cfg.parallel,
        mesh=dims.spmd_dense_mesh(),
        parallel_dims=dims,
        device=device,
        activation_checkpoint=ac_mode,
        selective_ac=cfg.training.selective_ac,
        compile=compile_stage,
        compile_config=CompileConfig(backend="aot_eager", per_block=True),
    )
    if ac_mode != "none":
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        assert sum(isinstance(m, CheckpointWrapper) for m in model.modules()) == 1
    if compiled and not compile_stage:
        grouped = model.layers[0].mlp.routed_experts.inner_experts
        assert grouped.use_grouped_mm
        grouped.compile(backend="aot_eager", fullgraph=True)
    expert = model.layers[0].mlp.routed_experts.inner_experts.w1_EFD
    assert isinstance(expert, DTensor)
    assert expert.device_mesh.size() == 2 * dp_size
    if dp_mode == "replicate":
        for parameter in model.parameters():
            assert isinstance(parameter, DTensor)
            axis = parameter.device_mesh.mesh_dim_names.index("dp_replicate")
            assert parameter.device_mesh.size(axis) == 2
            assert isinstance(parameter.placements[axis], Replicate)
    if dp_mode == "shard":
        embedding = model.tok_embeddings.weight
        assert isinstance(embedding, DTensor)
        assert embedding.to_local().numel() < embedding.numel()
    dp_mesh = dims.get_optional_mesh("dp")
    dp_rank = dp_mesh.get_local_rank() if dp_size > 1 else 0

    check_replay = os.environ.get("LLMTUNER_TEST_CHECK_AC_REPLAY", "0") == "1"
    replay_inputs = {}
    replay_matches = []
    if check_replay:
        assert ac_mode == "full" and not compiled

        def check_norm_input(module, inputs):
            with torch.no_grad():
                if in_checkpoint_recompute.get():
                    replay_matches.append(torch.equal(inputs[0], replay_inputs["norm"]))
                else:
                    replay_inputs["norm"] = inputs[0].detach().clone()

        model.layers[0].post_attention_layernorm.register_forward_pre_hook(
            check_norm_input
        )

    trace = os.environ.get("LLMTUNER_TEST_COMPILE_TRACE", "0") == "1"
    snapshots = None
    if trace:
        from tests.integration_tests.compile_boundary_trace import (
            install_compile_boundary_trace,
            report_compile_boundary_trace,
        )

        snapshots = install_compile_boundary_trace(reference, model)

    steps = int(os.environ.get("LLMTUNER_TEST_OPTIMIZER_STEPS", "1"))
    assert steps > 0
    start_step = int(os.environ.get("LLMTUNER_TEST_START_STEP", "0"))
    assert start_step in (0, 1)
    assert start_step == 0 or os.environ.get("LLMTUNER_TEST_REPLAY_STEP1_STATE_DIR")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    for step in range(start_step, start_step + steps):
        if step == 1 and (
            replay_dir := os.environ.get("LLMTUNER_TEST_REPLAY_STEP1_STATE_DIR")
        ):
            from tests.integration_tests.compile_boundary_trace import (
                replay_compile_training_weights,
            )

            with synchronize_local_errors("TP/EP diagnostic weight replay"):
                replay_compile_training_weights(replay_dir, dist.get_rank(), model)
                if start_step == 1:
                    replay_compile_training_weights(
                        replay_dir, dist.get_rank(), reference, label="reference"
                    )
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(7 + dp_rank + 101 * step)
        ids = torch.randint(0, 128, (32,), generator=generator)
        labels = torch.randint(0, 128, (32,), generator=generator)
        batch = {"input": moved(ids), "labels": moved(labels)}
        with spmd_context(None):
            ref_inputs, ref_labels, ref_kwargs = reference.preprocess_inputs(
                batch, parallel_dims=None
            )
            ref_logits = reference(ref_inputs, **ref_kwargs)
            reference_loss = (
                F.cross_entropy(ref_logits.float(), ref_labels, reduction="sum") / 32
            )
            reference_loss.backward()
        reference_global_loss = reference_loss.detach().clone()
        if dp_size > 1:
            dist.all_reduce(reference_global_loss, group=dp_mesh.get_group())
            reference_global_loss.div_(dp_size)
        if dp_size > 1:
            gradient_gap = 0.0
            for parameter in reference.parameters():
                assert parameter.grad is not None
                peers = [torch.empty_like(parameter.grad) for _ in range(dp_size)]
                dist.all_gather(peers, parameter.grad, group=dp_mesh.get_group())
                gradient_gap = max(
                    gradient_gap, float((peers[0] - peers[1]).abs().max())
                )
                dist.all_reduce(parameter.grad, group=dp_mesh.get_group())
                parameter.grad.div_(dp_size)
            assert gradient_gap > 1e-3
        inputs, targets, kwargs = model.preprocess_inputs(batch, parallel_dims=dims)
        with spmd_context(dims):
            with (
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                if compiled
                else nullcontext()
            ) as profiler:
                logits = model(inputs, **kwargs)
            if compiled:
                assert any(
                    event.key == "llmtuner::aligned_npu_grouped_mm"
                    for event in profiler.key_averages()
                ), "compiled aligned grouped GEMM did not run"
            # Match production FP32 loss accumulation. BF16 sums round at
            # different token partition sizes and confound TP loss comparison.
            local_loss = F.cross_entropy(logits.float(), targets, reduction="sum") / (
                32 * dp_size
            )
            local_loss.backward()
        if check_replay:
            with synchronize_local_errors("TP/EP checkpoint replay comparison"):
                assert replay_matches == [True], replay_matches
            replay_matches.clear()
            replay_inputs.clear()
        global_loss = local_loss.detach().clone()
        dist.all_reduce(global_loss)
        torch.testing.assert_close(
            global_loss, reference_global_loss, rtol=rtol, atol=atol
        )
        allreduce_replicated_tp_grads([model], parallel_dims=dims)
        tp_rank = dims.get_mesh("tp").get_local_rank()
        token_slice = slice(tp_rank * 8, (tp_rank + 1) * 8)
        with synchronize_local_errors("TP/EP trace comparison"):
            if trace and dist.get_rank() == 5:
                report_compile_boundary_trace(snapshots, step, token_slice)
        # Stop all peers before the upcoming expert-gradient full_tensor
        # collective when one rank's local logits comparison fails.
        with synchronize_local_errors("TP/EP logits comparison"):
            torch.testing.assert_close(
                logits, ref_logits[token_slice], rtol=rtol, atol=atol
            )
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
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            max_grad_diff = max(max_grad_diff, float((actual - expected).abs().max()))
        reference_parameters = dict(reference.named_parameters())
        dense_grad_diff = 0.0
        dense_count = 0
        for name, parameter in model.named_parameters():
            name = canonical_fqn(name)
            if "routed_experts.inner_experts." in name:
                continue
            reference_name = name.replace(".router.gate.weight", ".gate.weight")
            reference_parameter = reference_parameters[reference_name]
            expected = reference_parameter.grad
            assert expected is not None and parameter.grad is not None, name
            actual = (
                parameter.grad.full_tensor()
                if isinstance(parameter.grad, DTensor)
                else parameter.grad
            )
            if any(
                name.endswith(f".{projection}.weight")
                for projection in ("q_proj", "k_proj", "v_proj")
            ):
                expected = expected.chunk(4, dim=0)[tp_rank]
            elif name.endswith(".o_proj.weight"):
                expected = expected.chunk(4, dim=1)[tp_rank]
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            dense_grad_diff = max(
                dense_grad_diff, float((actual - expected).abs().max())
            )
            dense_count += 1
        reference_expert_ids = {
            id(parameter) for parameter in reference.layers[0].mlp.experts.parameters()
        }
        assert dense_count == sum(
            id(parameter) not in reference_expert_ids
            for parameter in reference_parameters.values()
        )
        optimizer.step()
        reference_optimizer.step()
        gate_up_weight, down_weight = _expert_weights(reference.layers[0].mlp)
        update_oracles = [
            (grouped.w1_EFD, gate_up_weight[expert_slice, :48]),
            (grouped.w3_EFD, gate_up_weight[expert_slice, 48:]),
            (grouped.w2_EDF, down_weight[expert_slice]),
        ]
        for name, parameter in model.named_parameters():
            name = canonical_fqn(name)
            if "routed_experts.inner_experts." in name:
                continue
            reference_name = name.replace(".router.gate.weight", ".gate.weight")
            expected_weight = reference_parameters[reference_name].detach()
            if any(
                name.endswith(f".{projection}.weight")
                for projection in ("q_proj", "k_proj", "v_proj")
            ):
                expected_weight = expected_weight.chunk(4, dim=0)[tp_rank]
            elif name.endswith(".o_proj.weight"):
                expected_weight = expected_weight.chunk(4, dim=1)[tp_rank]
            update_oracles.append((parameter, expected_weight))
        assert len(update_oracles) == dense_count + 3
        weight_diff = 0.0
        for parameter, expected_weight in update_oracles:
            actual_weight = (
                parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            ).detach()
            expected_weight = expected_weight.detach()
            torch.testing.assert_close(
                actual_weight, expected_weight, rtol=rtol, atol=atol
            )
            weight_diff = max(
                weight_diff, float((actual_weight - expected_weight).abs().max())
            )
        differences = torch.tensor(
            [
                float((logits.detach() - ref_logits.detach()[token_slice]).abs().max()),
                max_grad_diff,
                dense_grad_diff,
                weight_diff,
                float((global_loss - reference_global_loss).abs()),
            ],
            device=device,
        )
        dist.all_reduce(differences, op=dist.ReduceOp.MAX)
        if state_dir := os.environ.get("LLMTUNER_TEST_COMPILE_STATE_DIR"):
            from tests.integration_tests.compile_boundary_trace import (
                dump_compile_training_state,
            )

            with synchronize_local_errors("TP/EP local training state export"):
                dump_compile_training_state(
                    state_dir, dist.get_rank(), step, model, reference, logits
                )
        if dist.get_rank() == 0:
            print(
                f"step={step + 1} dp_mode={dp_mode} dp_size={dp_size} "
                f"tp=4 ep=2 dtype={dtype} compiled_experts={compiled} "
                f"compile_stage={compile_stage} ac={ac_mode}"
            )
            print(f"logits max abs diff = {differences[0].item():.3e}")
            print(f"expert grads max abs diff = {differences[1].item():.3e}")
            print(f"dense grads max abs diff = {differences[2].item():.3e}")
            print(f"dense parameters checked = {dense_count}")
            print(f"updated weights max abs diff = {differences[3].item():.3e}")
            print(f"global loss abs diff = {differences[4].item():.3e}")
            print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
