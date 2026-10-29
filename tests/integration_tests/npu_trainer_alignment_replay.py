"""Replay real baseline batches through the complete candidate Trainer loop.

PYTHONPATH=. python tests/integration_tests/npu_trainer_alignment_replay.py
Requires artifacts from npu_baseline_training_capture.py in FP32 debugmodel mode.
For distributed capture, use torchrun with the same rank count and FSDP layout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.seeding import set_deterministic_algorithms


class CapturedLoader:
    def __init__(self, batches, forwards):
        self.batches = batches
        self.forwards = forwards
        self.cursor = 0

    def __iter__(self):
        for batch, forward in zip(self.batches, self.forwards, strict=True):
            values = batch["kwargs"]
            self.cursor += 1
            yield {
                "input": values["input_dict"]["input"],
                "labels": values["labels"],
                "positions": forward["kwargs"]["positions"],
            }

    def state_dict(self):
        return {"cursor": self.cursor}

    def close(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("../torchtitan-npu/scripts/numerical_alignment/utils/metrics.py"),
    )
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    set_deterministic_algorithms()
    if world > 1:
        from tests.integration_tests.harness import init_distributed

        init_distributed()
    root = args.capture
    if world > 1:
        root = root / f"rank{rank}"
        args.output = args.output / f"rank{rank}"
    baseline_config = json.loads((root / "config.json").read_text())
    assert baseline_config["training"]["mixed_precision_param"] == "float32"
    allow_hf32 = baseline_config["training"]["extension"]["allow_hf32"]
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = allow_hf32
    batches = torch.load(root / "consumed-batches.pt", weights_only=False)
    forwards = torch.load(root / "forward-batches.pt", weights_only=False)
    baseline_metrics = torch.load(root / "step-metrics.pt", weights_only=False)
    for forward in forwards:
        positions = forward["kwargs"]["positions"]
        captured_mask = forward["kwargs"]["attention_masks"]["dense_token_mask"]
        length = positions.numel()
        documents = (positions.flatten() == 0).cumsum(dim=0)
        document_mask = documents[:, None] == documents[None, :]
        torch.testing.assert_close(
            captured_mask[..., :length, :length],
            (document_mask & torch.ones(length, length, dtype=torch.bool).tril())[
                None, None
            ],
            atol=0,
            rtol=0,
        )
    steps = len(batches)
    args.output.mkdir(parents=True, exist_ok=False)
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=3072,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=8,
            norm_backend=os.environ.get("LLMTUNER_ALIGNMENT_NORM_BACKEND", "native"),
            qkv_projection=os.environ.get(
                "LLMTUNER_ALIGNMENT_QKV_PROJECTION", "separate"
            ),
            rotary_init_device=os.environ.get(
                "LLMTUNER_ALIGNMENT_ROTARY_INIT_DEVICE", "constructor"
            ),
            attention_backend=os.environ.get(
                "LLMTUNER_ALIGNMENT_ATTENTION_BACKEND", "auto"
            ),
            arch_overrides={
                "head_dim": 128,
                "tie_word_embeddings": True,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            },
        ),
        optimizer=OptimizerConfig(
            learning_rate=8e-4,
            weight_decay=0.1,
            betas=[0.9, 0.95],
            implementation=baseline_config["optimizer"]["implementation"],
            lr_scheduler_config=LRSchedulerConfig(
                warmup_steps=2, total_steps=steps, decay_ratio=0.8
            ),
        ),
        training=TrainingConfig(
            steps=steps,
            max_seq_len=32,
            global_batch_size=world,
            dump_folder=str(args.output),
            deterministic=True,
        ),
    )
    trainer = Trainer(config)
    model = trainer.model_parts[0]
    if os.environ.get("LLMTUNER_ALIGNMENT_EAGER_ATTENTION") == "1":
        assert model.model.config._attn_implementation == "sdpa"
        model.model.config._attn_implementation = "eager"
        print("Diagnostic: selected HF eager attention arithmetic")
    if os.environ.get("LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ") == "1":
        rotary = model.model.model.rotary_emb
        assert rotary.inv_freq.numel() == 64
        rotary.inv_freq = 1.0 / (
            1000000.0
            ** (torch.arange(0, 128, 2, device=rotary.inv_freq.device).float() / 128)
        )
        print("Diagnostic: rebuilt debugmodel RoPE inverse frequencies on NPU")
    if os.environ.get("LLMTUNER_ALIGNMENT_TORCH_RMS_NORM") == "1":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        def torch_rms_norm(module, hidden_states):
            return torch.nn.functional.rms_norm(
                hidden_states,
                (module.weight.numel(),),
                module.weight,
                module.variance_epsilon,
            )

        replaced_norms = 0
        for module in model.modules():
            if isinstance(module, Qwen3RMSNorm):
                module.forward = MethodType(torch_rms_norm, module)
                replaced_norms += 1
        assert replaced_norms == 33, replaced_norms
        print(f"Diagnostic: replaced {replaced_norms} HF norms with torch RMSNorm")
    if os.environ.get("LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS") == "1":
        from llmtuner.models.hf.model_setup import _isolate_norm_input

        removed_hooks = 0
        for module in model.modules():
            for key, hook in list(module._forward_pre_hooks.items()):
                if hook is _isolate_norm_input:
                    del module._forward_pre_hooks[key]
                    module._forward_pre_hooks_with_kwargs.pop(key, None)
                    removed_hooks += 1
        assert removed_hooks == 16, removed_hooks
        print(f"Diagnostic: removed {removed_hooks} residual norm view hooks")
    if world > 1:
        from torch.distributed.fsdp import FSDPModule

        assert isinstance(model, FSDPModule)
        assert trainer.parallel_dims.dp_shard == world
    initial = torch.load(root / "initial-hf.pt", weights_only=True)
    assert model.model.lm_head.weight is model.model.model.embed_tokens.weight
    assert "lm_head.weight" not in initial
    initial["lm_head.weight"] = initial["model.embed_tokens.weight"]
    if world > 1:
        sharded_initial = {}
        for name, parameter in model.model.state_dict().items():
            assert isinstance(parameter, DTensor), name
            sharded_initial[name] = distribute_tensor(
                initial[name].to(parameter.device),
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
        model.model.load_state_dict(sharded_initial, strict=True)
    else:
        model.model.load_state_dict(initial, strict=True)
    for name, parameter in model.model.state_dict().items():
        if isinstance(parameter, DTensor):
            parameter = parameter.full_tensor()
        torch.testing.assert_close(parameter.cpu(), initial[name], atol=0, rtol=0)
    trainer_module = importlib.import_module("llmtuner.trainer.trainer")
    original_clip = trainer_module.clip_grad_norm_
    reorder_norm = os.environ.get("LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER") == "1"
    if reorder_norm:
        assert baseline_config["model_spec"]["flavor"] == "debugmodel_non_fused_qkv"
        named_parameters = dict(model.model.named_parameters())
        ordered_names = [name for name in initial if name in named_parameters]
        assert set(ordered_names) == set(named_parameters)
        parameter_order = {
            id(named_parameters[name]): index
            for index, name in enumerate(ordered_names)
        }

        def ordered_clip(parameters, *clip_args, **clip_kwargs):
            parameters = list(parameters)
            assert {id(parameter) for parameter in parameters} == set(parameter_order)
            parameters.sort(key=lambda parameter: parameter_order[id(parameter)])
            return original_clip(parameters, *clip_args, **clip_kwargs)

        trainer_module.clip_grad_norm_ = ordered_clip
        (args.output / "norm-parameter-order.json").write_text(
            json.dumps(ordered_names, indent=2)
        )
        print("Diagnostic: gradient norm uses baseline mapped parameter order")
    trainer.dataloader.close()
    trainer.dataloader = CapturedLoader(batches, forwards)
    module_trace = {}
    module_handles = []
    if os.environ.get("LLMTUNER_ALIGNMENT_MODULE_TRACE") == "1":
        from tests.integration_tests.npu_baseline_training_capture import (
            observe_first_step_modules,
        )

        modules = {"embedding": model.tok_embeddings, "norm": model.norm}
        modules["rotary_cache"] = model.model.model.rotary_emb
        for index, layer in enumerate(model.layers):
            for name, attribute in (
                ("input_norm", "input_layernorm"),
                ("attention", "self_attn"),
                ("post_norm", "post_attention_layernorm"),
                ("mlp", "mlp"),
            ):
                modules[f"layer{index}.{name}"] = getattr(layer, attribute)
            modules[f"layer{index}"] = layer
            for name, attribute in (
                ("q_norm", "q_norm"),
                ("k_norm", "k_norm"),
                ("output_projection", "o_proj"),
            ):
                modules[f"layer{index}.{name}"] = getattr(layer.self_attn, attribute)
        module_trace, module_handles = observe_first_step_modules(trainer, modules)
        if os.environ.get("LLMTUNER_ALIGNMENT_INNER_TRACE") == "1":
            from transformers.integrations.sdpa_attention import sdpa_attention_forward
            from transformers.modeling_utils import AttentionInterface

            def traced_sdpa(module, query, key, value, attention_mask, **kwargs):
                if trainer.step == 1:
                    prefix = f"layer{module.layer_idx}.inner_attention"
                    for suffix, tensor in (
                        ("input", query),
                        ("input1", key),
                        ("input2", value),
                    ):
                        module_trace[f"{prefix}.{suffix}"] = (
                            tensor.transpose(1, 2).detach().cpu().clone()
                        )
                return sdpa_attention_forward(
                    module, query, key, value, attention_mask, **kwargs
                )

            assert model.model.config._attn_implementation == "sdpa"
            AttentionInterface.register("sdpa", traced_sdpa)
    if os.environ.get("LLMTUNER_ALIGNMENT_FLEX_ATTENTION") == "1":
        importlib.import_module("torchtitan_npu")
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        from transformers.modeling_utils import AttentionInterface

        assert model.model.config._attn_implementation == "sdpa"

        def baseline_flex(module, query, key, value, attention_mask, **kwargs):
            assert kwargs.get("dropout", 0.0) == 0.0
            batch, _, length, _ = query.shape
            assert length == 32 and key.shape[-2] == length
            expected = forwards[trainer.step - 1]["kwargs"]["attention_masks"][
                "dense_token_mask"
            ][..., :length, :length].to(query.device)
            if attention_mask is None:
                dense = torch.ones(
                    length, length, dtype=torch.bool, device=query.device
                ).tril()[None, None]
            else:
                dense = attention_mask[..., :length, :length]
                if dense.dtype != torch.bool:
                    dense = dense == 0
            torch.testing.assert_close(dense, expected, atol=0, rtol=0)

            def mask_mod(batch_index, head, query_index, key_index):
                return (
                    (query_index < length)
                    & (key_index < length)
                    & dense[
                        batch_index,
                        0,
                        query_index.clamp(max=length - 1),
                        key_index.clamp(max=length - 1),
                    ]
                )

            block_mask = create_block_mask(
                mask_mod, batch, None, length, length, device=query.device
            )
            output = flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                scale=kwargs.get("scaling", 128**-0.5),
                enable_gqa=True,
            )
            return output.transpose(1, 2).contiguous(), None

        AttentionInterface.register("sdpa", baseline_flex)
        print("Diagnostic: actual baseline FlexAttention under HF sdpa mask dispatch")
    logits = []
    step_metrics = []
    original_step = trainer.train_step
    trace_steps = {
        int(value)
        for value in os.environ.get("LLMTUNER_ALIGNMENT_TRACE_STEPS", "").split(",")
        if value
    }

    def snapshot_state(values):
        result = {}
        for name, value in values.items():
            assert value is not None, name
            if isinstance(value, DTensor):
                value = value.full_tensor()
            result[name] = value.detach().cpu().clone()
        return result

    def observe_step(self, data_iterator):
        result = original_step(data_iterator)
        if self.step == 1 and module_handles:
            torch.save(module_trace, args.output / "module-trace-step-1.pt")
            for module_handle in module_handles:
                module_handle.remove()
        assert result is not None
        step_metrics.append(result)
        if self.step in trace_steps:
            torch.save(
                {
                    "weights_after_update": snapshot_state(model.model.state_dict()),
                    "clipped_gradients": snapshot_state(
                        {
                            name: parameter.grad
                            for name, parameter in model.model.named_parameters()
                        }
                    ),
                },
                args.output / f"trace-step-{self.step}.pt",
            )
        return result

    trainer.train_step = MethodType(observe_step, trainer)

    def observe_forward(module, inputs, output):
        logits.append(output.detach().cpu().clone())

    handle = model.register_forward_hook(observe_forward)
    try:
        trainer.train()
        assert trainer.step == steps
        assert len(logits) == steps
        torch.save(logits, args.output / "candidate-logits.pt")
        (args.output / "candidate-step-metrics.json").write_text(
            json.dumps(step_metrics, indent=2)
        )
        torch.save(model.model.state_dict(), args.output / "candidate-final.pt")
        errors = []
        comparisons = []
        metric_functions = runpy.run_path(str(args.metrics))
        for step, (actual, forward, batch, measured, baseline_measured) in enumerate(
            zip(logits, forwards, batches, step_metrics, baseline_metrics, strict=True),
            start=1,
        ):
            expected = forward["logits"].reshape_as(actual)
            assert torch.isfinite(actual).all().item()
            errors.append(float((actual - expected).abs().max()))
            result = metric_functions["compare_tensors"](actual, expected)
            assert not metric_functions["failed_logits_metrics"](result), result
            expected_loss = torch.nn.functional.cross_entropy(
                expected.reshape(-1, 2048).float(),
                batch["kwargs"]["labels"].flatten(),
                reduction="sum",
            )
            if world > 1:
                expected_loss = expected_loss.to("npu")
                dist.all_reduce(expected_loss)
                expected_loss = expected_loss.cpu()
            expected_loss /= batch["kwargs"]["global_valid_tokens"].item()
            loss_metrics = metric_functions["compare_tensors"](
                torch.tensor(measured["loss"]), expected_loss
            )
            assert loss_metrics["mean_ae"] <= 0.02, loss_metrics
            assert loss_metrics["mean_re"] <= 0.005, loss_metrics
            assert loss_metrics["max_re"] <= 0.005, loss_metrics
            reference_values = baseline_measured["args"]
            reference_extra = baseline_measured["kwargs"]["extra_metrics"]
            assert measured["lr/AdamW"] == reference_extra["lr/AdamW"]
            assert measured["n_tokens_seen"] == reference_extra["n_tokens_seen"]
            recorded_loss = metric_functions["compare_tensors"](
                torch.tensor(measured["loss"]), torch.tensor(reference_values[1])
            )
            norm_metrics = metric_functions["compare_tensors"](
                torch.tensor(measured["grad_norm"]), torch.tensor(reference_values[3])
            )
            for metric_name, compared in (
                ("loss", recorded_loss),
                ("grad_norm", norm_metrics),
            ):
                context = {"step": step, "metric": metric_name, **compared}
                assert compared["mean_ae"] <= 0.02, context
                assert compared["mean_re"] <= 0.005, context
                assert compared["max_re"] <= 0.005, context
            comparisons.append(
                {
                    "logits": result,
                    "loss_from_logits": loss_metrics,
                    "loss": recorded_loss,
                    "grad_norm": norm_metrics,
                    "candidate": measured,
                    "baseline": baseline_measured,
                }
            )
        (args.output / "diagnostic.json").write_text(
            json.dumps(
                {"logits_max_ae_per_step": errors, "comparisons": comparisons}, indent=2
            )
        )
        baseline_final = torch.load(root / "final-hf.pt", weights_only=True)
        candidate_final = model.model.state_dict()
        weight_metrics = {}
        for name, expected in baseline_final.items():
            observed = candidate_final[name]
            if isinstance(observed, DTensor):
                observed = observed.full_tensor()
            weight_metrics[name] = metric_functions["compare_tensors"](
                observed.detach().cpu(), expected
            )
        (args.output / "weight-diagnostics.json").write_text(
            json.dumps(weight_metrics, indent=2)
        )
        print(f"Real candidate Trainer replay complete; logits max AE={errors}")
    finally:
        if reorder_norm:
            trainer_module.clip_grad_norm_ = original_clip
        handle.remove()
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
