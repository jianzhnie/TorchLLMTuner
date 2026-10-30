"""Replay real baseline batches through the complete candidate Trainer loop.

PYTHONPATH=. python tests/integration_tests/npu_trainer_alignment_replay.py
Requires artifacts from npu_baseline_training_capture.py. The default remains
FP32 debugmodel; Qwen3-30B requires explicit LLMTUNER_ALIGNMENT_MODEL opt-in.
For distributed capture, use torchrun with the same rank count and FSDP layout.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import runpy
from datetime import timedelta
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor

from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.seeding import set_deterministic_algorithms
from tests.integration_tests.alignment_history import (
    AlignmentHistory,
    capture_artifact_path,
    load_alignment_history,
)
from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    alignment_runtime_manifest,
    baseline_allow_hf32,
    baseline_learning_rate,
    build_candidate_30b_config,
    captured_attention,
    construction_dtype,
    debug_candidate_config,
    effective_forward_positions,
    input_fingerprints,
    is_qwen3_30b,
    iter_local_state,
    localize_parallel_reference,
    map_candidate_to_canonical_hf,
    map_capture_to_candidate_hf,
    mesh_manifest,
    optimizer_manifest,
    read_hf_architecture,
    selected_model,
    summed_cross_entropy,
    validate_baseline_30b_dict,
    validate_capture_hf32,
    validate_capture_runtime,
)


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
                "positions": effective_forward_positions(forward),
            }

    def state_dict(self):
        return {"cursor": self.cursor}

    def close(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-atol", type=float, default=1e-5)
    parser.add_argument("--weight-rtol", type=float, default=1e-4)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("../torchtitan-npu/scripts/numerical_alignment/utils/metrics.py"),
    )
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value < 0
        for value in (args.weight_atol, args.weight_rtol)
    ):
        parser.error("weight tolerances must be finite and non-negative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    set_deterministic_algorithms()
    if world > 1:
        from tests.integration_tests.harness import init_distributed

        init_distributed(
            timeout=timedelta(hours=2) if selected_model() == QWEN3_30B else None
        )
    root = args.capture
    if world > 1:
        root = root / f"rank{rank}"
        args.output = args.output / f"rank{rank}"
    baseline_config = json.loads((root / "config.json").read_text())
    captured_30b = is_qwen3_30b(baseline_config)
    alignment_model = selected_model()
    if captured_30b != (alignment_model == QWEN3_30B):
        raise ValueError(
            "Capture and replay must use the same explicit alignment model"
        )
    if captured_30b:
        validate_capture_runtime(json.loads((root / "manifest.json").read_text()))
        validate_baseline_30b_dict(baseline_config, world)
        architecture = read_hf_architecture(baseline_config["hf_assets_path"])
    else:
        architecture = None
        assert baseline_config["training"]["mixed_precision_param"] == "float32"
    allow_hf32 = baseline_allow_hf32(baseline_config)
    if captured_30b:
        validate_capture_hf32(
            json.loads((root / "manifest.json").read_text()), allow_hf32
        )
    for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
        backend.allow_hf32 = allow_hf32
    batches = torch.load(root / "consumed-batches.pt", weights_only=False)
    forwards = load_alignment_history(root, "forward-batches")
    baseline_metrics = torch.load(root / "step-metrics.pt", weights_only=False)
    for forward in forwards:
        if (
            captured_30b
            and captured_attention(baseline_config)["mask_type"] == "causal"
        ):
            assert not forward["kwargs"].get("attention_masks")
            continue
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
    config = (
        build_candidate_30b_config(baseline_config, args.output, world)
        if captured_30b
        else debug_candidate_config(baseline_config, args.output, steps, world)
    )
    if captured_30b:
        debug_only_settings = (
            "LLMTUNER_ALIGNMENT_NORM_BACKEND",
            "LLMTUNER_ALIGNMENT_QKV_PROJECTION",
            "LLMTUNER_ALIGNMENT_ROTARY_INIT_DEVICE",
            "LLMTUNER_ALIGNMENT_ATTENTION_BACKEND",
        )
        if any(name in os.environ for name in debug_only_settings):
            raise ValueError("Dense debugmodel settings are invalid for 30B replay")
        debug_only_switches = (
            "LLMTUNER_ALIGNMENT_EAGER_ATTENTION",
            "LLMTUNER_ALIGNMENT_NPU_ROPE_FREQ",
            "LLMTUNER_ALIGNMENT_TORCH_RMS_NORM",
            "LLMTUNER_ALIGNMENT_REMOVE_NORM_VIEWS",
            "LLMTUNER_ALIGNMENT_BASELINE_NORM_ORDER",
            "LLMTUNER_ALIGNMENT_MODULE_TRACE",
            "LLMTUNER_ALIGNMENT_INNER_TRACE",
            "LLMTUNER_ALIGNMENT_FLEX_ATTENTION",
        )
        if any(os.environ.get(name) == "1" for name in debug_only_switches):
            raise ValueError("Dense debugmodel diagnostics are invalid for 30B replay")
    previous_dtype = torch.get_default_dtype()
    dtype = construction_dtype(captured_30b)
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        trainer = Trainer(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model = trainer.model_parts[0]
    if captured_30b:
        model.model.config.router_aux_loss_coef = 0.0
        model.model.config.output_router_logits = False
    (args.output / "config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2)
    )
    (args.output / "optimizer.json").write_text(
        json.dumps(optimizer_manifest(trainer), indent=2)
    )
    candidate_hashes = input_fingerprints(batches, forwards)
    expected_hashes = json.loads((root / "input-hashes.json").read_text())
    if candidate_hashes != expected_hashes:
        raise ValueError("Replay input hashes differ from capture")
    (args.output / "input-hashes.json").write_text(
        json.dumps(candidate_hashes, indent=2)
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world,
                "alignment_model": alignment_model or "debugmodel",
                "runtime": alignment_runtime_manifest(
                    ("llmtuner", "llmtuner.trainer.trainer")
                ),
                "hf_assets_path": baseline_config["hf_assets_path"],
                "actual_mesh": mesh_manifest(trainer.parallel_dims),
                "aux_loss_disabled": (
                    config.parallel.ep_config.router_aux_loss_coef == 0.0
                    and model.model.config.router_aux_loss_coef == 0.0
                    and not model.model.config.output_router_logits
                    if captured_30b
                    else None
                ),
                "chunked_loss_num_chunks": config.training.chunked_loss_num_chunks,
                "trace_layers": os.environ.get("LLMTUNER_ALIGNMENT_TRACE_LAYERS"),
                "trace_boundaries_only": os.environ.get(
                    "LLMTUNER_ALIGNMENT_TRACE_BOUNDARIES_ONLY"
                )
                == "1",
                "gradient_trace_layers": os.environ.get(
                    "LLMTUNER_ALIGNMENT_GRADIENT_LAYERS", "0"
                ),
                "gradient_trace_enabled": os.environ.get(
                    "LLMTUNER_ALIGNMENT_GRADIENT_TRACE"
                )
                == "1",
                "weight_tolerances": {
                    "atol": args.weight_atol,
                    "rtol": args.weight_rtol,
                },
            },
            indent=2,
        )
    )
    if captured_30b:
        assert model.model.config.model_type == "qwen3_moe"
        assert model.model.config.vocab_size == architecture["vocab_size"]
        assert not model.model.config.tie_word_embeddings
        assert model.model.lm_head.weight is not model.model.model.embed_tokens.weight
        assert config.training.max_seq_len == 4096
        assert config.training.global_batch_size == 8
        assert config.training.activation_checkpoint_mode == "full"
        assert config.training.chunked_loss_num_chunks == 1
        assert config.parallel.ep_config.router_aux_loss_coef == 0.0
        assert model.model.config.router_aux_loss_coef == 0.0
        assert not model.model.config.output_router_logits
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
        assert trainer.parallel_dims.dp_shard == (
            config.parallel.data_parallel_shard_size if captured_30b else world
        )
    if captured_30b:
        assert trainer.parallel_dims.dp_replicate == 1
        assert trainer.parallel_dims.tp == config.parallel.tensor_parallel_size
        assert trainer.parallel_dims.ep == config.parallel.expert_parallel_size
        assert trainer.parallel_dims.pp == trainer.parallel_dims.cp == 1
        assert trainer.checkpointer.load(trainer.cfg.checkpoint.load_step)
        assert trainer.step == 0
    initial = torch.load(
        capture_artifact_path(args.capture, root, "initial-hf.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    captured_initial_count = len(initial)
    if captured_30b:
        assert model.model.lm_head.weight is not model.model.model.embed_tokens.weight
        assert "lm_head.weight" in initial
        initial = map_capture_to_candidate_hf(
            initial,
            set(model.model.state_dict()),
            architecture["num_experts"],
        )
        initial = localize_parallel_reference(initial, model, parameter_prefix="model.")
    else:
        assert model.model.lm_head.weight is model.model.model.embed_tokens.weight
        assert "lm_head.weight" not in initial
        initial["lm_head.weight"] = initial["model.embed_tokens.weight"]
    candidate_initial = model.model.state_dict()
    if set(candidate_initial) != set(initial):
        raise ValueError(
            "Candidate HF canonical key mismatch: "
            f"missing={sorted(set(initial) - set(candidate_initial))[:8]}, "
            f"unexpected={sorted(set(candidate_initial) - set(initial))[:8]}"
        )
    for name, parameter in candidate_initial.items():
        expected = initial[name]
        if parameter.shape != expected.shape or parameter.dtype != expected.dtype:
            raise ValueError(
                f"Candidate initial metadata mismatch for {name}: "
                f"{parameter.shape}/{parameter.dtype} != "
                f"{expected.shape}/{expected.dtype}"
            )
        if captured_30b:
            assert parameter.dtype == torch.bfloat16, name
    if world > 1 and captured_30b:
        for parameter in candidate_initial.values():
            assert isinstance(parameter, DTensor)
    elif world > 1:
        sharded_initial = {}
        for name, parameter in candidate_initial.items():
            assert isinstance(parameter, DTensor), name
            sharded_initial[name] = distribute_tensor(
                initial[name].to(parameter.device),
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
        model.model.load_state_dict(sharded_initial, strict=True)
    else:
        model.model.load_state_dict(initial, strict=True)
    for _name, parameter, expected in iter_local_state(
        model.model.state_dict(), initial
    ):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    (args.output / "initial-verification.json").write_text(
        json.dumps(
            {
                "namespace": "HF canonical, including native fused HF experts",
                "strict": True,
                "all_keys_shapes_dtypes_values_checked": True,
                "captured_tensor_count": captured_initial_count,
                "candidate_tensor_count": len(initial),
                "independent_lm_head": captured_30b,
            },
            indent=2,
        )
    )
    if captured_30b:
        del initial, candidate_initial
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
    gradient_trace = os.environ.get("LLMTUNER_ALIGNMENT_GRADIENT_TRACE") == "1"
    if gradient_trace:
        from tests.integration_tests.alignment_gradient_trace import (
            observe_gradient_clipping,
        )

        trainer_module.clip_grad_norm_ = observe_gradient_clipping(
            trainer_module.clip_grad_norm_, trainer, args.output
        )
    trainer.dataloader.close()
    trainer.dataloader = CapturedLoader(batches, forwards)
    module_trace = {}
    module_handles = []
    if selection := os.environ.get("LLMTUNER_ALIGNMENT_TRACE_LAYERS"):
        from tests.integration_tests.alignment_module_trace import (
            selected_trace_modules,
        )
        from tests.integration_tests.npu_baseline_training_capture import (
            observe_first_step_modules,
        )

        modules = selected_trace_modules(trainer, baseline=False, selection=selection)
        module_trace, module_handles = observe_first_step_modules(trainer, modules)
    elif os.environ.get("LLMTUNER_ALIGNMENT_MODULE_TRACE") == "1":
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
    logits = AlignmentHistory(args.output / "candidate-logits", writable=True)
    step_metrics = []
    observed_global_valid_tokens = []
    original_forward_backward = trainer.forward_backward_step

    def observe_forward_backward(self, *args, **kwargs):
        index = len(observed_global_valid_tokens)
        assert index < len(batches)
        actual = kwargs["global_valid_tokens"]
        expected = torch.as_tensor(
            batches[index]["kwargs"]["global_valid_tokens"], dtype=actual.dtype
        )
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        observed_global_valid_tokens.append(actual.detach().cpu().clone())
        return original_forward_backward(*args, **kwargs)

    trainer.forward_backward_step = MethodType(observe_forward_backward, trainer)
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
        return map_candidate_to_canonical_hf(result) if captured_30b else result

    def observe_step(self, data_iterator):
        result = original_step(data_iterator)
        if self.step == 1 and module_handles:
            torch.save(module_trace, args.output / "module-trace-step-1.pt")
            for module_handle in module_handles:
                module_handle.remove()
        assert result is not None
        step_metrics.append(result)
        if self.step in trace_steps:
            gradients = snapshot_state(
                {
                    name: parameter.grad
                    for name, parameter in model.model.named_parameters()
                    if not captured_30b or name.endswith("norm.weight")
                }
            )
            trace = {"clipped_gradients": gradients}
            if not captured_30b:
                trace["weights_after_update"] = snapshot_state(model.model.state_dict())
            torch.save(trace, args.output / f"trace-step-{self.step}.pt")
        return result

    trainer.train_step = MethodType(observe_step, trainer)

    def observe_forward(module, inputs, output):
        logits.append(output.detach().cpu().clone())

    handle = model.register_forward_hook(observe_forward)
    try:
        if captured_30b:
            torch.npu.empty_cache()
        trainer.train()
        assert trainer.step == steps
        assert len(logits) == steps
        assert len(observed_global_valid_tokens) == steps
        torch.save(
            observed_global_valid_tokens,
            args.output / "global-valid-tokens.pt",
        )
        (args.output / "candidate-step-metrics.json").write_text(
            json.dumps(step_metrics, indent=2)
        )
        errors = []
        comparisons = []
        metric_functions = runpy.run_path(str(args.metrics))
        for actual, forward, batch, measured, baseline_measured in zip(
            logits, forwards, batches, step_metrics, baseline_metrics, strict=True
        ):
            expected = forward["logits"]
            if captured_30b and trainer.parallel_dims.tp > 1:
                tp_mesh = trainer.parallel_dims.get_mesh("tp")
                expected = expected.flatten(0, -2).chunk(
                    trainer.parallel_dims.tp, dim=0
                )[tp_mesh.get_local_rank()]
            expected = expected.reshape_as(actual)
            if captured_30b:
                assert not any("aux" in name for name in measured)
            assert torch.isfinite(actual).all().item()
            errors.append(float((actual - expected).abs().max()))
            result = metric_functions["compare_tensors"](actual, expected)
            loss_metrics = None
            if world == 1:
                expected_loss = summed_cross_entropy(
                    expected, batch["kwargs"]["labels"]
                )
                expected_loss /= torch.as_tensor(
                    batch["kwargs"]["global_valid_tokens"]
                ).item()
                loss_metrics = metric_functions["compare_tensors"](
                    torch.tensor(measured["loss"]), expected_loss
                )
            reference_values = baseline_measured["args"]
            reference_extra = baseline_measured["kwargs"]["extra_metrics"]
            assert measured["lr/AdamW"] == baseline_learning_rate(reference_extra)
            assert measured["n_tokens_seen"] == reference_extra["n_tokens_seen"]
            recorded_loss = metric_functions["compare_tensors"](
                torch.tensor(measured["loss"]), torch.tensor(reference_values[1])
            )
            norm_metrics = metric_functions["compare_tensors"](
                torch.tensor(measured["grad_norm"]), torch.tensor(reference_values[3])
            )
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
        weight_failures = {}

        def compare_weight(name, value, expected):
            try:
                if (
                    not torch.isfinite(value).all()
                    or not torch.isfinite(expected).all()
                ):
                    raise AssertionError("Final weights contain non-finite values")
                torch.testing.assert_close(
                    value, expected, atol=args.weight_atol, rtol=args.weight_rtol
                )
            except AssertionError as error:
                weight_failures[name] = str(error)
            return (
                metric_functions["compare_tensors"](value, expected)
                if value.numel()
                else {"empty_shard": True}
            )

        if captured_30b:
            baseline_final = map_capture_to_candidate_hf(
                torch.load(
                    capture_artifact_path(args.capture, root, "final-hf.pt"),
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                ),
                set(model.model.state_dict()),
                architecture["num_experts"],
            )
            baseline_final = localize_parallel_reference(
                baseline_final, model, parameter_prefix="model."
            )
            candidate_local = {}
            weight_metrics = {}
            for name, value, expected in iter_local_state(
                model.model.state_dict(), baseline_final
            ):
                candidate_local[name] = value
                weight_metrics[name] = compare_weight(name, value, expected)
            torch.save(candidate_local, args.output / "candidate-final-local.pt")
            (args.output / "weight-diagnostics.json").write_text(
                json.dumps(
                    {
                        "namespace": "candidate fused HF experts, local DTensor shards",
                        "metrics": weight_metrics,
                    },
                    indent=2,
                )
            )
        else:
            candidate_final = snapshot_state(model.model.state_dict())
            torch.save(candidate_final, args.output / "candidate-final.pt")
            baseline_final = torch.load(
                capture_artifact_path(args.capture, root, "final-hf.pt"),
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            if set(candidate_final) != set(baseline_final):
                raise ValueError("Final HF canonical key sets differ")
            weight_metrics = {}
            for name, expected in baseline_final.items():
                observed = candidate_final[name]
                weight_metrics[name] = compare_weight(name, observed, expected)
            (args.output / "weight-diagnostics.json").write_text(
                json.dumps(weight_metrics, indent=2)
            )
        (args.output / "weight-acceptance.json").write_text(
            json.dumps(
                {
                    "passed": not weight_failures,
                    "atol": args.weight_atol,
                    "rtol": args.weight_rtol,
                    "failures": weight_failures,
                },
                indent=2,
            )
        )
        if dist.is_initialized():
            # Preserve every rank's diagnostics before an acceptance failure
            # makes torchrun terminate the remaining workers.
            dist.barrier()
        for step, comparison in enumerate(comparisons, start=1):
            result = comparison["logits"]
            assert not metric_functions["failed_logits_metrics"](result), result
            for metric_name in ("loss_from_logits", "loss", "grad_norm"):
                compared = comparison[metric_name]
                if compared is None:
                    continue
                context = {"step": step, "metric": metric_name, **compared}
                assert compared["mean_ae"] <= 0.02, context
                assert compared["mean_re"] <= 0.005, context
                assert compared["max_re"] <= 0.005, context
        assert not weight_failures, (
            f"Final weights exceed predeclared tolerances: {list(weight_failures)[:8]}"
        )
        print(f"Real candidate Trainer replay complete; logits max AE={errors}")
    finally:
        if reorder_norm or gradient_trace:
            trainer_module.clip_grad_norm_ = original_clip
        handle.remove()
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
