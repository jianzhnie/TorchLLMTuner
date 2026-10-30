"""Capture real baseline Trainer weights and forward batches for alignment.

Run with torchrun and baseline ConfigManager arguments.
PYTHONPATH must expose this repository, the baseline and its real dependencies.
This observer does not replace the Trainer loop, loss, optimizer, or dataloader.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils._python_dispatch import TorchDispatchMode

from tests.integration_tests.alignment_history import (
    AlignmentHistory,
    save_shared_snapshot,
)
from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    alignment_runtime_manifest,
    baseline_allow_hf32,
    baseline_cli_args,
    baseline_config_snapshot,
    call_after_load,
    configure_baseline_30b,
    input_fingerprints,
    is_qwen3_30b,
    mesh_manifest,
    optimizer_manifest,
    selected_model,
)


def _snapshot(value):
    from torch.nn.attention.flex_attention import BlockMask

    if isinstance(value, BlockMask):
        batch, heads, queries, keys = value.shape
        device = value.kv_indices.device
        dense_mask = value.mask_mod(
            torch.arange(batch, device=device)[:, None, None, None],
            torch.arange(heads, device=device)[None, :, None, None],
            torch.arange(queries, device=device)[None, None, :, None],
            torch.arange(keys, device=device)[None, None, None, :],
        )
        return {"kind": "BlockMask", "dense_token_mask": _snapshot(dense_mask)}
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: _snapshot(item) for name, item in value.items()}
    if isinstance(value, tuple | list):
        return type(value)(_snapshot(item) for item in value)
    return value


def _local_tensor_layout(value):
    local = value.to_local() if isinstance(value, DTensor) else value
    layout = {
        "shape": tuple(local.shape),
        "stride": local.stride(),
        "contiguous": local.is_contiguous(),
        "storage_offset": local.storage_offset(),
    }
    if local.device.type == "npu":
        import torch_npu

        layout["npu_format"] = int(torch_npu.get_npu_format(local))
    return layout


class ProjectionDispatchCapture(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        name = str(function)
        selected = any(token in name for token in ("mm", "matmul", "linear"))
        selected = selected and any(
            isinstance(value, torch.Tensor) and 4096 in value.shape for value in args
        )
        record = None
        if selected:
            record = {
                "operator": name,
                "args": [],
                "kwargs": _snapshot(kwargs),
                "deterministic": torch.are_deterministic_algorithms_enabled(),
            }
            for value in args:
                if isinstance(value, torch.Tensor):
                    local = value.to_local() if isinstance(value, DTensor) else value
                    record["args"].append(
                        {
                            "value": _snapshot(local),
                            "layout": _local_tensor_layout(local),
                        }
                    )
                else:
                    record["args"].append(value)
        output = function(*args, **kwargs)
        if record is not None:
            record["output"] = _snapshot(output)
            self.records.append(record)
        return output


def observe_first_step_modules(trainer, modules, *, partial_path=None):
    """Capture local activations and output gradients without changing autograd."""
    captured = {}
    handles = []

    def observe(name, module, inputs, output):
        if trainer.step != 1:
            return
        value = output[0] if isinstance(output, tuple) else output
        assert isinstance(value, torch.Tensor), name
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured[f"{name}.input"] = _snapshot(inputs[0])
        for index, item in enumerate(inputs[1:], 1):
            if isinstance(item, torch.Tensor):
                captured[f"{name}.input{index}"] = _snapshot(item)
        for index, item in enumerate(inputs):
            if isinstance(item, torch.Tensor) and item.requires_grad:
                suffix = "input" if index == 0 else f"input{index}"

                def save_input_gradient(gradient, key=f"{name}.{suffix}_gradient"):
                    captured[f"{key}_layout"] = _local_tensor_layout(gradient)
                    captured[key] = _snapshot(gradient)
                    if partial_path is not None and name == "final_norm":
                        torch.save(captured, partial_path)

                item.register_hook(save_input_gradient)
        captured[f"{name}.output"] = _snapshot(value)
        if ".projection." in name:
            pending = [(value.grad_fn, 0)]
            visited = set()
            graph = []
            while pending:
                node, depth = pending.pop()
                if node is None or node in visited or depth > 6:
                    continue
                visited.add(node)
                entry = {"node": type(node).__name__, "depth": depth, "saved": {}}
                for attribute in (
                    "_saved_self",
                    "_saved_other",
                    "_saved_mat2",
                    "_saved_weight",
                ):
                    tensor = getattr(node, attribute, None)
                    if isinstance(tensor, torch.Tensor):
                        entry["saved"][attribute] = _local_tensor_layout(tensor)
                graph.append(entry)
                pending.extend((parent, depth + 1) for parent, _ in node.next_functions)
            captured[f"{name}.autograd_graph"] = graph
        if isinstance(output, tuple):
            for index, item in enumerate(output[1:], 1):
                if isinstance(item, torch.Tensor):
                    captured[f"{name}.output{index}"] = _snapshot(item)
        if name.endswith(".rope"):
            captured[f"{name}.cache"] = _snapshot(module.cache)
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            captured[f"{name}.weight_layout"] = _local_tensor_layout(weight)
        if value.requires_grad:

            def save_gradient(gradient):
                captured[f"{name}.gradient_layout"] = _local_tensor_layout(gradient)
                captured[f"{name}.gradient"] = _snapshot(gradient)
                if partial_path is not None and name == "final_norm":
                    torch.save(captured, partial_path)

            value.register_hook(save_gradient)

    from functools import partial

    for name, module in modules.items():
        handles.append(module.register_forward_hook(partial(observe, name)))
    return captured, handles


def validate_hf_snapshot(snapshot, hf_assets_path):
    index_path = Path(hf_assets_path) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if set(snapshot) != set(weight_map):
        raise ValueError(
            "Initial HF canonical key mismatch: "
            f"missing={sorted(set(weight_map) - set(snapshot))[:8]}, "
            f"unexpected={sorted(set(snapshot) - set(weight_map))[:8]}"
        )
    from safetensors import safe_open

    for filename in sorted(set(weight_map.values())):
        names = [name for name, source in weight_map.items() if source == filename]
        with safe_open(
            str(Path(hf_assets_path) / filename), framework="pt", device="cpu"
        ) as tensors:
            for name in names:
                expected = tensors.get_tensor(name)
                actual = snapshot[name]
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise ValueError(
                        f"Initial HF metadata mismatch for {name}: "
                        f"{actual.shape}/{actual.dtype} != "
                        f"{expected.shape}/{expected.dtype}"
                    )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def main() -> None:
    importlib.import_module("torchtitan_npu")
    from torchtitan.config import ConfigManager
    from torchtitan.tools.logging import init_logger

    init_logger()
    alignment_model = selected_model()
    config = ConfigManager().parse_args(baseline_cli_args(sys.argv[1:]))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    disabled_balancing = (
        configure_baseline_30b(config, world) if alignment_model == QWEN3_30B else []
    )
    root = Path(config.dump_folder) / "alignment-capture"
    shared_root = root
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    if alignment_model == QWEN3_30B:
        allow_hf32 = baseline_allow_hf32(baseline_config_snapshot(config))
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = allow_hf32
    trainer = config.build()
    assert dist.get_world_size() == world
    dist.barrier()
    if world > 1:
        root = root / f"rank{rank}"
        root.mkdir(exist_ok=False)
    assert len(trainer.model_parts) == 1
    model = trainer.model_parts[0]
    spec = config.model_spec
    adapter = spec.state_dict_adapter(spec.model, config.hf_assets_path)
    initial_path = os.environ.get("LLMTUNER_ALIGNMENT_INITIAL_HF")
    if alignment_model == QWEN3_30B and initial_path:
        raise ValueError(
            "30B initial weights must come from the configured real HF load"
        )
    if initial_path:
        initial = torch.load(initial_path, weights_only=True)
        mapped = adapter.from_hf(dict(initial))
        loaded = {}
        for name, parameter in model.state_dict().items():
            source = mapped[name].to(parameter.device)
            loaded[name] = (
                distribute_tensor(
                    source,
                    device_mesh=parameter.device_mesh,
                    placements=parameter.placements,
                )
                if isinstance(parameter, DTensor)
                else source
            )
        model.load_state_dict(loaded, strict=True)
        actual = adapter.to_hf(_snapshot(model.state_dict()))
        for name, value in initial.items():
            torch.testing.assert_close(actual[name], value, atol=0, rtol=0)
    config_dict = baseline_config_snapshot(config)
    (root / "config.json").write_text(json.dumps(config_dict, indent=2))
    manifest = {
        "runtime": alignment_runtime_manifest(
            ("torchtitan", "torchtitan.trainer", "torchtitan_npu")
        ),
        "rank": rank,
        "world_size": world,
        "capture_artifact_layout": "shared-hf-v1",
        "shared_artifact_writer_rank": 0,
        "alignment_model": alignment_model or "debugmodel",
        "hf_assets_path": config.hf_assets_path,
        "actual_mesh": mesh_manifest(trainer.parallel_dims),
        "disabled_balancing": disabled_balancing,
        "module_trace_enabled": bool(os.environ.get("LLMTUNER_ALIGNMENT_TRACE_LAYERS"))
        or (
            alignment_model != QWEN3_30B
            and os.environ.get("LLMTUNER_ALIGNMENT_MODULE_TRACE") == "1"
        ),
        "trace_layers": os.environ.get("LLMTUNER_ALIGNMENT_TRACE_LAYERS"),
        "trace_boundaries_only": os.environ.get(
            "LLMTUNER_ALIGNMENT_TRACE_BOUNDARIES_ONLY"
        )
        == "1",
        "gradient_trace_layers": os.environ.get(
            "LLMTUNER_ALIGNMENT_GRADIENT_LAYERS", "0"
        ),
        "gradient_trace_enabled": os.environ.get("LLMTUNER_ALIGNMENT_GRADIENT_TRACE")
        == "1",
        "dispatch_trace_enabled": alignment_model != QWEN3_30B
        and os.environ.get("LLMTUNER_ALIGNMENT_DISPATCH_TRACE") == "1",
    }
    if alignment_model == QWEN3_30B:
        assert is_qwen3_30b(config_dict)
        assert trainer.parallel_dims.dp_replicate == 1
        assert (
            trainer.parallel_dims.dp_shard
            == config.parallelism.data_parallel_shard_degree
        )
        assert trainer.parallel_dims.tp == config.parallelism.tensor_parallel_degree
        assert trainer.parallel_dims.ep == config.parallelism.expert_parallel_degree
        assert trainer.parallel_dims.pp == trainer.parallel_dims.cp == 1
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "optimizer.json").write_text(
        json.dumps(optimizer_manifest(trainer), indent=2)
    )
    initial_captured = []

    def capture_initial():
        initial_hf = adapter.to_hf(_snapshot(model.state_dict()))
        save_shared_snapshot(
            initial_hf,
            shared_root / "initial-hf.pt",
            validate=(lambda state: validate_hf_snapshot(state, config.hf_assets_path))
            if alignment_model == QWEN3_30B
            else None,
        )
        initial_captured.append(True)
        (root / "initial-load.json").write_text(
            json.dumps(
                {
                    "snapshot_point": (
                        "after checkpointer.load, before first train_step"
                    ),
                    "loaded_from_hf": alignment_model == QWEN3_30B,
                    "trainer_step": trainer.step,
                    "canonical_tensor_count": len(initial_hf),
                    "all_values_checked": alignment_model == QWEN3_30B,
                },
                indent=2,
            )
        )

    original_load = trainer.checkpointer.load

    def observe_load(self, *args, **kwargs):
        if alignment_model == QWEN3_30B:
            assert not initial_captured
        result = call_after_load(original_load, capture_initial, *args, **kwargs)
        if alignment_model == QWEN3_30B:
            assert result and trainer.step == 0
            # Full-tensor observation must not retain its free gather buffers
            # in the allocator before the near-capacity training workload.
            torch.npu.empty_cache()
        return result

    trainer.checkpointer.load = MethodType(observe_load, trainer.checkpointer)
    captured = AlignmentHistory(root / "forward-batches", writable=True)
    consumed = []
    measured = []
    module_trace = {}
    module_handles = []
    if selection := os.environ.get("LLMTUNER_ALIGNMENT_TRACE_LAYERS"):
        from tests.integration_tests.alignment_module_trace import (
            selected_trace_modules,
        )

        modules = selected_trace_modules(trainer, baseline=True, selection=selection)
        module_trace, module_handles = observe_first_step_modules(
            trainer, modules, partial_path=root / "module-trace-step-1.partial.pt"
        )
    elif (
        alignment_model != QWEN3_30B
        and os.environ.get("LLMTUNER_ALIGNMENT_MODULE_TRACE") == "1"
    ):
        modules = {"embedding": model.tok_embeddings, "norm": model.norm}
        for index, layer in model.layers.items():
            for name, attribute in (
                ("input_norm", "attention_norm"),
                ("attention", "attention"),
                ("post_norm", "ffn_norm"),
                ("mlp", "feed_forward"),
            ):
                modules[f"layer{index}.{name}"] = getattr(layer, attribute)
            modules[f"layer{index}"] = layer
            for name, attribute in (
                ("q_norm", "q_norm"),
                ("k_norm", "k_norm"),
                ("output_projection", "wo"),
                ("inner_attention", "inner_attention"),
                ("rope", "rope"),
            ):
                modules[f"layer{index}.{name}"] = getattr(layer.attention, attribute)
            projection = layer.attention.qkv_linear
            for attribute in ("wqkv", "wq", "wk", "wv"):
                if hasattr(projection, attribute):
                    modules[f"layer{index}.projection.{attribute}"] = getattr(
                        projection, attribute
                    )
        module_trace, module_handles = observe_first_step_modules(trainer, modules)
    original_log = trainer.metrics_processor.log

    def observe_log(self, *args, **kwargs):
        measured.append({"args": _snapshot(args), "kwargs": _snapshot(kwargs)})
        return original_log(*args, **kwargs)

    trainer.metrics_processor.log = MethodType(observe_log, trainer.metrics_processor)
    original_step = trainer.forward_backward_step

    def observe_step(self, *args, **kwargs):
        consumed.append({"args": _snapshot(args), "kwargs": _snapshot(kwargs)})
        return original_step(*args, **kwargs)

    trainer.forward_backward_step = MethodType(observe_step, trainer)
    trace_steps = {
        int(value)
        for value in os.environ.get("LLMTUNER_ALIGNMENT_TRACE_STEPS", "").split(",")
        if value
    }
    original_train_step = trainer.train_step
    gradient_clip_owner = None
    if os.environ.get("LLMTUNER_ALIGNMENT_GRADIENT_TRACE") == "1":
        from torchtitan.trainer import dist_utils

        from tests.integration_tests.alignment_gradient_trace import (
            observe_gradient_clipping,
        )

        gradient_clip_owner = dist_utils
        original_gradient_clip = dist_utils.clip_grad_norm_
        dist_utils.clip_grad_norm_ = observe_gradient_clipping(
            original_gradient_clip, trainer, root
        )

    def observe_train_step(self, *args, **kwargs):
        if (
            self.step == 1
            and alignment_model != QWEN3_30B
            and os.environ.get("LLMTUNER_ALIGNMENT_DISPATCH_TRACE") == "1"
        ):
            with ProjectionDispatchCapture() as dispatch_capture:
                result = original_train_step(*args, **kwargs)
            torch.save(dispatch_capture.records, root / "projection-dispatch-step-1.pt")
        else:
            result = original_train_step(*args, **kwargs)
        if self.step == 1 and module_handles:
            torch.save(module_trace, root / "module-trace-step-1.pt")
            for handle in module_handles:
                handle.remove()
        if self.step in trace_steps:
            gradients = _snapshot(
                {
                    name: parameter.grad
                    for name, parameter in model.named_parameters()
                    if alignment_model != QWEN3_30B or name.endswith("norm.weight")
                }
            )
            if alignment_model != QWEN3_30B:
                for name, module in model.named_modules():
                    split = getattr(module, "_split_qkv_on_save", None)
                    if split is not None:
                        split(module, gradients, f"{name}.", {})
            gradients = adapter.to_hf(gradients)
            assert all(value is not None for value in gradients.values())
            trace = {"clipped_gradients": gradients}
            if alignment_model != QWEN3_30B:
                trace["weights_after_update"] = adapter.to_hf(
                    _snapshot(model.state_dict())
                )
            torch.save(trace, root / f"trace-step-{self.step}.pt")
        return result

    trainer.train_step = MethodType(observe_train_step, trainer)

    def before_forward(module, args, kwargs):
        captured.append({"args": _snapshot(args), "kwargs": _snapshot(kwargs)})

    def after_forward(module, args, output):
        values = _snapshot(output)
        assert isinstance(values, torch.Tensor)
        assert torch.isfinite(values).all().item()
        observation = captured[-1]
        observation["logits"] = values
        captured[-1] = observation

    handles = [
        model.register_forward_pre_hook(before_forward, with_kwargs=True),
        model.register_forward_hook(after_forward),
    ]
    try:
        trainer.train()
        assert initial_captured == [True]
        assert trainer.step == config.training.steps
        assert len(captured) == config.training.steps
        assert len(consumed) == config.training.steps
        final_hf = adapter.to_hf(_snapshot(model.state_dict()))
        save_shared_snapshot(final_hf, shared_root / "final-hf.pt")
        torch.save(consumed, root / "consumed-batches.pt")
        torch.save(measured, root / "step-metrics.pt")
        (root / "input-hashes.json").write_text(
            json.dumps(input_fingerprints(consumed, captured), indent=2)
        )
        print(f"PASS real baseline Trainer capture: {len(captured)} steps at {root}")
    finally:
        if gradient_clip_owner is not None:
            gradient_clip_owner.clip_grad_norm_ = original_gradient_clip
        for handle in handles:
            handle.remove()
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
