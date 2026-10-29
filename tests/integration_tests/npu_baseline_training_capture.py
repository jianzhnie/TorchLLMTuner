"""Capture real baseline Trainer weights and forward batches for alignment.

Run with torchrun and baseline ConfigManager arguments.
PYTHONPATH must expose this repository, the baseline and its real dependencies.
This observer does not replace the Trainer loop, loss, optimizer, or dataloader.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils._python_dispatch import TorchDispatchMode


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


def observe_first_step_modules(trainer, modules):
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

            value.register_hook(save_gradient)

    from functools import partial

    for name, module in modules.items():
        handles.append(module.register_forward_hook(partial(observe, name)))
    return captured, handles


def main() -> None:
    importlib.import_module("torchtitan_npu")
    from torchtitan.config import ConfigManager
    from torchtitan.tools.logging import init_logger

    init_logger()
    config = ConfigManager().parse_args()
    root = Path(config.dump_folder) / "alignment-capture"
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    trainer = config.build()
    dist.barrier()
    if dist.get_world_size() > 1:
        root = root / f"rank{rank}"
        root.mkdir(exist_ok=False)
    assert len(trainer.model_parts) == 1
    model = trainer.model_parts[0]
    spec = config.model_spec
    adapter = spec.state_dict_adapter(spec.model, config.hf_assets_path)
    initial_path = os.environ.get("LLMTUNER_ALIGNMENT_INITIAL_HF")
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
    torch.save(adapter.to_hf(_snapshot(model.state_dict())), root / "initial-hf.pt")
    (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    captured = []
    consumed = []
    measured = []
    module_trace = {}
    module_handles = []
    if os.environ.get("LLMTUNER_ALIGNMENT_MODULE_TRACE") == "1":
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

    def observe_train_step(self, *args, **kwargs):
        if (
            self.step == 1
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
                {name: parameter.grad for name, parameter in model.named_parameters()}
            )
            for name, module in model.named_modules():
                split = getattr(module, "_split_qkv_on_save", None)
                if split is not None:
                    split(module, gradients, f"{name}.", {})
            gradients = adapter.to_hf(gradients)
            assert all(value is not None for value in gradients.values())
            torch.save(
                {
                    "weights_after_update": adapter.to_hf(
                        _snapshot(model.state_dict())
                    ),
                    "clipped_gradients": gradients,
                },
                root / f"trace-step-{self.step}.pt",
            )
        return result

    trainer.train_step = MethodType(observe_train_step, trainer)

    def before_forward(module, args, kwargs):
        captured.append({"args": _snapshot(args), "kwargs": _snapshot(kwargs)})

    def after_forward(module, args, output):
        values = _snapshot(output)
        assert isinstance(values, torch.Tensor)
        assert torch.isfinite(values).all().item()
        captured[-1]["logits"] = values

    handles = [
        model.register_forward_pre_hook(before_forward, with_kwargs=True),
        model.register_forward_hook(after_forward),
    ]
    try:
        trainer.train()
        assert trainer.step == config.training.steps
        assert len(captured) == config.training.steps
        assert len(consumed) == config.training.steps
        torch.save(captured, root / "forward-batches.pt")
        torch.save(consumed, root / "consumed-batches.pt")
        torch.save(measured, root / "step-metrics.pt")
        torch.save(adapter.to_hf(_snapshot(model.state_dict())), root / "final-hf.pt")
        print(f"PASS real baseline Trainer capture: {len(captured)} steps at {root}")
    finally:
        for handle in handles:
            handle.remove()
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
