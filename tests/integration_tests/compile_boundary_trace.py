"""Opt-in local snapshots for TP/EP compile diagnostics.

Snapshots alter graph outputs and are diagnostic evidence, not performance data.
Call the report on rank five inside the caller's coordinated error boundary.
"""

import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from llmtuner.utils.module_names import canonical_fqn


def dump_compile_training_state(folder, rank, step, model, reference, logits):
    """Export local state outside the compiled graph, without extra collectives."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)

    def local_cpu(tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return tensor.detach().cpu().clone()

    state = {"logits": local_cpu(logits)}
    for label, target in (("candidate", model), ("reference", reference)):
        state[label] = {
            canonical_fqn(name): {
                "weight": local_cpu(parameter),
                "gradient": (
                    local_cpu(parameter.grad) if parameter.grad is not None else None
                ),
            }
            for name, parameter in target.named_parameters()
        }
    torch.save(state, destination / f"rank{rank}-step{step + 1}.pt")


def replay_compile_training_weights(folder, rank, model, *, label="candidate"):
    """Diagnostic intervention for this fixture's stateless SGD, not resume."""
    state = torch.load(
        Path(folder) / f"rank{rank}-step1.pt", map_location="cpu", weights_only=True
    )[label]
    parameters = {
        canonical_fqn(name): parameter for name, parameter in model.named_parameters()
    }
    assert parameters.keys() == state.keys()
    targets = []
    for name, parameter in parameters.items():
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        weight = state[name]["weight"]
        assert local.shape == weight.shape and local.dtype == weight.dtype, name
        targets.append((local, weight))
    with torch.no_grad():
        for local, weight in targets:
            local.copy_(weight)
            torch.testing.assert_close(local.cpu(), weight, rtol=0, atol=0)


def install_compile_boundary_trace(reference, model):
    snapshots = {"reference": {}, "candidate": {}}
    for label, target_model in (("reference", reference), ("candidate", model)):
        layer = target_model.layers[0]
        modules = {
            "input_norm": layer.input_layernorm,
            "attention": layer.self_attn,
            "post_norm": layer.post_attention_layernorm,
            "moe": layer.mlp,
            "gate": (layer.mlp.router.gate if label == "candidate" else layer.mlp.gate),
        }
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
            modules[name] = getattr(layer.self_attn, name)
        if label == "candidate":
            modules["router"] = layer.mlp.router
        for name, module in modules.items():

            def capture(module, inputs, kwargs, output, label=label, name=name):
                values = output if isinstance(output, tuple) else (output,)
                value = inputs[0] if inputs else kwargs["hidden_states"]
                snapshots[label][name + "_input"] = (value.detach().clone(),)
                if hasattr(module, "weight"):
                    snapshots[label][name + "_weight"] = (
                        module.weight.detach().clone(),
                    )
                snapshots[label][name] = tuple(
                    value.detach().clone()
                    for value in values
                    if isinstance(value, torch.Tensor)
                )

            module.register_forward_hook(capture, with_kwargs=True)

    return snapshots


def report_compile_boundary_trace(snapshots, step, token_slice):
    folder = Path(
        os.environ.get(
            "LLMTUNER_TEST_TRACE_DIR",
            "/tmp/torchllmtuner-compile-boundaries",
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            label: {
                name: tuple(value.cpu() for value in values)
                for name, values in captures.items()
            }
            for label, captures in snapshots.items()
        },
        folder / f"rank5-step{step + 1}.pt",
    )
    for name, values in snapshots["candidate"].items():
        if name == "router" or name.endswith(("_input", "_weight")):
            continue
        actual_trace = values[0].reshape(-1, values[0].shape[-1])
        original_trace = snapshots["reference"][name][0]
        # This fixture has eight Q/KV heads and TP4. Rank five is TP rank
        # one: projection outputs/head norms hold its physical head slice.
        if name in {"q_norm", "k_norm"}:
            original_trace = original_trace.reshape(-1, 8, 8)[:, 2:4]
        elif name in {"q_proj", "k_proj", "v_proj"}:
            original_trace = original_trace.chunk(4, dim=-1)[1]
        expected_trace = original_trace.reshape(-1, original_trace.shape[-1])
        if actual_trace.shape[0] != expected_trace.shape[0]:
            expected_trace = expected_trace[token_slice]
        difference = (actual_trace.float() - expected_trace.float()).abs()
        print(
            f"TRACE step={step + 1} {name} "
            f"shape={tuple(actual_trace.shape)} "
            f"max_abs={difference.max().item()}",
            flush=True,
        )
    router_ids = snapshots["candidate"]["router"][1].reshape(-1, 2)
    expected_ids = snapshots["reference"]["gate"][0].float().topk(2, dim=-1).indices
    expected_ids = expected_ids.reshape(-1, 2)
    if router_ids.shape[0] != expected_ids.shape[0]:
        expected_ids = expected_ids[token_slice]
    mismatches = (router_ids.sort(-1).values != expected_ids.sort(-1).values).any(-1)
    print(
        f"TRACE step={step + 1} router_id_mismatch={mismatches.sum().item()}",
        flush=True,
    )
