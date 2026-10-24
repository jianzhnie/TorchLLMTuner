"""Four CPU/Gloo ranks: real Trainer steps against independent HF CE.

Run with torchrun --nproc_per_node=4; optional native/custom (default custom).
Append aux to validate coefficient 0.01 against an independent routing formula.
Append adam after aux to use the production AdamW optimizer container.
Use noaux in place of aux and append none after adam for diagnostic controls.
Append fp32_router after the AC mode to diagnose the HF gate dtype contract.
This explicitly changes the oracle; the default HF reference remains unchanged.
Append probe to that option for two-layer/two-microbatch diagnostics.
Probe requires adam and fp32_router; its exact checks target the initial step.
Use trajectory instead of probe to collect all three steps before failing;
all original comparisons and tolerances remain required for a successful exit.
Trajectory also records per-step activation/gradient differences. Tensor hooks
can affect backward scheduling, so its failure counts are diagnostic evidence,
not interchangeable with a run without those hooks.
Use forward_trajectory to record only forward values without tensor grad hooks.
Set LLMTUNER_TEST_FSDP_REDUCE_DTYPE=float64 for explicit FP64 reduction.
The default remains float32 and all numerical assertions remain unchanged.
Uses the explicit CPU attention substitute, not target-device flex acceptance.
Model assembly, preprocessing, token accounting, accumulation, replicated gradient
reduction, clipping, optimizer ordering and metrics use production Trainer code.
"""

import os
import sys
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.models.common.aux_loss import (
    AuxLoss,
    in_checkpoint_recompute,
    register_aux_loss_metrics_hook,
)
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention
from tests.integration_tests.tp_ep_cp_equivalence import _build, _cfg, _data, _gather
from tests.integration_tests.tp_ep_cp_training_equivalence import full, oracle_view


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        cfg = _cfg()
        implementation = sys.argv[1] if len(sys.argv) > 1 else "custom"
        assert implementation in {"native", "custom"}
        cfg.parallel.fsdp_config.fsdp_implementation = implementation
        cfg.parallel.fsdp_config.fsdp_reduce_dtype = os.environ.get(
            "LLMTUNER_TEST_FSDP_REDUCE_DTYPE", "float32"
        )
        cfg.parallel.fsdp_config.__post_init__()
        aux = len(sys.argv) > 2 and sys.argv[2] == "aux"
        adam = len(sys.argv) > 3 and sys.argv[3] == "adam"
        ac_mode = sys.argv[4] if len(sys.argv) > 4 else "full"
        assert ac_mode in {"full", "none"}
        if len(sys.argv) > 7:
            raise ValueError("Unexpected arguments; see the script docstring")
        for position, allowed in (
            (2, {"aux", "noaux"}),
            (3, {"adam", "sgd"}),
            (5, {"fp32_router", "hf_router"}),
            (6, {"probe", "trajectory", "forward_trajectory"}),
        ):
            if len(sys.argv) > position and sys.argv[position] not in allowed:
                raise ValueError(
                    f"Invalid argument {position}: {sys.argv[position]!r}; "
                    f"expected one of {sorted(allowed)}"
                )
        if len(sys.argv) > 6 and (not adam or sys.argv[5] != "fp32_router"):
            raise ValueError("Diagnostics require adam and fp32_router")
        cfg.model.arch_overrides["router_aux_loss_coef"] = 0.01 if aux else 0.0
        topology = ParallelTopology.from_config(cfg.parallel, 4)
        with cpu_cp_attention():
            model = _build(cfg).train()
            oracle = _build(cfg).train()
            fp32_router_reference = len(sys.argv) > 5 and sys.argv[5] == "fp32_router"
            if fp32_router_reference:

                def fp32_gate(module, hidden):
                    hidden = hidden.reshape(-1, module.hidden_dim)
                    logits = F.linear(hidden.float(), module.weight.float())
                    probabilities = logits.softmax(-1)
                    module.reference_probabilities = probabilities
                    scores, selected = probabilities.topk(module.top_k, dim=-1)
                    if module.norm_topk_prob:
                        scores = scores / scores.sum(-1, keepdim=True)
                    return logits, scores, selected

                for layer in oracle.layers:
                    layer.mlp.gate.forward = MethodType(fp32_gate, layer.mlp.gate)
            model = parallelize_hf_transformers(
                model,
                cfg=cfg.parallel,
                mesh=topology.spmd_dense_mesh(),
                parallel_dims=topology,
                device=torch.device("cpu"),
                activation_checkpoint=ac_mode,
            )
            activation_probe = len(sys.argv) > 6 and sys.argv[6] == "probe"
            forward_only = len(sys.argv) > 6 and sys.argv[6] == "forward_trajectory"
            trajectory = len(sys.argv) > 6 and sys.argv[6] in {
                "trajectory",
                "forward_trajectory",
            }
            comparison_failures = []

            def compare(actual, expected, **kwargs):
                try:
                    torch.testing.assert_close(actual, expected, **kwargs)
                except AssertionError as error:
                    if not trajectory:
                        raise
                    comparison_failures.append((step, str(error)))
                    print(
                        f"TRAJECTORY_DIFF rank={dist.get_rank()} step={step} "
                        f"dtype={actual.dtype} shape={tuple(actual.shape)} "
                        f"max_abs={(actual - expected).abs().max().item()}: {error}",
                        flush=True,
                    )

            observed_activations = {}
            reference_activations = {}
            if activation_probe or trajectory:
                for layer_index in range(2):
                    for name in (
                        "input_layernorm",
                        "self_attn.q_norm",
                        "self_attn.k_norm",
                        "self_attn",
                        "post_attention_layernorm",
                        "mlp",
                        "mlp.gate",
                    ):
                        for root, storage in (
                            (model, observed_activations),
                            (oracle, reference_activations),
                        ):
                            layer = root.layers[layer_index]
                            layer = getattr(layer, "_checkpoint_wrapped_module", layer)
                            module = layer.get_submodule(
                                "mlp.router.gate"
                                if name == "mlp.gate" and root is model
                                else name
                            )
                            key = f"{layer_index}:{name}"

                            def record_activation(
                                module, inputs, output, storage=storage, name=key
                            ):
                                if in_checkpoint_recompute.get():
                                    return
                                if isinstance(output, tuple):
                                    output = output[0]
                                entries = storage.setdefault(name, [])
                                entry = {"value": output.detach().clone()}
                                entries.append(entry)
                                if forward_only:
                                    return
                                if name.endswith(("q_norm", "k_norm")):
                                    entry["input"] = inputs[0].detach().clone()

                                    def save_input_gradient(gradient, entry=entry):
                                        entry["input_gradient"] = (
                                            gradient.detach().clone()
                                        )

                                    inputs[0].register_hook(save_input_gradient)
                                if output.requires_grad:

                                    def save_gradient(gradient, entry=entry):
                                        entry["gradient"] = gradient.detach().clone()

                                    output.register_hook(save_gradient)

                            module.register_forward_hook(record_activation)
            expert_records = [[], []]
            if activation_probe:
                for index, layer in enumerate(model.layers):
                    body = getattr(layer, "_checkpoint_wrapped_module", layer)

                    def record_experts(module, inputs, output, index=index):
                        if in_checkpoint_recompute.get():
                            return
                        entry = {
                            "input": inputs[0].detach().clone(),
                            "counts": inputs[1].detach().clone(),
                        }
                        expert_records[index].append(entry)

                        def save_expert_gradient(gradient, entry=entry):
                            entry["gradient"] = gradient.detach().clone()

                        output.register_hook(save_expert_gradient)

                    body.mlp.routed_experts.inner_experts.register_forward_hook(
                        record_experts
                    )
            layer_calls = [0, 0]
            for index, layer in enumerate(model.layers):
                body = getattr(layer, "_checkpoint_wrapped_module", layer)

                def count_execution(module, inputs, index=index):
                    layer_calls[index] += 1

                body.register_forward_pre_hook(count_execution)
            runtime = Trainer.__new__(Trainer)
            runtime.cfg = SimpleNamespace(
                gradient_accumulation_steps=2, max_seq_len=256, max_norm=1e6
            )
            runtime.parallel_dims = topology
            runtime.device = torch.device("cpu")
            runtime.model = model
            runtime.model_parts = [model]
            runtime.optimizer = torch.optim.SGD(
                model.parameters(), lr=0.02, momentum=0.9
            )
            if adam:
                runtime.optimizer = OptimizersContainer(
                    OptimizerConfig(
                        learning_rate=0.02,
                        weight_decay=0.1,
                        implementation="for-loop",
                    ),
                    model_parts=[model],
                    initialization_group=dist.group.WORLD,
                )
            inner_optimizer = (
                runtime.optimizer.optimizers[0] if adam else runtime.optimizer
            )
            # A second oracle isolates optimizer arithmetic from model-gradient
            # agreement: evolve independent weights/moments from the initial
            # values and observed gradients, never from live optimizer state.
            adam_formula_state = {}
            if adam:
                for group in inner_optimizer.param_groups:
                    assert group["betas"] == (0.9, 0.999)
                    assert group["eps"] == 1e-8
                    assert group["weight_decay"] == 0.1
                for name, parameter in model.named_parameters():
                    weight = full(parameter).detach().clone()
                    adam_formula_state[canonical_fqn(name)] = (
                        weight,
                        torch.zeros_like(weight),
                        torch.zeros_like(weight),
                    )
            register_aux_loss_metrics_hook(runtime.optimizer, model_parts=[model])
            runtime.lr_scheduler = build_lr_scheduler(
                LRSchedulerConfig(warmup_steps=2, total_steps=4, decay_ratio=0.5),
                optimizers=[inner_optimizer],
                training_steps=3,
            )
            runtime.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
            runtime.ema = EMA(model_parts=[model], decay=0.75)
            runtime.ntokens_seen = 0
            runtime._chunked_loss_num_chunks = 1
            runtime.should_log = lambda: True
            reference_optimizer = torch.optim.SGD(
                oracle.parameters(), lr=0.02, momentum=0.9
            )
            if adam:
                reference_optimizer = torch.optim.AdamW(
                    oracle.parameters(),
                    lr=0.02,
                    weight_decay=0.1,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    foreach=False,
                    fused=False,
                )
            optimizer_name = "AdamW" if adam else "SGD"
            state_names = ("exp_avg", "exp_avg_sq") if adam else ("momentum_buffer",)
            parameters = dict(oracle.named_parameters())
            reference_ema = {
                key: parameter.detach().clone() for key, parameter in parameters.items()
            }
            routes = []
            for layer in oracle.layers:
                layer.mlp.gate.register_forward_hook(
                    lambda module, inputs, output: routes.append(
                        (*output, getattr(module, "reference_probabilities", None))
                    )
                )
            for step in range(1, 4):
                calls_before = layer_calls.copy()
                runtime.step = step
                batches = []
                for index, tail in enumerate((64, 96)):
                    ids, labels, positions = _data(seed=step * 2 + index)
                    padding = torch.zeros(256, dtype=torch.bool)
                    padding[-tail:] = True
                    padding[16:24] = True
                    labels[padding] = -100
                    labels[: 16 if index == 0 else 32] = -100
                    batches.append(
                        dict(
                            input=ids,
                            labels=labels,
                            positions=positions,
                            padding_mask=padding,
                        )
                    )
                assert [Trainer.count_valid_tokens(b) for b in batches] == [168, 128]
                # Independent rates for warmup2/stable1/linear decay2, base0.02.
                applied_lr = (0.01, 0.02, 0.02)[step - 1]
                next_lr = (0.02, 0.02, 0.01)[step - 1]
                reference_optimizer.param_groups[0]["lr"] = applied_lr
                reference_optimizer.zero_grad(set_to_none=True)
                reference_selections = []
                reference_route_weights = []
                losses = []
                auxiliary = []
                auxiliary_objectives = []
                total_routed = sum(int((~b["padding_mask"]).sum()) for b in batches)
                assert total_routed == 336
                for batch in batches:
                    routes.clear()
                    logits = oracle(batch["input"], positions=batch["positions"])
                    losses.append(
                        F.cross_entropy(
                            logits.float(), batch["labels"], reduction="sum"
                        )
                    )
                    reference_route_weights.append(
                        [route[1].detach().clone() for route in routes]
                    )
                    reference_selections.append(
                        [route[2].detach().clone() for route in routes]
                    )
                    for gate_logits, _, selected, shared_probabilities in routes:
                        mask = ~batch["padding_mask"]
                        probabilities = F.normalize(
                            shared_probabilities
                            if shared_probabilities is not None
                            else gate_logits.float().softmax(-1),
                            p=1,
                            dim=-1,
                        )[mask]
                        selection = selected[mask]
                        counts = torch.bincount(
                            selection.flatten(), minlength=probabilities.shape[-1]
                        ).to(probabilities.dtype)
                        frequencies = (
                            F.normalize(counts, p=1, dim=0) * (probabilities.shape[-1])
                        )
                        raw_auxiliary = (frequencies * probabilities.sum(0)).sum()
                        auxiliary.append(raw_auxiliary / total_routed)
                        scale = 1.0 / torch.tensor(total_routed)
                        auxiliary_objectives.append(raw_auxiliary * (0.01 * scale))
                reference_loss = torch.stack(losses).sum() / 296
                reference_auxiliary = torch.stack(auxiliary).sum()
                auxiliary_objective = torch.stack(auxiliary_objectives).sum()
                if aux:
                    contribution = torch.autograd.grad(
                        auxiliary_objective,
                        [layer.mlp.gate.weight for layer in oracle.layers],
                        retain_graph=True,
                    )
                    assert all(value.abs().max() > 1e-6 for value in contribution)
                rounded_reference = {}
                if activation_probe and not aux:
                    projection_names = [
                        key for key in parameters if ".k_proj.weight" in key
                    ]
                    per_batch = [
                        torch.autograd.grad(
                            loss / 296,
                            [parameters[key] for key in projection_names],
                            retain_graph=True,
                        )
                        for loss in losses
                    ]
                    rounded_reference = {
                        key: sum(values[index].float().double() for values in per_batch)
                        for index, key in enumerate(projection_names)
                    }
                (reference_loss + (auxiliary_objective if aux else 0)).backward()
                before_update = (
                    {
                        canonical_fqn(name): full(parameter).detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                    if adam and step == 1
                    else {}
                )
                metrics = runtime.train_step(iter(batches))
                if adam:
                    formula_max_error = 0.0
                    for name, parameter in model.named_parameters():
                        key = canonical_fqn(name)
                        weight, first, second = adam_formula_state[key]
                        gradient = full(parameter.grad).detach()
                        first = first * 0.9 + gradient * (1 - 0.9)
                        second = second * 0.999 + gradient.square() * (1 - 0.999)
                        corrected_first = first / (1 - 0.9**step)
                        corrected_second = second / (1 - 0.999**step)
                        weight = weight * (1 - applied_lr * 0.1) - (
                            applied_lr
                            * corrected_first
                            / (corrected_second.sqrt() + 1e-8)
                        )
                        adam_formula_state[key] = (weight, first, second)
                        actual = full(parameter)
                        formula_max_error = max(
                            formula_max_error, (actual - weight).abs().max().item()
                        )
                        torch.testing.assert_close(
                            actual,
                            weight,
                            atol=2e-16,
                            rtol=1e-14,
                            msg=f"step={step} independent AdamW weight {key}",
                        )
                        for state_name, expected in (
                            ("exp_avg", first),
                            ("exp_avg_sq", second),
                        ):
                            torch.testing.assert_close(
                                full(inner_optimizer.state[parameter][state_name]),
                                expected,
                                atol=1e-18,
                                rtol=1e-12,
                                msg=f"step={step} independent {state_name} {key}",
                            )
                    print(
                        f"ADAM_FORMULA rank={dist.get_rank()} step={step} "
                        f"parameters={len(adam_formula_state)} "
                        f"max_weight_error={formula_max_error}",
                        flush=True,
                    )
                if activation_probe or trajectory:
                    for storage in (observed_activations, reference_activations):
                        assert all(len(entries) == 2 for entries in storage.values())
                    for name in observed_activations:
                        for batch_index in range(2):
                            fields = (
                                ["value"] if forward_only else ["value", "gradient"]
                            )
                            if not forward_only and name.endswith(("q_norm", "k_norm")):
                                fields += ["input", "input_gradient"]
                            for field in fields:
                                actual = observed_activations[name][batch_index][
                                    field
                                ].squeeze(0)
                                expected = reference_activations[name][batch_index][
                                    field
                                ].squeeze(0)
                                tp_mesh = topology.get_mesh("tp")
                                cp_mesh = topology.get_mesh("cp")
                                if not name.endswith(
                                    ("self_attn.q_norm", "self_attn.k_norm")
                                ):
                                    actual = _gather(_gather(actual, tp_mesh), cp_mesh)
                                else:
                                    chunks = [
                                        torch.empty_like(actual) for _ in range(2)
                                    ]
                                    dist.all_gather(
                                        chunks,
                                        actual.contiguous(),
                                        group=tp_mesh.get_group(),
                                    )
                                    actual = torch.cat(chunks, dim=-2)
                                    actual = _gather(actual, cp_mesh)
                                if dist.get_rank() == 0:
                                    print(
                                        "ACTIVATION_PROBE",
                                        f"step={step}",
                                        batch_index,
                                        dist.get_rank(),
                                        name,
                                        field,
                                        tuple(actual.shape),
                                        tuple(expected.shape),
                                        (actual - expected).abs().max().item(),
                                        flush=True,
                                    )
                if activation_probe:
                    assert all(len(entries) == 2 for entries in expert_records)
                    for layer_index in range(2):
                        for projection in ("q", "k"):
                            reconstructed = None
                            for batch_index in range(2):
                                x = reference_activations[
                                    f"{layer_index}:input_layernorm"
                                ][batch_index]["value"].squeeze(0)
                                dy = reference_activations[
                                    f"{layer_index}:self_attn.{projection}_norm"
                                ][batch_index]["input_gradient"].squeeze(0)
                                tp_rank = topology.get_mesh("tp").get_local_rank()
                                cp_rank = topology.get_mesh("cp").get_local_rank()
                                x = x.chunk(2, dim=0)[cp_rank]
                                dy = (
                                    dy.chunk(2, dim=0)[cp_rank]
                                    .chunk(2, dim=1)[tp_rank]
                                    .flatten(1)
                                )
                                local = (dy.T @ x).float()
                                dist.all_reduce(
                                    local, group=topology.get_mesh("cp").get_group()
                                )
                                value = local.double()
                                reconstructed = (
                                    value
                                    if reconstructed is None
                                    else reconstructed + value
                                )
                            layer = model.layers[layer_index]
                            layer = getattr(layer, "_checkpoint_wrapped_module", layer)
                            parameter = layer.self_attn.get_submodule(
                                f"{projection}_proj"
                            ).weight
                            torch.testing.assert_close(
                                full(parameter.grad),
                                reconstructed,
                                atol=0,
                                rtol=0,
                                msg=(
                                    f"local Q/K reconstruction: layer={layer_index} "
                                    f"projection={projection} aux={aux} "
                                    f"ac={ac_mode}; exact noaux contract"
                                ),
                            )
                            if adam and step == 1:
                                key = canonical_fqn(
                                    next(
                                        name
                                        for name, value in model.named_parameters()
                                        if value is parameter
                                    )
                                )
                                predicted = before_update[key] * (1 - applied_lr * 0.1)
                                predicted -= (
                                    applied_lr
                                    * reconstructed
                                    / (reconstructed.abs() + 1e-8)
                                )
                                torch.testing.assert_close(
                                    full(parameter), predicted, atol=2e-17, rtol=0
                                )
                                for state_name, target in (
                                    ("exp_avg", reconstructed * 0.1),
                                    ("exp_avg_sq", reconstructed.square() * 0.001),
                                ):
                                    torch.testing.assert_close(
                                        full(
                                            inner_optimizer.state[parameter][state_name]
                                        ),
                                        target,
                                        atol=1e-18,
                                        rtol=1e-12,
                                    )
                            error = (full(parameter.grad) - reconstructed).abs().max()
                            print(
                                "LOCAL_WGRAD_PROBE",
                                dist.get_rank(),
                                layer_index,
                                projection,
                                error.item(),
                                flush=True,
                            )
                    if adam and step == 1:
                        for index, records in enumerate(expert_records):
                            prefix = (
                                f"model.model.layers.{index}.mlp."
                                "routed_experts.inner_experts."
                            )
                            names = [
                                prefix + suffix
                                for suffix in ("w1_EFD", "w3_EFD", "w2_EDF")
                            ]
                            accumulated = None
                            for batch_index, entry in enumerate(records):
                                expected_inputs = reference_activations[
                                    f"{index}:post_attention_layernorm"
                                ][batch_index]["value"].squeeze(0)
                                selected = reference_selections[batch_index][index]
                                cp_rank = topology.get_mesh("cp").get_local_rank()
                                ep_rank = topology.get_mesh("ep").get_local_rank()
                                expected_inputs = expected_inputs.chunk(2)[cp_rank]
                                selected = selected.chunk(2)[cp_rank]
                                route_weights = reference_route_weights[batch_index][
                                    index
                                ].chunk(2)[cp_rank]
                                output_gradient = (
                                    reference_activations[f"{index}:mlp"][batch_index][
                                        "gradient"
                                    ]
                                    .squeeze(0)
                                    .chunk(2)[cp_rank]
                                )
                                matched_inputs = []
                                matched_gradients = []
                                offset_check = 0
                                for local_expert, count in enumerate(
                                    entry["counts"].tolist()
                                ):
                                    global_expert = ep_rank * 4 + local_expert
                                    token_rows = (selected == global_expert).nonzero()[
                                        :, 0
                                    ]
                                    expected_rows = expected_inputs[token_rows]
                                    actual_rows = entry["input"][
                                        offset_check : offset_check + count
                                    ]
                                    assert expected_rows.shape == actual_rows.shape
                                    # Activation matching needs unique rows.
                                    if (
                                        torch.unique(expected_rows, dim=0).shape[0]
                                        != count
                                    ):
                                        raise AssertionError(
                                            "Ambiguous expert token identity: "
                                            "use explicit "
                                            "token IDs for combine gradients"
                                        )
                                    # Sort rows after checking identity.
                                    ordered = []
                                    orders = []
                                    for rows in (actual_rows, expected_rows):
                                        order = torch.arange(rows.shape[0])
                                        for column in reversed(range(rows.shape[1])):
                                            order = order[
                                                torch.argsort(
                                                    rows[order, column], stable=True
                                                )
                                            ]
                                        ordered.append(rows[order])
                                        orders.append(order)
                                    torch.testing.assert_close(*ordered, atol=0, rtol=0)
                                    matches = (selected == global_expert).nonzero()
                                    expected_gradient = (
                                        output_gradient[token_rows]
                                        * (
                                            route_weights[
                                                matches[:, 0], matches[:, 1], None
                                            ]
                                        )
                                    )
                                    actual_gradient = entry["gradient"][
                                        offset_check : offset_check + count
                                    ]
                                    torch.testing.assert_close(
                                        actual_gradient[orders[0]],
                                        expected_gradient[orders[1]],
                                        atol=1e-18,
                                        rtol=1e-12,
                                        msg="expert combine weighted output gradient",
                                    )
                                    inverse_actual = torch.argsort(orders[0])
                                    expected_order = orders[1][inverse_actual]
                                    matched_inputs.append(expected_rows[expected_order])
                                    matched_gradients.append(
                                        expected_gradient[expected_order]
                                    )
                                    offset_check += count
                                assert offset_check == entry["input"].shape[0]
                                weights = [
                                    before_update[key].clone().requires_grad_()
                                    for key in names
                                ]
                                x = torch.cat(matched_inputs)
                                outputs = []
                                offset = 0
                                for expert, count in enumerate(
                                    entry["counts"].tolist()
                                ):
                                    rows = x[offset : offset + count]
                                    gate = F.linear(rows, weights[0][expert])
                                    up = F.linear(rows, weights[1][expert])
                                    outputs.append(
                                        F.linear(F.silu(gate) * up, weights[2][expert])
                                    )
                                    offset += count
                                grads = torch.autograd.grad(
                                    torch.cat(outputs),
                                    weights,
                                    grad_outputs=torch.cat(matched_gradients),
                                )
                                reduced = []
                                for gradient in grads:
                                    gradient = gradient.float()
                                    dist.all_reduce(
                                        gradient,
                                        group=topology.get_mesh("cp").get_group(),
                                    )
                                    reduced.append(gradient.double())
                                accumulated = (
                                    reduced
                                    if accumulated is None
                                    else [
                                        a + b
                                        for a, b in zip(
                                            accumulated, reduced, strict=True
                                        )
                                    ]
                                )
                            live = dict(model.named_parameters())
                            for key, gradient in zip(names, accumulated, strict=True):
                                parameter = next(
                                    value
                                    for name, value in live.items()
                                    if canonical_fqn(name) == key
                                )
                                torch.testing.assert_close(
                                    full(parameter.grad), gradient, atol=0, rtol=0
                                )
                                predicted = before_update[key] * (1 - applied_lr * 0.1)
                                predicted -= (
                                    applied_lr * gradient / (gradient.abs() + 1e-8)
                                )
                                torch.testing.assert_close(
                                    full(parameter), predicted, atol=2e-17, rtol=0
                                )
                                for state_name, target in (
                                    ("exp_avg", gradient * 0.1),
                                    ("exp_avg_sq", gradient.square() * 0.001),
                                ):
                                    torch.testing.assert_close(
                                        full(
                                            inner_optimizer.state[parameter][state_name]
                                        ),
                                        target,
                                        atol=1e-18,
                                        rtol=1e-12,
                                    )
                                print(
                                    "EXPERT_WGRAD_PROBE",
                                    dist.get_rank(),
                                    key,
                                    (full(parameter.grad) - gradient)
                                    .abs()
                                    .max()
                                    .item(),
                                    flush=True,
                                )
                    for records in expert_records:
                        records.clear()
                observed_activations.clear()
                reference_activations.clear()
                compare(
                    torch.tensor(metrics["loss"], dtype=reference_loss.dtype),
                    reference_loss.detach(),
                    atol=1e-6,
                    rtol=1e-6,
                )
                assert metrics[f"lr/{optimizer_name}"] == applied_lr
                assert inner_optimizer.param_groups[0]["lr"] == next_lr
                assert runtime.lr_scheduler.state_dict() == {"last_epoch": step}
                executions = [
                    after - before
                    for after, before in zip(layer_calls, calls_before, strict=True)
                ]
                if ac_mode == "full":
                    assert all(count >= 4 for count in executions), executions
                else:
                    assert executions == [2, 2], executions
                if aux:
                    names = {
                        module.metric_name
                        for module in model.modules()
                        if isinstance(module, AuxLoss)
                    }
                    assert len(names) == 1
                    actual_aux = metrics[f"{names.pop()}/mean"]
                    compare(
                        torch.tensor(actual_aux),
                        reference_auxiliary.detach() / 2,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                reference_optimizer.step()
                with torch.no_grad():
                    for key, parameter in parameters.items():
                        reference_ema[key].mul_(0.75).add_(parameter, alpha=0.25)
                reference_states = {
                    state_name: {
                        key: reference_optimizer.state[parameter][state_name]
                        for key, parameter in parameters.items()
                    }
                    for state_name in state_names
                }
                tp_rank = topology.get_mesh("tp").get_local_rank()
                ep_rank = topology.get_mesh("ep").get_local_rank()
                count = 0
                for name, parameter in model.named_parameters():
                    key = canonical_fqn(name)
                    if (
                        adam
                        and step == 1
                        and ("w1_EFD" in key or ".k_proj.weight" in key)
                    ):
                        observed_gradient = full(parameter.grad)
                        expected_gradient = oracle_view(
                            parameters, key, tp_rank, ep_rank, gradient=True
                        )
                        if key in rounded_reference:
                            rounded = oracle_view(
                                rounded_reference,
                                key,
                                tp_rank,
                                ep_rank,
                                gradient=False,
                            )
                            print(
                                "REDUCE_DTYPE_PROBE",
                                dist.get_rank(),
                                key,
                                "fp64_error",
                                (observed_gradient - expected_gradient)
                                .abs()
                                .max()
                                .item(),
                                "rounded_error",
                                (observed_gradient - rounded).abs().max().item(),
                                flush=True,
                            )
                        actual_weight = full(parameter)
                        expected_weight = oracle_view(
                            parameters, key, tp_rank, ep_rank, gradient=False
                        )
                        predicted = before_update[key] * (1 - applied_lr * 0.1)
                        predicted = predicted - applied_lr * observed_gradient / (
                            observed_gradient.abs() + 1e-8
                        )
                        difference = (actual_weight - expected_weight).abs()
                        flat_index = int(difference.argmax())
                        print(
                            "ADAM_DIAGNOSTIC",
                            {
                                "rank": dist.get_rank(),
                                "key": key,
                                "dtype": str(actual_weight.dtype),
                                "shape": tuple(actual_weight.shape),
                                "flat_index": flat_index,
                                "grad_actual": observed_gradient.flatten()[
                                    flat_index
                                ].item(),
                                "grad_reference": expected_gradient.flatten()[
                                    flat_index
                                ].item(),
                                "weight_error": difference.max().item(),
                                "same_grad_formula_error": (predicted - actual_weight)
                                .abs()
                                .max()
                                .item(),
                            },
                            flush=True,
                        )
                    for actual, gradient in (
                        (full(parameter.grad), True),
                        (full(parameter), False),
                    ):
                        compare(
                            actual,
                            oracle_view(
                                parameters, key, tp_rank, ep_rank, gradient=gradient
                            ),
                            atol=1e-6,
                            rtol=1e-6,
                            msg=lambda error, key=key, gradient=gradient, step=step: (
                                f"step={step} gradient={gradient} {key}: {error}"
                            ),
                        )
                    for state_name in state_names:
                        compare(
                            full(inner_optimizer.state[parameter][state_name]),
                            oracle_view(
                                reference_states[state_name],
                                key,
                                tp_rank,
                                ep_rank,
                                gradient=False,
                            ),
                            atol=1e-6,
                            rtol=1e-6,
                            msg=lambda error, key=key: f"optimizer {key}: {error}",
                        )
                    if adam:
                        assert inner_optimizer.state[parameter]["step"].item() == step
                    compare(
                        full(runtime.ema.optimizers[0].state[parameter]["ema_params"]),
                        oracle_view(
                            reference_ema, key, tp_rank, ep_rank, gradient=False
                        ),
                        atol=1e-6,
                        rtol=1e-6,
                        msg=lambda error, key=key: f"EMA {key}: {error}",
                    )
                    count += 1
                assert count == 27
                assert runtime.ntokens_seen == step * 128
                assert metrics["n_tokens_seen"] == step * 512
                print(
                    f"rank={dist.get_rank()} {implementation} aux={aux} "
                    f"{optimizer_name} step={step}: "
                    f"Trainer unequal-token accumulation/ac={ac_mode}: "
                    f"27 parameters compared; failures="
                    f"{sum(item[0] == step for item in comparison_failures)}; "
                    f"body calls={executions}",
                    flush=True,
                )
            if comparison_failures:
                raise AssertionError(
                    f"{len(comparison_failures)} trajectory comparisons failed; "
                    f"first step={comparison_failures[0][0]}: "
                    f"{comparison_failures[0][1]}"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
