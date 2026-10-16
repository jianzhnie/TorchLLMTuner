"""Four CPU/Gloo ranks: TP2/KVCP2/EP2/FSDP against independent HF training.

Run with torchrun --nproc_per_node=4, followed by this script and
optional native/custom (default custom). Uses the explicit dense-mask SDPA
substitute from pp_cp_equivalence; production CP communication/autograd,
TP, EP all-to-all and FSDP remain active. Pass a second argument "aux"
to include the sequence-wise balance objective with coefficient 0.01.
Append "padding" after "aux" to mask an entire rank and an interior span,
while retaining prompt tokens in the routing denominator.
Append full/selective after padding to enable production activation checkpointing.
Append an accumulation count after the AC mode to sum distinct microbatches.
Append uneven after count 2 to test unequal valid-token microbatches.
No production flex-kernel or target-device acceptance.
"""

import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.spmd import spmd_context
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention
from tests.integration_tests.tp_ep_cp_equivalence import (
    SEQ,
    _build,
    _cfg,
    _data,
    _gather,
)


def oracle_view(parameters, key, tp_rank, ep_rank, *, gradient):
    """Independent mapping of this Qwen fixture's physical TP/EP slices."""
    if ".router.gate.weight" in key:
        key = key.replace(".router.gate.weight", ".gate.weight")
    for suffix, source, half in (
        ("w1_EFD", "gate_up_proj", 0),
        ("w3_EFD", "gate_up_proj", 1),
        ("w2_EDF", "down_proj", None),
    ):
        marker = f"routed_experts.inner_experts.{suffix}"
        if marker in key:
            parameter = parameters[key.replace(marker, f"experts.{source}")]
            value = parameter.grad if gradient else parameter
            assert value is not None, key
            if half is not None:
                value = value.chunk(2, dim=1)[half]
            return value.chunk(2, dim=0)[ep_rank]
    parameter = parameters[key]
    value = parameter.grad if gradient else parameter
    assert value is not None, key
    if any(f".{name}.weight" in key for name in ("q_proj", "k_proj", "v_proj")):
        return value.chunk(2, dim=0)[tp_rank]
    if ".o_proj.weight" in key:
        return value.chunk(2, dim=1)[tp_rank]
    return value


def full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        cfg = _cfg()
        aux_coefficient = 0.01 if len(sys.argv) > 2 and sys.argv[2] == "aux" else 0.0
        padded = len(sys.argv) > 3 and sys.argv[3] == "padding"
        ac_mode = sys.argv[4] if len(sys.argv) > 4 else "none"
        assert ac_mode in {"none", "full", "selective"}
        accumulation = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        assert accumulation > 0
        uneven = len(sys.argv) > 6 and sys.argv[6] == "uneven"
        assert not uneven or (padded and accumulation == 2)
        cfg.model.arch_overrides["router_aux_loss_coef"] = aux_coefficient
        implementation = sys.argv[1] if len(sys.argv) > 1 else "custom"
        assert implementation in {"native", "custom"}
        cfg.parallel.fsdp_config.fsdp_implementation = implementation
        topology = ParallelTopology.from_config(cfg.parallel, 4)
        tp = topology.get_mesh("tp")
        cp = topology.get_mesh("cp")
        ep = topology.get_mesh("ep")
        with cpu_cp_attention():
            model = _build(cfg).train()
            oracle = _build(cfg).train()
            model = parallelize_hf_transformers(
                model,
                cfg=cfg.parallel,
                mesh=topology.spmd_dense_mesh(),
                parallel_dims=topology,
                device=torch.device("cpu"),
                activation_checkpoint=ac_mode,
                selective_ac=cfg.training.selective_ac,
            )
            assert any(isinstance(p, DTensor) for p in model.parameters())
            if ac_mode != "none":
                from torch.distributed.algorithms._checkpoint import checkpoint_wrapper

                assert (
                    sum(
                        isinstance(module, checkpoint_wrapper.CheckpointWrapper)
                        for module in model.modules()
                    )
                    == 2
                )
            layer_calls = [0, 0]
            for index, layer in enumerate(model.layers):
                # Count the actual decoder body, inside the checkpoint wrapper.
                body = getattr(layer, "_checkpoint_wrapped_module", layer)

                def count_execution(module, inputs, index=index):
                    layer_calls[index] += 1

                body.register_forward_pre_hook(count_execution)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            oracle_optimizer = torch.optim.SGD(
                oracle.parameters(), lr=0.02, momentum=0.9
            )
            parameters = dict(oracle.named_parameters())
            oracle_routes = []
            for layer in oracle.layers:
                layer.mlp.gate.register_forward_hook(
                    lambda module, inputs, output: oracle_routes.append(output)
                )
            runtime = SimpleNamespace(parallel_dims=topology, model_parts=[model])
            for step in range(3):
                calls_before = layer_calls.copy()
                optimizer.zero_grad(set_to_none=True)
                oracle_optimizer.zero_grad(set_to_none=True)
                expected_aux_totals = [
                    torch.zeros((), dtype=torch.float64) for _ in range(2)
                ]
                batches = []
                for microbatch in range(accumulation):
                    ids, labels, positions = _data(
                        seed=step * accumulation + microbatch
                    )
                    padding = torch.zeros(SEQ, dtype=torch.bool)
                    if padded:
                        tail = 96 if uneven and microbatch == 1 else 64
                        padding[-tail:] = True
                        padding[16:24] = True
                        labels[padding] = -100
                        prompt = 32 if uneven and microbatch == 1 else 16
                        labels[:prompt] = -100
                    routed_tokens = int((~padding).sum())
                    supervised_tokens = int((labels != -100).sum())
                    batches.append(
                        (
                            ids,
                            labels,
                            positions,
                            padding,
                            routed_tokens,
                            supervised_tokens,
                        )
                    )
                total_routed = sum(batch[4] for batch in batches)
                total_supervised = sum(batch[5] for batch in batches)
                if uneven:
                    assert [(b[4], b[5]) for b in batches] == [(184, 168), (152, 128)]
                    assert (total_routed, total_supervised) == (336, 296)
                AuxLoss.set_step_denominator(
                    torch.tensor(total_routed), model_parts=[model]
                )
                for microbatch, batch in enumerate(batches):
                    (
                        ids,
                        labels,
                        positions,
                        padding,
                        routed_tokens,
                        supervised_tokens,
                    ) = batch
                    # Independent contiguous sequence slicing, without reusing the
                    # production input sharder as the source of expected ownership.
                    cp_ids = ids.chunk(2)[cp.get_local_rank()]
                    cp_labels = labels.chunk(2)[cp.get_local_rank()]
                    cp_positions = positions.chunk(2)[cp.get_local_rank()]
                    local_ids = cp_ids.chunk(2)[tp.get_local_rank()]
                    local_labels = cp_labels.chunk(2)[tp.get_local_rank()]
                    local_padding = padding.chunk(2)[cp.get_local_rank()].chunk(2)[
                        tp.get_local_rank()
                    ]
                    if padded and cp.get_local_rank() == tp.get_local_rank() == 1:
                        assert local_padding.all() and (local_labels == -100).all()
                    with spmd_context(topology):
                        output = model(
                            local_ids,
                            positions=cp_positions,
                            padding_mask=local_padding,
                        )
                    oracle_routes.clear()
                    expected = oracle(ids, positions=positions)
                    joined = _gather(_gather(output, tp), cp)
                    torch.testing.assert_close(joined, expected, atol=1e-6, rtol=1e-6)
                    loss = (
                        torch.nn.functional.cross_entropy(
                            output, local_labels, reduction="sum"
                        )
                        / total_supervised
                    )
                    reference_loss = torch.nn.functional.cross_entropy(
                        expected, labels
                    ) * (supervised_tokens / total_supervised)
                    global_loss = loss.detach().clone()
                    dist.all_reduce(global_loss)
                    torch.testing.assert_close(
                        global_loss, reference_loss, atol=1e-6, rtol=1e-6
                    )
                    # Independent full-stream formula, with discrete selection
                    # from HF's own router. No production AuxLoss code is reused.
                    auxiliary_reference = reference_loss.new_zeros(())
                    reference_aux_layers = []
                    for logits, _, selected in oracle_routes:
                        probabilities = logits.float().softmax(-1)[~padding]
                        valid_selection = selected[~padding]
                        counts = torch.bincount(
                            valid_selection.flatten(), minlength=probabilities.shape[-1]
                        ).to(probabilities.dtype)
                        frequencies = counts * (
                            probabilities.shape[-1] / valid_selection.numel()
                        )
                        layer_auxiliary = (frequencies * probabilities.mean(0)).sum()
                        reference_aux_layers.append(
                            layer_auxiliary * (routed_tokens / total_routed)
                        )
                        auxiliary_reference = auxiliary_reference + layer_auxiliary * (
                            routed_tokens / total_routed
                        )
                    assert len(oracle_routes) == 2
                    for index, value in enumerate(reference_aux_layers):
                        expected_aux_totals[index] += value.detach().double()
                    if aux_coefficient:
                        actual_aux_layers = [
                            module
                            for module in model.modules()
                            if isinstance(module, AuxLoss)
                        ]
                        assert len(actual_aux_layers) == len(reference_aux_layers)
                        for actual, target in zip(
                            actual_aux_layers, expected_aux_totals, strict=True
                        ):
                            torch.testing.assert_close(
                                actual.instance_acc.double(),
                                target.double(),
                                atol=1e-6,
                                rtol=1e-6,
                            )
                        auxiliary_gradients = torch.autograd.grad(
                            aux_coefficient * auxiliary_reference,
                            [layer.mlp.gate.weight for layer in oracle.layers],
                            retain_graph=True,
                        )
                        # The auxiliary contribution must exceed the comparison
                        # tolerance; CE-only gradients cannot pass this by accident.
                        assert all(
                            g.abs().max().item() > 1e-6 for g in auxiliary_gradients
                        )
                    with spmd_context(topology):
                        loss.backward()
                    (reference_loss + aux_coefficient * auxiliary_reference).backward()
                    executions = [
                        after - before
                        for after, before in zip(layer_calls, calls_before, strict=True)
                    ]
                    if ac_mode == "none":
                        assert executions == [microbatch + 1] * 2, executions
                    else:
                        assert all(
                            count >= 2 * (microbatch + 1) for count in executions
                        ), executions
                    if aux_coefficient:
                        # Replay must not add a second copy of metric scratch.
                        for actual, target in zip(
                            actual_aux_layers, expected_aux_totals, strict=True
                        ):
                            torch.testing.assert_close(
                                actual.instance_acc.double(),
                                target.double(),
                                atol=1e-6,
                                rtol=1e-6,
                            )
                Trainer._allreduce_replicated_tp_grads(runtime)
                count = 0
                for key, parameter in model.named_parameters():
                    assert parameter.grad is not None, key
                    target = oracle_view(
                        parameters,
                        canonical_fqn(key),
                        tp.get_local_rank(),
                        ep.get_local_rank(),
                        gradient=True,
                    )
                    torch.testing.assert_close(
                        full(parameter.grad),
                        target,
                        atol=1e-6,
                        rtol=1e-6,
                        msg=lambda error,
                        step=step,
                        key=key: f"step={step} gradient {key}: {error}",
                    )
                    count += 1
                assert count == 27, count
                optimizer.step()
                oracle_optimizer.step()
                oracle_momentum = {
                    key: oracle_optimizer.state[parameter]["momentum_buffer"]
                    for key, parameter in parameters.items()
                }
                for key, parameter in model.named_parameters():
                    target = oracle_view(
                        parameters,
                        canonical_fqn(key),
                        tp.get_local_rank(),
                        ep.get_local_rank(),
                        gradient=False,
                    )
                    torch.testing.assert_close(
                        full(parameter),
                        target,
                        atol=1e-6,
                        rtol=1e-6,
                        msg=lambda error,
                        step=step,
                        key=key: f"step={step} parameter {key}: {error}",
                    )
                    expected_momentum = oracle_view(
                        oracle_momentum,
                        canonical_fqn(key),
                        tp.get_local_rank(),
                        ep.get_local_rank(),
                        gradient=False,
                    )
                    torch.testing.assert_close(
                        full(optimizer.state[parameter]["momentum_buffer"]),
                        expected_momentum,
                        atol=1e-6,
                        rtol=1e-6,
                        msg=lambda error,
                        step=step,
                        key=key: f"step={step} momentum {key}: {error}",
                    )
                print(
                    f"rank={dist.get_rank()} {implementation} aux={aux_coefficient} "
                    f"padding={padded} ac={ac_mode} accumulation={accumulation} "
                    f"uneven={uneven} "
                    f"step={step + 1}: "
                    f"{count} gradients/updates/momentum passed; "
                    f"layer calls={executions}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
