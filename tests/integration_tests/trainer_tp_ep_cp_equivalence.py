"""Four CPU/Gloo ranks: real Trainer steps against independent HF CE.

Run with torchrun --nproc_per_node=4; optional native/custom (default custom).
Append aux to validate coefficient 0.01 against an independent routing formula.
Append adam after aux to use the production AdamW optimizer container.
Uses the explicit CPU attention substitute, not target-device flex acceptance.
Model assembly, preprocessing, token accounting, accumulation, replicated gradient
reduction, clipping, optimizer ordering and metrics use production Trainer code.
"""

import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.models.common.aux_loss import AuxLoss, register_aux_loss_metrics_hook
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention
from tests.integration_tests.tp_ep_cp_equivalence import _build, _cfg, _data
from tests.integration_tests.tp_ep_cp_training_equivalence import full, oracle_view


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        cfg = _cfg()
        implementation = sys.argv[1] if len(sys.argv) > 1 else "custom"
        assert implementation in {"native", "custom"}
        cfg.parallel.fsdp_config.fsdp_implementation = implementation
        aux = len(sys.argv) > 2 and sys.argv[2] == "aux"
        adam = len(sys.argv) > 3 and sys.argv[3] == "adam"
        cfg.model.arch_overrides["router_aux_loss_coef"] = 0.01 if aux else 0.0
        topology = ParallelTopology.from_config(cfg.parallel, 4)
        with cpu_cp_attention():
            model = _build(cfg).train()
            oracle = _build(cfg).train()
            model = parallelize_hf_transformers(
                model,
                cfg=cfg.parallel,
                mesh=topology.spmd_dense_mesh(),
                parallel_dims=topology,
                device=torch.device("cpu"),
                activation_checkpoint="full",
            )
            layer_calls = [0, 0]
            for index, layer in enumerate(model.layers):
                body = layer._checkpoint_wrapped_module

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
                    lambda module, inputs, output: routes.append(output)
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
                losses = []
                auxiliary = []
                total_routed = sum(int((~b["padding_mask"]).sum()) for b in batches)
                assert total_routed == 336
                for batch in batches:
                    routes.clear()
                    logits = oracle(batch["input"], positions=batch["positions"])
                    losses.append(
                        F.cross_entropy(logits, batch["labels"], reduction="sum")
                    )
                    for gate_logits, _, selected in routes:
                        mask = ~batch["padding_mask"]
                        probabilities = gate_logits.float().softmax(-1)[mask]
                        selection = selected[mask]
                        counts = torch.bincount(
                            selection.flatten(), minlength=probabilities.shape[-1]
                        ).to(probabilities.dtype)
                        frequencies = counts * (
                            probabilities.shape[-1] / selection.numel()
                        )
                        auxiliary.append(
                            (frequencies * probabilities.mean(0)).sum()
                            * (int(mask.sum()) / total_routed)
                        )
                reference_loss = torch.stack(losses).sum() / 296
                reference_auxiliary = torch.stack(auxiliary).sum()
                if aux:
                    contribution = torch.autograd.grad(
                        0.01 * reference_auxiliary,
                        [layer.mlp.gate.weight for layer in oracle.layers],
                        retain_graph=True,
                    )
                    assert all(value.abs().max() > 1e-6 for value in contribution)
                (reference_loss + (0.01 if aux else 0) * reference_auxiliary).backward()
                before_update = (
                    {
                        canonical_fqn(name): full(parameter).detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                    if adam and step == 1
                    else {}
                )
                metrics = runtime.train_step(iter(batches))
                torch.testing.assert_close(
                    torch.tensor(metrics["loss"], dtype=torch.float64),
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
                assert all(count >= 4 for count in executions), executions
                if aux:
                    names = {
                        module.metric_name
                        for module in model.modules()
                        if isinstance(module, AuxLoss)
                    }
                    assert len(names) == 1
                    actual_aux = metrics[f"{names.pop()}/mean"]
                    torch.testing.assert_close(
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
                    if adam and step == 1 and "w1_EFD" in key:
                        observed_gradient = full(parameter.grad)
                        expected_gradient = oracle_view(
                            parameters, key, tp_rank, ep_rank, gradient=True
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
                        torch.testing.assert_close(
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
                        torch.testing.assert_close(
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
                    torch.testing.assert_close(
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
                    "Trainer unequal-token accumulation/full AC: "
                    "27 gradients/updates/optimizer states/EMA OK; "
                    f"body calls={executions}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
