"""TP x FSDP check: the combined mesh must train like the single-rank model.

Run under torchrun with 4 ranks (from the repo root):

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/tp_fsdp_equivalence.py

tp=2 + dp_shard=2: every rank holds a T/tp sequence shard (the sequence-
parallelism premise) AND a dp_shard parameter shard. A tiny offline qwen3
(random init, fixed seed) goes through the real ``Trainer`` for 4 optimizer
steps and is compared against a single-process reference over the same global
batches -- the loss trajectory AND the final parameters.

Two seams are pinned at once:

* the reported loss: each rank's loss sum covers only its ``T / tp`` token
  shard, so the loss reduce-group must span the tp axis (the ``loss`` view
  includes it); without that the report is 1/tp of the true value. The token
  count needs no tp factor: TP ranks of a DP group read the same rows, and
  the count is taken from the unsharded batch.
* the gradient wiring: TP-sharded weights reduce inside the fused GEMMs,
  TP-replicated ones (embedding, norms, head) through the trainer's
  ``_allreduce_replicated_tp_grads``, and FSDP reduces/shards everything over
  dp_shard -- which must NOT include the tp axis, or replicas' partial
  gradients would be averaged instead of summed.

Non-vacuity: TP weight shards are literal slices of the reference weights and
the two tp ranks hold DIFFERENT slices; FSDP parameters are DTensors whose
local shard is half the global tensor. A combination that silently skipped
either reduction would diverge from the reference within the tolerances below.

Runs in FP32 on the device/backend selected by the integration harness,
including NPU/HCCL; CPU/Gloo uses the fallback TP collectives.
Set TP_FSDP_EQ_DP_SHARD=1 and TP_FSDP_EQ_DP_REPLICATE=2 for replicated DP;
set both to 2 and launch eight ranks for HSDP.
Set TP_FSDP_EQ_EP=2 for a Qwen3 MoE fixture with real Trainer/AdamW and
an independent unsharded HF reference. EP shares the dense rank space.
Set TP_FSDP_EQ_AC=full or selective to compare checkpointed Trainer execution
with the uncheckpointed reference; checks actual wrappers and decoder replay.
Set TP_FSDP_EQ_CPU_OFFLOAD=1 on an accelerator host for CPU parameter, gradient
and AdamW moment storage, with pin_memory=False and the for-loop optimizer.
Only final comparison copies the offloaded DTensors back to the accelerator.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.components.loss import IGNORE_INDEX, cross_entropy_loss
from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.config.parallel_config import ExpertParallelConfig, FSDPConfig
from llmtuner.datasets.contracts import Batch
from llmtuner.datasets.synthetic import RandomTokenSource
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.tensor_parallel.tensor_parallel import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
)
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.harness import device, moved, moved_batch

STEPS = 4
GLOBAL_BATCH = 8
SEQ = 32
VOCAB = 128
SEED = 42

TP = 2
DP_SHARD = int(os.environ.get("TP_FSDP_EQ_DP_SHARD", "2"))
DP_REPLICATE = int(os.environ.get("TP_FSDP_EQ_DP_REPLICATE", "1"))
DP_SIZE = DP_SHARD * DP_REPLICATE
EP = int(os.environ.get("TP_FSDP_EQ_EP", "1"))
DISPATCHER = os.environ.get("TP_FSDP_EQ_DISPATCHER", "alltoall")
CPU_OFFLOAD = os.environ.get("TP_FSDP_EQ_CPU_OFFLOAD", "0") == "1"
if EP not in (1, 2):
    raise ValueError("this fixture supports EP 1 or 2")

# fp32; the sanctioned divergences are summation orders (gathered GEMMs,
# per-rank reduce-scatter), which sit at ~1e-6 relative. A wiring error (a
# missing tp reduction, an averaged-instead-of-summed gradient) is O(1).
TOL = 1e-4


def _cfg() -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            # llama, not qwen3: qwen3's HF tp_config carries
            # ``replicated_with_grad_allreduce`` norm entries that the minimal
            # TP engine deliberately rejects (apply_tp fails loudly on them).
            model_name_or_path="qwen3_moe" if EP > 1 else "llama",
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides=(
                {
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 48,
                    "norm_topk_prob": True,
                    "router_aux_loss_coef": 0.0,
                }
                if EP > 1
                else {}
            ),
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=TP,
            expert_parallel_size=EP,
            ep_config=ExpertParallelConfig(ep_token_dispatcher=DISPATCHER),
            data_parallel_shard_size=DP_SHARD,
            data_parallel_replicate_size=DP_REPLICATE,
            fsdp_config=FSDPConfig(
                fsdp_cpu_offload=CPU_OFFLOAD,
                fsdp_cpu_offload_pin_memory=not CPU_OFFLOAD,
            ),
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-4,
            weight_decay=0.0,
            implementation="for-loop" if CPU_OFFLOAD else "fused",
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            max_seq_len=SEQ,
            steps=STEPS,
            seed=SEED,
            deterministic=True,
            activation_checkpoint_mode=os.environ.get("TP_FSDP_EQ_AC", "none"),
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _reference_trajectory(
    cfg: LLMTunerConfig,
) -> tuple[list[float], list[float], dict]:
    """The same training with no parallelism: one process, the whole batch.

    Mirrors the trainer's non-PP step arithmetic exactly. One subtlety: dp
    sharding turns the global batch into per-group flattened sequences (each
    group's rows collapse into one causal sequence with its own arange
    positions), so the reference must run the DP groups as SEPARATE forwards
    and sum -- a single flat forward over the union would give the second
    group's rows different positions and a different causal boundary. The
    target shift is row-local, so the full-batch targets still double as the
    token count.
    """
    torch.manual_seed(cfg.seed)
    model = moved(HFTransformerModel(build_model_config_for(cfg)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    batches = iter(
        RandomTokenSource(
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            batch_size=cfg.global_batch_size,
            seq_len=cfg.max_seq_len,
        )
    )
    rows_per_dp = cfg.global_batch_size // DP_SIZE
    losses = []
    norms = []
    for _ in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        batch = moved_batch(next(batches))
        num_valid = int(
            (
                model.preprocess_inputs(batch, parallel_dims=None)[1] != IGNORE_INDEX
            ).sum()
        )
        loss_sum = None
        for d in range(DP_SIZE):
            rows = slice(d * rows_per_dp, (d + 1) * rows_per_dp)
            sub = Batch(input_ids=batch.input_ids[rows], labels=batch.labels[rows])
            inputs, targets, _ = model.preprocess_inputs(sub, parallel_dims=None)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss_sum = loss if loss_sum is None else loss_sum + loss
        # Normalized BEFORE backward, as the trainer does, so the clip below
        # reads the gradient the optimizer actually applied.
        (loss_sum / num_valid).backward()
        norm = clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm, foreach=True)
        norms.append(float(norm))
        optimizer.step()
        losses.append(float(loss_sum.detach() / num_valid))
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    return losses, norms, params


def _full(t: torch.Tensor) -> torch.Tensor:
    """Global tensor at this rank's TP coordinate: redistribute the FSDP shard."""
    if isinstance(t, DTensor) and t.device.type == "cpu" and CPU_OFFLOAD:
        t = t.to(device())
    return t.full_tensor() if isinstance(t, DTensor) else t.detach()


def _assert_cpu_offload(trainer: Trainer) -> None:
    """Check actual storage and initialized AdamW moments after every step."""
    checked = 0
    for optimizer in trainer.optimizer.optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                assert isinstance(parameter, DTensor)
                assert parameter.to_local().device.type == "cpu"
                assert parameter.grad is not None
                assert isinstance(parameter.grad, DTensor)
                assert parameter.grad.to_local().device.type == "cpu"
                state = optimizer.state[parameter]
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = state[key]
                    assert isinstance(moment, DTensor)
                    assert moment.to_local().device.type == "cpu"
                    assert moment.shape == parameter.shape
                checked += 1
    assert checked > 0, "CPU offload storage check was vacuous"


def main() -> None:
    cfg = _cfg()
    failures: list[str] = []

    # Trainer init owns the process group (torchrun env): tp=2, dp_shard=2.
    trainer = Trainer(cfg)
    rank = trainer.rank
    world = trainer.world_size
    assert world == TP * DP_SIZE, f"expected {TP * DP_SIZE} ranks, got {world}"
    assert GLOBAL_BATCH % DP_SIZE == 0
    parallel_dims = trainer.parallel_dims
    assert isinstance(parallel_dims, ParallelTopology)
    tp_rank = parallel_dims.get_mesh("tp").get_local_rank()
    if EP > 1:
        from llmtuner.models.common.moe.layer import MoE

        assert parallel_dims.get_mesh("ep").size() == EP
        moe_layers = [
            module
            for module in trainer.model_parts[0].modules()
            if isinstance(module, MoE)
        ]
        assert len(moe_layers) == cfg.model.num_hidden_layers
        for module in moe_layers:
            if DISPATCHER == "torchao":
                from llmtuner.parallel.expert_parallel.torchao_dispatch import (
                    TorchAOTokenDispatcher,
                )

                assert isinstance(
                    module.routed_experts.token_dispatcher, TorchAOTokenDispatcher
                )
            experts = module.routed_experts.inner_experts
            assert experts.num_experts == 8 // EP
            for parameter in (experts.w1_EFD, experts.w3_EFD, experts.w2_EDF):
                assert isinstance(parameter, DTensor)
                assert parameter.shape[0] == 8 // EP

    reference, reference_norms, ref_params = _reference_trajectory(cfg)
    decoder_calls = 0
    replay_hooks = []
    if cfg.training.activation_checkpoint_mode in {"full", "selective"}:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        def count_decoder(module, args):
            nonlocal decoder_calls
            decoder_calls += 1

        for layer in trainer.model_parts[0].layers:
            assert isinstance(layer, CheckpointWrapper)
            replay_hooks.append(
                layer._checkpoint_wrapped_module.register_forward_pre_hook(
                    count_decoder
                )
            )

    # -- the trajectory, through the real Trainer -----------------------------
    data_iterator = trainer.data_iterator()
    losses = []
    norms = []
    for _ in range(STEPS):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        if CPU_OFFLOAD:
            _assert_cpu_offload(trainer)
        assert metrics is not None  # log_freq=1: every step reports
        # The reported loss is the global one: summed over every rank's token
        # shard (the loss mesh spans dp and tp) over the global token count.
        losses.append(metrics["loss"])
        norms.append(metrics["grad_norm"])
    if replay_hooks:
        for hook in replay_hooks:
            hook.remove()
        assert decoder_calls >= 2 * STEPS * cfg.model.num_hidden_layers

    max_loss_diff = 0.0
    for step, (got, want) in enumerate(zip(losses, reference, strict=True), 1):
        diff = abs(got - want)
        max_loss_diff = max(max_loss_diff, diff)
        if not math.isfinite(got) or not math.isfinite(want) or diff > TOL:
            failures.append(
                f"rank {rank}: step {step} loss {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )

    max_norm_diff = 0.0
    for step, (got, want) in enumerate(zip(norms, reference_norms, strict=True), 1):
        diff = abs(got - want)
        max_norm_diff = max(max_norm_diff, diff)
        if not math.isfinite(got) or not math.isfinite(want) or diff > TOL:
            failures.append(
                f"rank {rank}: step {step} grad norm {got:.6f} vs reference "
                f"{want:.6f} (diff {diff:.3e})"
            )

    # -- final parameters ------------------------------------------------------
    model = trainer.model_parts[0]
    # The TP layout of each sharded projection, keyed by its parameter name:
    # colwise shards rows, rowwise shards columns.
    tp_layouts = {}
    for path, mod in model.named_modules():
        if isinstance(mod, ColumnParallelLinear | ColwiseLinearNoGather):
            tp_layouts[f"{path}.weight"] = "row"
        elif isinstance(mod, RowParallelLinear):
            tp_layouts[f"{path}.weight"] = "col"
    if not tp_layouts:
        failures.append(f"rank {rank}: no TP-sharded projections -- vacuous")

    max_param_diff = 0.0
    checked_experts = 0
    ep_rank = parallel_dims.get_mesh("ep").get_local_rank() if EP > 1 else 0
    saw_fsdp_shard = False
    saw_replication = False
    for name, p in model.named_parameters():
        full = _full(p)
        reference_name = canonical_fqn(name).replace(
            ".router.gate.weight", ".gate.weight"
        )
        expert_marker = ".routed_experts.inner_experts."
        if expert_marker in reference_name:
            prefix, projection = reference_name.split(expert_marker)
            if projection in ("w1_EFD", "w3_EFD"):
                want = ref_params[f"{prefix}.experts.gate_up_proj"]
                want = want.chunk(2, dim=1)[projection == "w3_EFD"]
            else:
                assert projection == "w2_EDF", name
                want = ref_params[f"{prefix}.experts.down_proj"]
            want = want.chunk(EP, dim=0)[ep_rank]
            checked_experts += 1
        else:
            want = ref_params[reference_name]
        layout = tp_layouts.get(name)
        if layout == "row":
            n = want.shape[0]
            want = want[tp_rank * n // TP : (tp_rank + 1) * n // TP]
        elif layout == "col":
            k = want.shape[1]
            want = want[:, tp_rank * k // TP : (tp_rank + 1) * k // TP]
        elif full.numel() != want.numel():
            # A replicated parameter must arrive whole: FSDP redistributes it.
            failures.append(f"rank {rank}: {name} not replicated whole")
            continue
        # Non-vacuity, FSDP half: a sharded parameter is a DTensor whose local
        # shard is smaller than the global tensor.
        if isinstance(p, DTensor) and p.to_local().numel() < p.numel():
            saw_fsdp_shard = True
        if isinstance(p, DTensor) and DP_REPLICATE > 1:
            assert p.device_mesh["dp_replicate"].size() == DP_REPLICATE
            assert p.placements[0].is_replicate()
            saw_replication = True
        diff = (full - want).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
        if not math.isfinite(diff) or diff > TOL:
            failures.append(f"rank {rank}: param {name} diff {diff:.3e}")
    if DP_SHARD > 1 and not saw_fsdp_shard:
        failures.append(f"rank {rank}: no parameter was FSDP-sharded -- vacuous")
    if DP_SHARD == 1 and saw_fsdp_shard:
        failures.append(f"rank {rank}: replicated DP unexpectedly shards parameters")
    if DP_REPLICATE > 1 and not saw_replication:
        failures.append(f"rank {rank}: no replicated DTensor -- vacuous")
    if EP > 1:
        assert checked_experts == 3 * cfg.model.num_hidden_layers

    # Non-vacuity, TP half: the two tp ranks of a dp group must hold different
    # slices of the same weight.
    some_tp_name = next(iter(tp_layouts))
    shard = _full(dict(model.named_parameters())[some_tp_name]).contiguous()
    gathered = [torch.zeros_like(shard) for _ in range(TP)]
    dist.all_gather(gathered, shard, group=parallel_dims.get_mesh("tp").get_group())
    if torch.equal(gathered[0], gathered[1]):
        failures.append(f"rank {rank}: tp ranks hold identical shards (vacuous)")

    # Every rank must agree that every check passed, not just report its own.
    local_ok = torch.tensor([0.0 if not failures else 1.0], device=device())
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"tp={TP} ep={EP} dp_shard={DP_SHARD} "
            f"dp_replicate={DP_REPLICATE} steps={STEPS} "
            f"ac={cfg.training.activation_checkpoint_mode} "
            f"decoder_calls={decoder_calls} "
            f"dispatcher={DISPATCHER} "
            f"cpu_offload={CPU_OFFLOAD} "
            f"global_batch={GLOBAL_BATCH} seq={SEQ} tol={TOL:.0e}"
        )
        print(f"reference losses = {[f'{x:.6f}' for x in reference]}")
        print(f"tp+fsdp losses   = {[f'{x:.6f}' for x in losses]}")
        print(f"loss     max abs diff = {max_loss_diff:.3e}")
        print(f"gradnorm max abs diff = {max_norm_diff:.3e}")
        print(f"params   max abs diff = {max_param_diff:.3e}")
        print(f"failed ranks       = {int(local_ok.item())}")
        for f in failures:
            print(f"  FAIL {f}")
        if not failures:
            print("all checks passed")

    assert local_ok.item() == 0, "TP x FSDP equivalence check failed"
    trainer.checkpointer.close()
    trainer.metrics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
