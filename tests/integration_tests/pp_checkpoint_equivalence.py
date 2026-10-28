"""PP checkpoint round-trip: a resumed pp=2 run must match the uninterrupted one.

Run under torchrun with 2 ranks, in two phases that share a workdir:

    D=$(mktemp -d)
    torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_checkpoint_equivalence.py full "$D"
    torchrun --nproc_per_node=2 \\
        tests/integration_tests/pp_checkpoint_equivalence.py resume "$D"

Append ``--packed`` to both commands to use a checkpointable local JSONL corpus
with independently packed pipeline microbatches. On CPU that variant uses the
explicit dense-mask SDPA substitute from the real-corpus harness, rather than
the production CUDA flex kernel.

Set ``LLMTUNER_TEST_FROZEN_PP_STAGE=0`` or ``1`` in both phases to freeze
all parameters of that stage before partitioning (two or four ranks, PP2,
1F1B/Interleaved1F1B). Checks optimizer counts per trainable local part and
unchanged frozen weights. An interleaved rank can own both a frozen and a
trainable part.

Set ``LLMTUNER_TEST_LEGACY_LAYOUT=1`` for the full phase to save the historical
loader format without batch layout metadata, then resume normally to exercise
the manager's strict planner migration on both ranks.

Set ``LLMTUNER_TEST_RESUME_LR=0.003`` on the resume phase to verify that saved
optimizer learning rates override fresh construction values before scheduling.

Phase ``full`` trains a tiny offline qwen3 for N+M optimizer steps with
``pp=2`` (schedule 1F1B, 4 micro-batches per step) through the real
``Trainer``, saving a full DCP checkpoint -- model, FQN-keyed optimizer state,
and the trainer's step/token counters -- at step N, and records the
uninterrupted loss trajectory to the workdir.

Phase ``resume`` builds a brand-new Trainer against the same workdir, loads
the step-N checkpoint, and continues for M steps. The resumed losses must
match the uninterrupted run's last M losses BITWISE: the optimizer's Adam
moments enter the update, so a checkpoint that restored weights but not
moments (or moments keyed to the wrong stage's parameters) would diverge
visibly, while a correct round-trip reproduces the uninterrupted run's
arithmetic exactly.

The FQN keying is what pp > 1 exercises that a single-rank round-trip does
not: every stage's optimizer numbers its own parameters from 0, so positional
keys collide across ranks in one shared checkpoint (the flat FQN-keyed
optimizer state dict is what prevents that).

The synthetic random corpus is registered in the checkpoint's ``states`` under
``DATALOADER`` (its cursor is ``_position``, restored by directly generating
from the saved absolute batch index), so the resume phase reads exactly the batches the
uninterrupted run consumed at steps N+1..N+M with no manual fast-forward.

Final model, optimizer, scheduler, trainer counters and enabled EMA are compared
exactly. Set LLMTUNER_TEST_EMA=dynamic in both phases to exercise EMA aging.

Set LLMTUNER_TEST_TP_SIZE=2 and LLMTUNER_TEST_PP_SIZE=1 on two ranks
for tensor-parallel checkpoint ownership checks.

Set LLMTUNER_TEST_AMSGRAD=0 or 1 independently in each phase to verify
that saved optimizer options determine maximum-moment layout across PP/FSDP.

Everything runs in fp32 on CPU/gloo. The caller owns the workdir and its cleanup.
Set LLMTUNER_TEST_RELOCATED_HOOKS=1 with custom in both phases to
configure four attention projection units per ancestor hook and verify real
Trainer/AdamW/DCP/EMA resume with four communication contexts.
Set LLMTUNER_TEST_AC=full or selective in both phases to compose checkpoint
replay with those hooks; Interleaved1F1B exercises every local model part.
Set LLMTUNER_TEST_COMPILE_BACKEND=aot_eager in both phases to exercise
whole-model compilation; asserts OptimizedModule and nonzero compiled graphs.
Set LLMTUNER_TEST_PP_SIZE=1 and LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom
in both phases to check a real Trainer with FSDP2 over the same two ranks.
Set LLMTUNER_TEST_FORWARD_PREFETCH and LLMTUNER_TEST_BACKWARD_PREFETCH
in both phases to verify explicit FSDP prefetch windows across resume.
Set LLMTUNER_TEST_NUM_LAYERS=3 for two later decoder units with PP disabled.
For PP2 × FSDP2 use four ranks with LLMTUNER_TEST_WORLD_SIZE=4,
LLMTUNER_TEST_PP_SIZE=2 and LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom.
Set LLMTUNER_TEST_PP_SCHEDULE=Interleaved1F1B or ZBVZeroBubble in both
phases for two local stages per rank; those cases use six decoder layers.
The loss reference writer follows the schedule's last-stage ownership.
For packed PP2 × KV-CP2 on four ranks, additionally set LLMTUNER_TEST_CP_SIZE=2
and LLMTUNER_TEST_SEQ_LEN=256 in both phases and append --packed. CPU retains
CP mask predicates and collectives with explicit dense SDPA arithmetic.
Set LLMTUNER_TEST_ASYNC_MODE=async in both phases for CPU-compatible threaded
DCP upload; the full phase closes the Trainer before inspecting the checkpoint.
The pinned process stager requires its target-device environment.

Set LLMTUNER_TEST_KIMI=1 with LLMTUNER_TEST_PP_SIZE=1 on two ranks for
local dense Kimi KDA/custom-FSDP checkpoint recovery. This does not cover
Kimi MoE, hybrid attention, KCP, or TP. Add LLMTUNER_TEST_EMA=dynamic
to check the EMA state as well. LLMTUNER_TEST_KIMI_MOE=1 enables local
E4/top2 group-limited Kimi experts and one shared expert. Add
LLMTUNER_TEST_EP_SIZE=2 for native alltoall EP2/custom-FSDP recovery.
LLMTUNER_TEST_KIMI_LATENT=1 adds width32 latent projections and RMSNorm.

Set LLMTUNER_TEST_FIXED_ROUTER=1 with LLMTUNER_TEST_MOE=1 and PP_SIZE=1
to verify fixed routing, including EP1 local experts, through Trainer recovery.

Set LLMTUNER_TEST_MOE=1 and LLMTUNER_TEST_EP_SIZE=2 on four ranks
(LLMTUNER_TEST_WORLD_SIZE=4, LLMTUNER_TEST_PP_SIZE=1) for real qwen3_moe
EP2/FSDP4 recovery. Add LLMTUNER_TEST_TP_SIZE=2 for TP2/EP2 with
custom FSDP on the same four ranks (dense DP2, sparse efsdp2).
Add LLMTUNER_TEST_SHARED_LAYERS=1 to share Q projections
across the two MoE layers; all phases must use the same configuration.
Set LLMTUNER_TEST_EXCLUDED_EXPERTS=1 with custom FSDP/EP2/PP1 to exclude
each original fused gate/up projection and test ordinary expert gradient and
checkpoint state together. Both phases must use the same exclusion setting.

Set LLMTUNER_TEST_SHARED_ROOT_EXPERT=1 with the MoE/EP scenario to register
an alias to the converted local expert down projection at the wrapper root,
after EP conversion and before FSDP; check sparse parameter identity on recovery.

Set LLMTUNER_TEST_SHARED_ROOT=1 with LLMTUNER_TEST_PP_SIZE=1 to register
layer 0 Q projection weight at the wrapper root and check shared identity during
training and new-runtime checkpoint recovery.

Set LLMTUNER_TEST_SHARED_BOUNDARY=1 with LLMTUNER_TEST_PP_SIZE=1 to share
layer 0 input RMSNorm weight with the final RMSNorm, checking the default FSDP
boundary ownership through training and recovery.

Set LLMTUNER_TEST_SHARED_LAYERS=1 with LLMTUNER_TEST_PP_SIZE=1 to tie the
first two decoder Q projections before FSDP and check grouped-state recovery.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.components.metrics import get_metrics_rank
from llmtuner.config import (
    CheckpointConfig,
    DataloaderConfig,
    EMAConfig,
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.optimizer_config import ParamGroupConfig
from llmtuner.config.parallel_config import (
    ExpertParallelConfig,
    FSDPConfig,
    PipelineParallelConfig,
)
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.harness import device as harness_device

SPLIT_STEP = 2  # N: checkpoint taken after this many steps
EXTRA_STEPS = 2  # M: steps trained after the checkpoint (both runs)
TOTAL_STEPS = SPLIT_STEP + EXTRA_STEPS
MICROBATCHES = 4
GLOBAL_BATCH = 8
SEQ = int(os.environ.get("LLMTUNER_TEST_SEQ_LEN", "32"))
VOCAB = 128
SEED = 42

FULL_LOSSES_FILE = "losses_full.json"


def _equal_rank_state(
    actual: torch.Tensor, expected: torch.Tensor, *, label: str = "tensor"
) -> bool:
    """Compare every rank's shard without DTensor.equal's world collective.

    Pipeline stages own different numbers of tensors. DTensor.equal reduces
    over the default process group, so comparing their state directly can
    mismatch collectives across stages. All ranks validate their local shards;
    the final world reduction aggregates failures after all comparisons.
    """
    if isinstance(actual, DTensor) != isinstance(expected, DTensor):
        print(f"{label}: tensor types differ: {type(actual)} vs {type(expected)}")
        return False
    if isinstance(actual, DTensor):
        if (
            actual.shape != expected.shape
            or actual.placements != expected.placements
            or not torch.equal(actual.device_mesh.mesh, expected.device_mesh.mesh)
        ):
            print(
                f"{label}: DTensor layouts differ: "
                f"shape={actual.shape}/{expected.shape}, "
                f"placements={actual.placements}/{expected.placements}, "
                f"mesh={actual.device_mesh.mesh}/{expected.device_mesh.mesh}"
            )
            return False
        actual, expected = actual.to_local(), expected.to_local()
    actual, expected = actual.cpu(), expected.cpu()
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        print(
            f"{label}: local metadata differs: "
            f"dtype={actual.dtype}/{expected.dtype}, "
            f"shape={actual.shape}/{expected.shape}"
        )
        return False
    if torch.equal(actual, expected):
        return True
    # Diagnostics preserve exact equality as the acceptance criterion.
    actual_values, expected_values = (
        actual.to(torch.float64),
        expected.to(torch.float64),
    )
    difference = (actual_values - expected_values).abs()
    relative = difference / expected_values.abs().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    print(
        f"{label}: values differ: max_abs={difference.max().item():.9g}, "
        f"max_rel={relative.max().item():.9g}, "
        f"unequal={torch.count_nonzero(actual != expected).item()}/{actual.numel()}"
    )
    return False


def _cfg(workdir: str, *, packed: bool = False) -> LLMTunerConfig:
    data_root = os.path.join(workdir, f"data-rank-{os.environ.get('RANK', '0')}")
    schedule = os.environ.get("LLMTUNER_TEST_PP_SCHEDULE", "1F1B")
    moe = os.environ.get("LLMTUNER_TEST_MOE") == "1"
    kimi = os.environ.get("LLMTUNER_TEST_KIMI") == "1"
    quantile = os.environ.get("LLMTUNER_TEST_QUANTILE") == "1"
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path=(
                "deepseek_v3"
                if quantile
                else "kimi_linear"
                if kimi
                else "qwen3_moe"
                if moe
                else "qwen3"
            ),
            experts_implementation="eager" if moe else "native",
            norm_backend=os.environ.get("LLMTUNER_TEST_NORM_BACKEND", "native"),
            arch_overrides={
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "n_shared_experts": 1,
                "n_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 0,
                "q_lora_rank": 16,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 16,
            }
            if quantile
            else {
                **(
                    {"attn_res_block_size": int(os.environ["LLMTUNER_TEST_ATTN_RES"])}
                    if "LLMTUNER_TEST_ATTN_RES" in os.environ
                    else {}
                ),
                "linear_attn_config": {
                    "kda_layers": [1, 2],
                    "full_attn_layers": [],
                    "head_dim": 16,
                    "num_heads": 4,
                    "short_conv_kernel_size": 3,
                },
                **(
                    {
                        "routed_expert_hidden_size": (
                            32
                            if os.environ.get("LLMTUNER_TEST_KIMI_LATENT") == "1"
                            else None
                        ),
                        "latent_moe_use_norm": os.environ.get(
                            "LLMTUNER_TEST_KIMI_LATENT"
                        )
                        == "1",
                        "num_experts": 4,
                        "num_experts_per_token": 2,
                        "moe_intermediate_size": 48,
                        "num_shared_experts": 1,
                        "num_expert_group": 2,
                        "topk_group": 1,
                    }
                    if os.environ.get("LLMTUNER_TEST_KIMI_MOE") == "1"
                    else {}
                ),
            }
            if kimi
            else {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "norm_topk_prob": True,
            }
            if moe
            else {},
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=int(
                os.environ.get(
                    "LLMTUNER_TEST_NUM_LAYERS",
                    "2" if schedule in {"1F1B", "GPipe"} else "6",
                )
            ),
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=int(os.environ.get("LLMTUNER_TEST_TP_SIZE", "1")),
            expert_parallel_size=int(os.environ.get("LLMTUNER_TEST_EP_SIZE", "1")),
            ep_config=ExpertParallelConfig(
                moe_quantile_balancing=quantile,
                fixed_router=os.environ.get("LLMTUNER_TEST_FIXED_ROUTER") == "1",
                ep_token_dispatcher=os.environ.get(
                    "LLMTUNER_TEST_DISPATCHER", "alltoall"
                ),
            ),
            pipeline_parallel_size=int(os.environ.get("LLMTUNER_TEST_PP_SIZE", "2")),
            context_parallel_kv_allgather_size=int(
                os.environ.get("LLMTUNER_TEST_CP_SIZE", "1")
            ),
            data_parallel_shard_size=-1,
            fsdp_config=FSDPConfig(
                fsdp_implementation=os.environ.get(
                    "LLMTUNER_TEST_FSDP_IMPLEMENTATION", "native"
                ),
                fsdp_forward_prefetch_count=int(
                    os.environ.get("LLMTUNER_TEST_FORWARD_PREFETCH", "0")
                ),
                fsdp_backward_prefetch_count=int(
                    os.environ.get("LLMTUNER_TEST_BACKWARD_PREFETCH", "0")
                ),
            ),
            pp_config=PipelineParallelConfig(
                pipeline_parallel_schedule=schedule, num_pp_microbatches=MICROBATCHES
            ),
        ),
        optimizer=OptimizerConfig(
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 3e-4,
                        "weight_decay": 0.0,
                        "amsgrad": os.environ["LLMTUNER_TEST_AMSGRAD"] == "1",
                    },
                )
            ]
            if "LLMTUNER_TEST_AMSGRAD" in os.environ
            else [],
            learning_rate=3e-4,
            weight_decay=0.0,
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH,
            gradient_accumulation_steps=int(
                os.environ.get("LLMTUNER_TEST_ACCUMULATION", "1")
            ),
            max_seq_len=SEQ,
            steps=TOTAL_STEPS,
            seed=SEED,
            deterministic=True,
            compile="LLMTUNER_TEST_COMPILE_BACKEND" in os.environ,
            compile_config=CompileConfig(
                backend=os.environ.get("LLMTUNER_TEST_COMPILE_BACKEND", "inductor")
            ),
            activation_checkpoint_mode=os.environ.get("LLMTUNER_TEST_AC", "none"),
            dump_folder=workdir,
            checkpoint_config=CheckpointConfig(
                enable=True,
                folder="checkpoint",
                interval=SPLIT_STEP,
                async_mode=os.environ.get("LLMTUNER_TEST_ASYNC_MODE", "disabled"),
                # No retention policy to exercise: keep the purge thread out.
                keep_latest_k=0,
            ),
            metrics_config=MetricsConfig(log_freq=1),
            ema_config=EMAConfig(
                decay=None if os.environ.get("LLMTUNER_TEST_EMA") == "dynamic" else 0.5,
                half_life_fraction=0.5,
            )
            if os.environ.get("LLMTUNER_TEST_EMA") in {"1", "dynamic"}
            else None,
            dataloader_config=(
                DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path=os.path.join(data_root, "tokenizer"),
                    dataset_path=os.path.join(data_root, "rows.jsonl"),
                    shuffle=False,
                )
                if packed
                else DataloaderConfig()
            ),
        ),
    )


def _train_steps(trainer: Trainer, data_iterator, num_steps: int) -> list[float]:
    """Drive ``num_steps`` optimizer steps; return the last-stage losses."""
    losses = []
    frozen_stage = os.environ.get("LLMTUNER_TEST_FROZEN_PP_STAGE")
    frozen_parameters = {}
    if frozen_stage is not None:
        owns_frozen_stage = trainer.parallel_dims.get_mesh(
            "pp"
        ).get_local_rank() == int(frozen_stage)
        trainable_parts = sum(
            any(parameter.requires_grad for parameter in part.parameters())
            for part in trainer.model_parts
        )
        assert len(trainer.optimizer) == trainable_parts
        frozen_parameters = {
            name: parameter.detach().clone()
            for part in trainer.model_parts
            for name, parameter in part.named_parameters()
            if not parameter.requires_grad
        }
        assert bool(frozen_parameters) == owns_frozen_stage
        if trainer.world_size == 4:
            from torch.distributed.fsdp import FSDPModule

            assert trainer.parallel_dims.dp_shard == 2
            assert all(isinstance(part, FSDPModule) for part in trainer.model_parts)
            assert all(
                isinstance(parameter, DTensor)
                for part in trainer.model_parts
                for parameter in part.parameters()
            )
    for _ in range(num_steps):
        trainer.step += 1
        metrics = trainer.train_step(data_iterator)
        if "LLMTUNER_TEST_COMPILE_BACKEND" in os.environ:
            from torch._dynamo.eval_frame import OptimizedModule
            from torch._dynamo.utils import counters

            assert all(
                isinstance(part, OptimizedModule) for part in trainer.model_parts
            )
            assert counters["stats"]["unique_graphs"] > 0
        _assert_shared_layers(trainer)
        if frozen_stage is not None:
            assert trainer.lr_scheduler.state_dict() == {"last_epoch": trainer.step}
            for part in trainer.model_parts:
                for name, parameter in part.named_parameters():
                    if name in frozen_parameters:
                        assert parameter.grad is None
                        assert _equal_rank_state(parameter, frozen_parameters[name])
        assert metrics is not None  # log_freq=1: every step reports
        if not trainer.parallel_dims.pp_enabled or trainer.pp_has_last_stage:
            losses.append(metrics["loss"])
    return losses


def _assert_shared_layers(trainer: Trainer) -> None:
    if os.environ.get("LLMTUNER_TEST_FIXED_ROUTER") == "1":
        from llmtuner.models.common.moe.layer import MoE

        for layer in trainer.model_parts[0].layers:
            assert isinstance(layer.mlp, MoE)
            assert layer.mlp.router.fixed_router
            assert layer.mlp.routed_experts.inner_experts.w1_EFD.shape[0] == (
                8 // trainer.parallel_dims.ep
            )
    if os.environ.get("LLMTUNER_TEST_DISPATCHER") == "domino":
        from llmtuner.parallel.expert_parallel.domino_dispatch import (
            DominoRoutedExperts,
        )

        routed = [
            module
            for part in trainer.model_parts
            for module in part.modules()
            if isinstance(module, DominoRoutedExperts)
        ]
        assert routed, "requested Domino must reach the production implementation"
        assert all(module.error_group is not None for module in routed)
    if os.environ.get("LLMTUNER_TEST_NORM_BACKEND") == "aten":
        from llmtuner.models.hf.normalization import forward_aten_rms_norm

        norms = [
            module
            for part in trainer.model_parts
            for module in part.modules()
            if type(module).__name__ in ("Qwen3RMSNorm", "Qwen3MoeRMSNorm")
        ]
        assert norms, "aten normalization scenario did not find any Qwen norms"
        assert all(
            getattr(module.forward, "__func__", None) is forward_aten_rms_norm
            for module in norms
        ), "requested aten normalization backend was not applied"
    if os.environ.get("LLMTUNER_TEST_RELOCATED_HOOKS") == "1":
        from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
            FineGrainedFSDPState,
        )

        for model in trainer.model_parts:
            root = model._get_fsdp_state()
            assert isinstance(root, FineGrainedFSDPState)
            if root._is_root is not None:
                assert len(root.communication_contexts) == 4
            for layer in model.layers:
                states = [
                    getattr(layer.self_attn, name)._get_fsdp_state()
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
                if root._is_root is not None:
                    assert len({id(state._comm_ctx) for state in states}) == 4
                assert all(state.hook_module() is layer.self_attn for state in states)
    if os.environ.get("LLMTUNER_TEST_SHARED_ROOT_EXPERT") == "1":
        model = trainer.model_parts[0]
        first = list(model.layers)[0]
        assert model.external_expert is first.mlp.routed_experts.inner_experts.w2_EDF
    if (
        os.environ.get("LLMTUNER_TEST_MOE") == "1"
        and int(os.environ.get("LLMTUNER_TEST_EP_SIZE", "1")) > 1
    ):
        from llmtuner.models.common.moe.layer import MoE

        ep_size = int(os.environ["LLMTUNER_TEST_EP_SIZE"])
        dense_degree = trainer.world_size // trainer.parallel_dims.pp
        for layer in trainer.model_parts[0].layers:
            assert isinstance(layer.mlp, MoE)
            expert = layer.mlp.routed_experts.inner_experts.w1_EFD
            router = layer.mlp.router.gate.weight
            if os.environ.get("LLMTUNER_TEST_EXCLUDED_EXPERTS") == "1":
                assert not isinstance(expert, DTensor)
                grouped = layer.mlp.routed_experts.inner_experts
                assert not isinstance(grouped.w3_EFD, DTensor)
                assert isinstance(grouped.w2_EDF, DTensor)
                assert grouped.w2_EDF.device_mesh.size() == dense_degree // ep_size
            else:
                assert isinstance(expert, DTensor)
                assert expert.device_mesh.size() == dense_degree // ep_size
            assert isinstance(router, DTensor)
            # Physical TP parameters carry only the dedicated DP/CP mesh.
            assert router.device_mesh.size() == dense_degree // trainer.parallel_dims.tp
    if os.environ.get("LLMTUNER_TEST_SHARED_ROOT") == "1":
        model = trainer.model_parts[0]
        first = list(model.layers)[0]
        assert model.external_weight is first.self_attn.q_proj.weight
    if os.environ.get("LLMTUNER_TEST_SHARED_BOUNDARY") == "1":
        model = trainer.model_parts[0]
        first = list(model.layers)[0]
        assert first.input_layernorm.weight is model.norm.weight
        assert first._get_fsdp_state() is model.norm._get_fsdp_state()
    if os.environ.get("LLMTUNER_TEST_SHARED_LAYERS") != "1":
        return
    first, second = list(trainer.model_parts[0].layers)[:2]
    assert first.self_attn.q_proj.weight is second.self_attn.q_proj.weight
    assert first._get_fsdp_state() is second._get_fsdp_state()


def _runtime_state(trainer: Trainer) -> dict:
    rng = {}
    if os.environ.get("LLMTUNER_TEST_RNG_STATE") == "1":
        import pickle

        from llmtuner.components.checkpointer.random_state import RNGCheckpointState

        snapshot = RNGCheckpointState(trainer.device, topology=trainer.parallel_dims)
        rng = pickle.loads(snapshot.state_dict()[snapshot.key])
        # NumPy arrays need tensor comparison rather than ambiguous array ==.
        numpy_state = rng["numpy"]
        rng["numpy"] = (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            *numpy_state[2:],
        )
    return {
        **({"rng": rng} if rng else {}),
        **(
            {"dataloader": trainer.dataloader.state_dict()}
            if os.environ.get("LLMTUNER_TEST_LOADER_STATE") == "1"
            else {}
        ),
        "model": {
            canonical_fqn(key): value.detach().cpu().clone()
            for part in trainer.model_parts
            for key, value in part.state_dict().items()
        },
        "ntokens_seen": trainer.ntokens_seen,
        "trainer": trainer.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.lr_scheduler.state_dict(),
        "ema": trainer.ema.state_dict() if trainer.ema is not None else {},
    }


def _compare_loaded_state(actual, expected, failures: list[str], path: str) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not _equal_rank_state(actual, expected, label=path):
            failures.append(f"{path}: loaded tensor differs before any update")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            failures.append(f"{path}: loaded keys differ before any update")
        for key in actual.keys() & expected.keys():
            _compare_loaded_state(actual[key], expected[key], failures, f"{path}.{key}")
    elif isinstance(actual, tuple | list) and isinstance(expected, type(actual)):
        if len(actual) != len(expected):
            failures.append(f"{path}: loaded sequence length differs")
        for index, (got, want) in enumerate(zip(actual, expected, strict=False)):
            _compare_loaded_state(got, want, failures, f"{path}.{index}")
    elif type(actual) is not type(expected) or actual != expected:
        failures.append(f"{path}: loaded metadata differs before any update")


def _phase_full(workdir: str, *, packed: bool = False) -> None:
    trainer = Trainer(_cfg(workdir, packed=packed))
    assert trainer.world_size == int(os.environ.get("LLMTUNER_TEST_WORLD_SIZE", "2")), (
        f"unexpected world size: {trainer.world_size}"
    )

    data_iterator = trainer.data_iterator()
    losses = _train_steps(trainer, data_iterator, SPLIT_STEP)

    # The interval policy agrees this is a checkpointing step; a full state
    # (not the model-only last-step export) because ``last_step`` is False.
    if os.environ.get("LLMTUNER_TEST_LEGACY_LAYOUT") == "1":
        original_state_dict = trainer.dataloader.state_dict

        def legacy_state_dict():
            state = original_state_dict()
            state.pop("batch_layout", None)
            return state

        with patch.object(trainer.dataloader, "state_dict", legacy_state_dict):
            saved = trainer.checkpointer.save(SPLIT_STEP)
    else:
        saved = trainer.checkpointer.save(SPLIT_STEP)
    assert saved, "the split-step checkpoint was not written"
    torch.save(
        _runtime_state(trainer),
        os.path.join(workdir, f"split-state-rank-{trainer.rank}.pt"),
    )

    losses += _train_steps(trainer, data_iterator, EXTRA_STEPS)
    torch.save(
        _runtime_state(trainer),
        os.path.join(workdir, f"final-state-rank-{trainer.rank}.pt"),
    )

    # close must drain an async upload before its checkpoint is inspected or
    # a separate resume process is started. Final state has already been captured.
    trainer.close()
    checkpoint_dir = os.path.join(workdir, "checkpoint", f"step-{SPLIT_STEP}")
    if trainer.rank == get_metrics_rank(
        parallel_dims=trainer.parallel_dims,
        pp_schedule=trainer.cfg.parallel.pp_config.pipeline_parallel_schedule,
    ):
        with open(os.path.join(workdir, FULL_LOSSES_FILE), "w") as f:
            json.dump(losses, f)
        assert os.path.isfile(os.path.join(checkpoint_dir, ".metadata")), (
            f"no DCP checkpoint at {checkpoint_dir}"
        )

    print(
        f"[full] rank {trainer.rank}: trained {TOTAL_STEPS} steps, "
        f"saved step-{SPLIT_STEP} checkpoint, losses={losses}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


def _phase_resume(workdir: str, *, packed: bool = False) -> None:
    cfg = _cfg(workdir, packed=packed)
    if "LLMTUNER_TEST_RESUME_LR" in os.environ:
        cfg.optimizer.learning_rate = float(os.environ["LLMTUNER_TEST_RESUME_LR"])
    trainer = Trainer(cfg)
    assert trainer.world_size == int(os.environ.get("LLMTUNER_TEST_WORLD_SIZE", "2")), (
        f"unexpected world size: {trainer.world_size}"
    )
    failures: list[str] = []

    if os.environ.get("LLMTUNER_TEST_FAILED_UPDATE") == "1":
        assert trainer.checkpointer.load(-1)
        data_iterator = trainer.data_iterator()
        trainer.step += 1
        committed = False
        original_step = trainer.optimizer.step

        def fail_after_update(*args, **kwargs):
            nonlocal committed
            original_step(*args, **kwargs)
            committed = True
            raise RuntimeError("injected failure after optimizer commit")

        with patch.object(trainer.optimizer, "step", fail_after_update):
            try:
                trainer.train_step(data_iterator)
            except RuntimeError as error:
                assert str(error) == "injected failure after optimizer commit"
            else:
                raise AssertionError("injected optimizer failure was not raised")
        assert committed, "fault must occur after a real parameter update"
        trainer.close()
        dist.barrier()
        # Discard the mutated runtime; normal checkpoint load must recover
        # weights, moments, data position, counters, EMA and random streams.
        trainer = Trainer(_cfg(workdir, packed=packed))
        print(f"rank {trainer.rank}: discarded failed runtime after optimizer commit")

    if os.environ.get("LLMTUNER_TEST_FAILED_EXPERT") == "1":
        from llmtuner.parallel.expert_parallel.domino_dispatch import (
            DominoRoutedExperts,
        )

        assert trainer.checkpointer.load(-1)
        routed = next(
            module
            for part in trainer.model_parts
            for module in part.modules()
            if isinstance(module, DominoRoutedExperts)
        )
        original_run = routed._run_experts
        original_linear = torch.nn.functional.linear
        injected = False
        expert_calls = 0
        fail_call = int(os.environ.get("LLMTUNER_TEST_FAILED_EXPERT_CALL", "1"))
        assert fail_call >= 1

        def fail_linear(*args, **kwargs):
            nonlocal injected
            if trainer.rank == 0 and not injected and expert_calls == fail_call:
                injected = True
                raise ValueError("injected Trainer local expert failure")
            return original_linear(*args, **kwargs)

        def run_with_fault(*args, **kwargs):
            nonlocal expert_calls
            expert_calls += 1
            if fail_call > 2 and expert_calls == fail_call:
                assert any(
                    parameter.grad is not None
                    for part in trainer.model_parts
                    for parameter in part.parameters()
                ), "failure must follow an accumulated backward"
            assert kwargs["coordinate_experts"]
            with patch("torch.nn.functional.linear", fail_linear):
                return original_run(*args, **kwargs)

        trainer.step += 1
        with patch.object(routed, "_run_experts", run_with_fault):
            try:
                trainer.train_step(trainer.data_iterator())
            except (ValueError, RuntimeError) as error:
                if trainer.rank == 0:
                    assert isinstance(error, ValueError)
                    assert "injected Trainer local expert failure" in str(error)
                else:
                    assert "Domino expert computation failed on another rank" in str(
                        error
                    )
            else:
                raise AssertionError("single-rank expert failure was not propagated")
        trainer.close()
        dist.barrier()
        trainer = Trainer(_cfg(workdir, packed=packed))
        print(
            f"rank {trainer.rank}: discarded runtime after single-rank expert failure"
        )

    loaded = trainer.checkpointer.load(-1)
    if os.environ.get("LLMTUNER_TEST_EMPTY_SCRATCH") == "1":
        from llmtuner.models.common.aux_loss import AuxLoss
        from llmtuner.models.common.moe.layer import MoE
        from llmtuner.models.common.moe.routing import QuantileBalancedTopKRouter

        moe_count = aux_count = quantile_count = 0
        for part in trainer.model_parts:
            for module in part.modules():
                if isinstance(module, MoE):
                    moe_count += 1
                    assert not module.tokens_per_expert_E.any()
                    if isinstance(module.router, QuantileBalancedTopKRouter):
                        quantile_count += 1
                        histogram = (
                            module.router.quantile_balancer.required_bias_histogram_EB
                        )
                        assert not histogram.any()
                if isinstance(module, AuxLoss):
                    aux_count += 1
                    assert not module.instance_acc.any()
                    assert module._step_metric is None
                    assert module._runtime_denominator is None
        assert moe_count > 0
        if os.environ.get("LLMTUNER_TEST_QUANTILE") == "1":
            assert quantile_count == moe_count
        else:
            assert aux_count > 0, "scratch checks must reach real aux losses"
        print(
            f"rank {trainer.rank}: restored scratch empty "
            f"(moe={moe_count}, aux={aux_count})"
        )
    _assert_shared_layers(trainer)
    if not loaded:
        failures.append(f"rank {trainer.rank}: no checkpoint was loaded")
    if trainer.step != SPLIT_STEP:
        failures.append(
            f"rank {trainer.rank}: resumed at step {trainer.step}, "
            f"expected {SPLIT_STEP}"
        )

    split_state_path = os.path.join(workdir, f"split-state-rank-{trainer.rank}.pt")
    # Older harness artifacts lack this independent save-point snapshot.
    if os.path.isfile(split_state_path):
        split_state = torch.load(
            split_state_path, map_location="cpu", weights_only=True
        )
        # A deliberate LR override tests a different contract from exact restore.
        if "LLMTUNER_TEST_RESUME_LR" not in os.environ:
            _compare_loaded_state(
                _runtime_state(trainer),
                split_state,
                failures,
                f"rank {trainer.rank}.split",
            )
            print(f"rank {trainer.rank}: pre-update state failures={len(failures)}")

    # Non-vacuity: the optimizer must have come back warm -- a cold Adam (no
    # or zeroed moments) is exactly the failure a loss match would then catch,
    # so assert it directly for a clearer message. The container's state dict is
    # flat and FQN-keyed, so the moments are the ``state.<fqn>.exp_avg`` entries.
    optim_state = trainer.optimizer.state_dict()
    moments = {
        key: value
        for key, value in optim_state.items()
        if key.startswith("state.") and key.endswith("exp_avg")
    }
    frozen_stage = os.environ.get("LLMTUNER_TEST_FROZEN_PP_STAGE")
    frozen_local = frozen_stage is not None and all(
        not parameter.requires_grad
        for part in trainer.model_parts
        for parameter in part.parameters()
    )
    if frozen_local:
        assert len(trainer.optimizer) == 0
        assert optim_state == {}
        assert not moments
    elif not moments:
        failures.append(f"rank {trainer.rank}: optimizer state is empty")
    elif not any(float(moment.abs().sum()) > 0 for moment in moments.values()):
        failures.append(f"rank {trainer.rank}: optimizer moments are all zero")

    # The loader's read position came back with the checkpoint (DATALOADER is
    # registered in ``states``), so step N+1 reads what the uninterrupted run
    # read -- no manual fast-forward.
    data_iterator = trainer.data_iterator()

    losses = _train_steps(trainer, data_iterator, EXTRA_STEPS)
    final_state = torch.load(
        os.path.join(workdir, f"final-state-rank-{trainer.rank}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    final_runtime = _runtime_state(trainer)
    for key in ("rng", "dataloader"):
        if key in final_runtime or key in final_state:
            _compare_loaded_state(
                {key: final_runtime[key]} if key in final_runtime else {},
                {key: final_state[key]} if key in final_state else {},
                failures,
                f"rank {trainer.rank}.final",
            )
    actual_parameters = {
        canonical_fqn(key): value.detach().cpu()
        for part in trainer.model_parts
        for key, value in part.state_dict().items()
    }
    if actual_parameters.keys() != final_state["model"].keys():
        failures.append(f"rank {trainer.rank}: final model keys differ")
    else:
        for key, actual in actual_parameters.items():
            if not _equal_rank_state(actual, final_state["model"][key], label=key):
                failures.append(f"rank {trainer.rank}: final parameter {key} differs")
    if trainer.ntokens_seen != final_state["ntokens_seen"]:
        failures.append(f"rank {trainer.rank}: restored token counter differs")
    if trainer.state_dict() != final_state["trainer"]:
        failures.append(f"rank {trainer.rank}: final trainer state differs")
    if trainer.lr_scheduler.state_dict() != final_state["scheduler"]:
        failures.append(f"rank {trainer.rank}: final scheduler state differs")
    actual_optimizer = trainer.optimizer.state_dict()
    expected_optimizer = final_state["optimizer"]
    if actual_optimizer.keys() != expected_optimizer.keys():
        failures.append(f"rank {trainer.rank}: final optimizer keys differ")
    else:
        for key, actual in actual_optimizer.items():
            expected = expected_optimizer[key]
            matches = (
                _equal_rank_state(actual, expected, label=key)
                if isinstance(actual, torch.Tensor)
                else actual == expected
            )
            if not matches:
                failures.append(f"rank {trainer.rank}: optimizer {key} differs")
    if trainer.ema is not None:
        actual_ema = trainer.ema.state_dict()
        expected_ema = final_state["ema"]
        if actual_ema.keys() != expected_ema.keys():
            failures.append(f"rank {trainer.rank}: final EMA keys differ")
        else:
            for key, actual in actual_ema.items():
                expected = expected_ema[key]
                if isinstance(actual, torch.Tensor):
                    if not _equal_rank_state(actual, expected, label=key):
                        failures.append(f"rank {trainer.rank}: EMA {key} differs")
                elif actual != expected:
                    failures.append(f"rank {trainer.rank}: EMA metadata {key} differs")

    with open(os.path.join(workdir, FULL_LOSSES_FILE)) as f:
        full_losses = json.load(f)
    uninterrupted_tail = full_losses[SPLIT_STEP:]

    # The real losses live on the last-stage rank; collect them everywhere.
    gathered = [losses]
    dist.broadcast_object_list(
        gathered,
        src=get_metrics_rank(
            parallel_dims=trainer.parallel_dims,
            pp_schedule=cfg.parallel.pp_config.pipeline_parallel_schedule,
        ),
    )
    resumed_losses = gathered[0]

    max_diff = 0.0
    for offset, (got, want) in enumerate(
        zip(resumed_losses, uninterrupted_tail, strict=True), 1
    ):
        diff = abs(got - want)
        max_diff = max(max_diff, diff)
        if got != want:  # bitwise: a correct round-trip reproduces the arithmetic
            failures.append(
                f"rank {trainer.rank}: step {SPLIT_STEP + offset} loss {got!r} "
                f"vs uninterrupted {want!r} (diff {diff:.3e})"
            )

    # Crosses an HCCL group on the accelerator: it has to live on that device.
    local_ok = torch.tensor([0.0 if not failures else 1.0], device=harness_device())
    dist.all_reduce(local_ok, op=dist.ReduceOp.MAX)
    rank_failures = [None] * trainer.world_size
    dist.all_gather_object(rank_failures, failures)

    if trainer.rank == 0:
        print(
            f"pp={cfg.parallel.pp_size} checkpoint round-trip: split={SPLIT_STEP} "
            f"extra={EXTRA_STEPS} fsdp={cfg.parallel.fsdp_config.fsdp_implementation}"
        )
        print(f"uninterrupted losses = {[f'{x:.6f}' for x in full_losses]}")
        print(f"resumed losses       = {[f'{x:.6f}' for x in resumed_losses]}")
        print(f"max abs diff         = {max_diff:.3e}")
        for errors in rank_failures:
            for error in errors:
                print(f"  FAIL {error}")
        if local_ok.item() == 0:
            print("all checks passed")

    dist.barrier()
    trainer.close()
    dist.destroy_process_group()

    assert local_ok.item() == 0, "PP checkpoint round-trip check failed"


def _run() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    workdir = sys.argv[2] if len(sys.argv) > 2 else None
    if phase not in ("full", "resume") or not workdir:
        raise SystemExit(
            "usage: torchrun --nproc_per_node=2 "
            "tests/integration_tests/pp_checkpoint_equivalence.py "
            "{full|resume} WORKDIR [--packed]"
        )
    packed = sys.argv[3:] == ["--packed"]
    if sys.argv[3:] and not packed:
        raise SystemExit("only --packed is accepted after WORKDIR")
    run_phase = _phase_full if phase == "full" else _phase_resume
    os.makedirs(workdir, exist_ok=True)
    if packed:
        from llmtuner.accelerator.device import device_type
        from llmtuner.models.hf.model import HFTransformerModel
        from tests.integration_tests.pp_real_corpus_equivalence import (
            _cpu_packed_attention,
            _write_corpus,
        )

        if phase == "full":
            data_root = os.path.join(
                workdir, f"data-rank-{os.environ.get('RANK', '0')}"
            )
            os.makedirs(data_root, exist_ok=True)
            _write_corpus(data_root)
        if (
            device_type == "cpu"
            and int(os.environ.get("LLMTUNER_TEST_CP_SIZE", "1")) > 1
        ):
            from tests.integration_tests.pp_cp_equivalence import cpu_cp_attention

            with cpu_cp_attention():
                run_phase(workdir, packed=True)
            return
        if device_type == "cpu":
            with patch.object(
                HFTransformerModel, "_apply_attention", _cpu_packed_attention
            ):
                run_phase(workdir, packed=True)
            return
    run_phase(workdir, packed=packed)


def main() -> None:
    if os.environ.get("LLMTUNER_TEST_EXCLUDED_EXPERTS") == "1":
        if (
            os.environ.get("LLMTUNER_TEST_FSDP_IMPLEMENTATION") != "custom"
            or os.environ.get("LLMTUNER_TEST_EP_SIZE") != "2"
            or os.environ.get("LLMTUNER_TEST_PP_SIZE") != "1"
            or os.environ.get("LLMTUNER_TEST_MOE") != "1"
        ):
            raise ValueError("Excluded experts require custom FSDP/EP2/PP1/MoE")
        from llmtuner.trainer.runtime import assembly

        apply_parallelism = assembly._apply_parallelism

        def exclude_experts(trainer, model, **kwargs):
            names = {module: name for name, module in model.named_modules()}
            trainer.cfg.parallel.fsdp_config.fsdp_apply_modules = {
                names[layer.mlp.experts]: {
                    "ignored_params": {layer.mlp.experts.gate_up_proj},
                    "hook_module": layer.mlp.experts,
                }
                for layer in model.layers
            }
            apply_parallelism(trainer, model, **kwargs)

        with patch.object(assembly, "_apply_parallelism", exclude_experts):
            _run()
        return
    frozen_stage = os.environ.get("LLMTUNER_TEST_FROZEN_PP_STAGE")
    if frozen_stage is not None:
        if (
            frozen_stage not in {"0", "1"}
            or int(os.environ.get("WORLD_SIZE", "2")) not in {2, 4}
            or os.environ.get("LLMTUNER_TEST_PP_SIZE", "2") != "2"
            or os.environ.get("LLMTUNER_TEST_PP_SCHEDULE", "1F1B")
            not in {"1F1B", "Interleaved1F1B"}
        ):
            raise ValueError(
                "Frozen stage scenario requires 2/4 ranks, PP2, 1F1B/Interleaved1F1B"
            )
        from llmtuner.parallel.pipeline_parallel import pipeline_parallel

        split = pipeline_parallel.split_model_into_stages

        def split_frozen(model, mesh, schedule, device, assignments, **kwargs):
            for name in assignments[int(frozen_stage)]:
                model.get_submodule(name).requires_grad_(False)
            return split(model, mesh, schedule, device, assignments, **kwargs)

        with patch.object(pipeline_parallel, "split_model_into_stages", split_frozen):
            _run()
        return
    if os.environ.get("LLMTUNER_TEST_RELOCATED_HOOKS") == "1":
        if os.environ.get("LLMTUNER_TEST_FSDP_IMPLEMENTATION") != "custom":
            raise ValueError("Relocated-hook checkpoint scenario requires custom FSDP")
        from llmtuner.trainer.runtime import assembly

        apply_parallelism = assembly._apply_parallelism

        def configure_hooks(trainer, model, **kwargs):
            # Hook targets must be actual modules, available after model build.
            names = {module: name for name, module in model.named_modules()}
            trainer.cfg.parallel.fsdp_config.fsdp_apply_modules = {
                names[getattr(layer.self_attn, name)]: {"hook_module": layer.self_attn}
                for layer in model.layers
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
            apply_parallelism(trainer, model, **kwargs)

        with patch.object(assembly, "_apply_parallelism", configure_hooks):
            _run()
        return
    shared_layers = os.environ.get("LLMTUNER_TEST_SHARED_LAYERS") == "1"
    shared_boundary = os.environ.get("LLMTUNER_TEST_SHARED_BOUNDARY") == "1"
    shared_root = os.environ.get("LLMTUNER_TEST_SHARED_ROOT") == "1"
    shared_root_expert = os.environ.get("LLMTUNER_TEST_SHARED_ROOT_EXPERT") == "1"
    if not (shared_layers or shared_boundary or shared_root or shared_root_expert):
        _run()
        return
    if os.environ.get("LLMTUNER_TEST_PP_SIZE") != "1":
        raise ValueError("Shared-parameter checkpoint scenarios require PP1")
    from llmtuner.models.hf.model import HFTransformerModel

    original_init = HFTransformerModel.__init__

    def initialize_shared(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if shared_layers:
            first, second = self.layers[:2]
            second.self_attn.q_proj.weight = first.self_attn.q_proj.weight
        if shared_boundary:
            self.layers[0].input_layernorm.weight = self.norm.weight
        if shared_root:
            self.register_parameter(
                "external_weight", self.layers[0].self_attn.q_proj.weight
            )

    from llmtuner.parallel.expert_parallel.expert_parallel import apply_ep

    def apply_ep_with_alias(model, **kwargs):
        model = apply_ep(model, **kwargs)
        if shared_root_expert:
            if os.environ.get("LLMTUNER_TEST_MOE") != "1" or not isinstance(
                model.layers[0].mlp, MoE
            ):
                raise ValueError("Root expert checkpoint scenario requires MoE/EP")
            expert = model.layers[0].mlp.routed_experts.inner_experts.w2_EDF
            model.register_parameter("external_expert", expert)
        return model

    from llmtuner.models.common.moe.layer import MoE

    with (
        patch.object(HFTransformerModel, "__init__", initialize_shared),
        patch("llmtuner.parallel.parallelize.apply_ep", apply_ep_with_alias),
    ):
        _run()


if __name__ == "__main__":
    main()
