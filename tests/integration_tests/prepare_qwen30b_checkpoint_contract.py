"""Prepare the full48 Qwen3-30B TP2/CP2/EP4 contract without opening DCP.

Run with the SAME Transformers/source versions as the target training recipe.
Only meta model tensors and CPU dataloader state are constructed; no NPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from torch.distributed.checkpoint._nested_dict import flatten_state_dict

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict
from llmtuner.datasets.loader_builder import build_dataloader
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_targets
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.audit_dcp_checkpoint import AuditContract, TensorSpec

RECIPE_ENV = {
    "DATA_PARALLEL_REPLICATE_SIZE": "1",
    "DATA_PARALLEL_SHARD_SIZE": "2",
    "TENSOR_PARALLEL_SIZE": "2",
    "CONTEXT_PARALLEL_SIZE": "2",
    "EXPERT_PARALLEL_SIZE": "4",
    "PIPELINE_PARALLEL_SIZE": "1",
    "NUM_PP_MICROBATCHES": "1",
    "EP_TOKEN_DISPATCHER": "alltoall",
    "COMPILE_BACKEND": "eager",
    "FSDP_CPU_OFFLOAD": "0",
    "OPTIMIZER_STATE_RESIDENCY": "device",
    "ACTIVATION_SWAP": "0",
    "ACTIVATION_CHECKPOINT_MODE": "full",
    "STEPS": "100",
    "MAX_SEQ_LEN": "4096",
    "GLOBAL_BATCH_SIZE": "8",
    "DETERMINISTIC": "1",
    "GLOBAL_CONTEXT_ROWS": "1",
    "MAX_NUM_DOCUMENTS": "1",
    "SAVE_CHECKPOINT": "1",
    "CHECKPOINT_INTERVAL": "50",
    "CHECKPOINT_LOAD_STEP": "-1",
    "CHECKPOINT_ASYNC_MODE": "disabled",
    "SHARDED_TENSOR_CHECKPOINT": "1",
    "SHARDED_EXPERT_CHECKPOINT": "1",
    "ENABLE_TENSORBOARD": "0",
}


def project_joint_global_schema(model, parallel):
    """Global projection of production's disjoint TP/EP chunk transform.

    resolve_tp_targets explicitly defers routed experts to EP. Dense TP changes
    slices, not FQNs; swap_hf_moe_blocks performs the actual router/expert rename.
    With ep_group=None it retains all experts, exactly the logical EP extent.
    No collective, local shard allocation, checkpoint read, or NPU is needed.
    """
    targets, moe = resolve_tp_targets(model, cfg=parallel)
    if moe or not targets:
        raise ValueError("Expected dense TP targets and experts deferred to EP")
    ownership = {}
    for path, module, spec in targets:
        axis = 0 if spec.kind == "colwise" else 1
        if module.weight.shape[axis] % 2:
            raise ValueError(f"Nondivisible TP target: {path}")
        ownership[canonical_fqn(f"{path}.weight")] = {"axis": axis, "degree": 2}
    swapped = swap_hf_moe_blocks(model, tp_enabled=True, token_dispatcher="alltoall")
    for name, parameter in model.named_parameters():
        if ".routed_experts.inner_experts." in name:
            key = canonical_fqn(name)
            if key in ownership or parameter.shape[0] % 4:
                raise ValueError(f"Invalid disjoint expert ownership: {name}")
            ownership[key] = {"axis": 0, "degree": 4}
    return swapped, ownership


def prepare_contract(model_path: str, dataset_path: str):
    env = {"LLMTUNER_" + k: v for k, v in RECIPE_ENV.items()}
    env.update(LLMTUNER_QWEN3_30B_PATH=model_path, LLMTUNER_DATASET_PATH=dataset_path)
    with patch.dict(os.environ, env):
        cfg = qwen3_30b_npu_config()
    if cfg.training.ema_config is not None:
        raise ValueError("This pinned acceptance recipe explicitly has EMA disabled")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = HFTransformerModel(build_model_config_for(cfg))
    finally:
        torch.set_default_dtype(previous)
    if len(model.layers) != 48:
        raise ValueError("Acceptance requires all 48 layers")
    with torch.device("meta"):
        swapped, ownership = project_joint_global_schema(model, cfg.parallel)
    if swapped != 48:
        raise ValueError(f"Expected 48 converted MoE layers, got {swapped}")
    state = model.state_dict()
    expected = {
        canonical_fqn(k): TensorSpec(tuple(v.shape), str(v.dtype))
        for k, v in state.items()
    }
    named = [
        (canonical_fqn(k), p) for k, p in model.named_parameters() if p.requires_grad
    ]
    # The recipe uses default fused AdamW, no decay/warmup. Construct its real
    # state serializer on meta; do NOT execute an optimizer update.
    optimizer = torch.optim.AdamW(
        [{"params": [p for _, p in named], "param_names": [k for k, _ in named]}],
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
        fused=torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_fused_adamw_", "PrivateUse1"
        ),
        foreach=False,
    )
    torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for _, parameter in named:
        optimizer.state[parameter] = {
            "step": torch.empty((), device="meta", dtype=torch.float32),
            "exp_avg": torch.empty_like(parameter),
            "exp_avg_sq": torch.empty_like(parameter),
        }
    optim = get_flat_optim_state_dict(optimizer)
    loader = build_dataloader(
        cfg, dp_rank=0, dp_world_size=2, num_tokens_per_batch=4 * 4096
    )
    try:
        data = loader.state_dict()
        cursor, _ = flatten_state_dict(data["dp_rank_0"])
        print(
            "Independent fresh-loader cursor types:",
            {k: type(v).__name__ for k, v in cursor.items()},
            flush=True,
        )
    finally:
        loader.close()
    contract = AuditContract(
        world_size=8,
        step=100,
        ntokens_seen=100 * (4 * 4096) // (2 * 2),
        device_type="npu",
        parallel_degrees=dict(
            dp_replicate=1, dp_shard=2, cp=2, tp=2, pp=1, ep=4, cp_kv=2, cp_ulysses=1
        ),
        expected_fqns=expected,
        optimizer_tensors={
            k: TensorSpec(tuple(v.shape), str(v.dtype))
            for k, v in optim.items()
            if isinstance(v, torch.Tensor)
        },
        optimizer_values={
            k: v for k, v in optim.items() if not isinstance(v, torch.Tensor)
        },
        optimizer_kinds={k: "AdamW" for k, _ in named},
        optimizer_steps={k: 100 for k, _ in named},
        data_layout=data["batch_layout"],
        cursor_schema={k: type(v).__name__ for k, v in cursor.items()},
        cursor_rank_values={
            f"dp_rank_{rank}": {"geometry": (cfg.global_batch_size, rank, 2)}
            for rank in range(2)
        },
        ema=None,
        ema_values={},
        model_prefix="",
    )
    contract.validate()
    return contract, ownership


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, ownership = prepare_contract(args.model, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(asdict(contract), stream, indent=2)
    provenance = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_config_sha256": hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest(),
        "dataset": args.data,
        "recipe": "examples/qwen3-moe/train_qwen3_30b_npu.py",
        "recipe_sha256": hashlib.sha256(
            Path("examples/qwen3-moe/train_qwen3_30b_npu.py").read_bytes()
        ).hexdigest(),
        "source": "independent recipe/meta model/new loader; no checkpoint opened",
        "model_tensors": len(contract.expected_fqns),
        "optimizer_tensors": len(contract.optimizer_tensors),
        "ema": False,
        "recipe_environment": RECIPE_ENV,
        "chunk_ownership": ownership,
        "projection": (
            "production resolve_tp_targets + full-expert swap_hf_moe_blocks; "
            "joint chunk transforms retain global FQNs/shapes"
        ),
        "source_hashes": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                __file__,
                "llmtuner/parallel/tensor_parallel/tensor_parallel.py",
                "llmtuner/parallel/tensor_parallel/checkpointing.py",
                "llmtuner/parallel/expert_parallel/model_replacement.py",
                "llmtuner/parallel/expert_parallel/block_conversion.py",
                "llmtuner/parallel/expert_parallel/checkpointing.py",
                "llmtuner/components/optimizer/state_dict.py",
                "llmtuner/trainer/runtime/batch.py",
                "llmtuner/datasets/loader_builder.py",
            )
        },
    }
    with args.output.with_suffix(".provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
