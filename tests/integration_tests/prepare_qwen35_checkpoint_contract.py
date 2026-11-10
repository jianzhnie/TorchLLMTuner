"""Prepare the full DP8 Qwen3.5 step100 audit contract without opening DCP.

Run with the SAME Transformers/source versions as the target training recipe.
Only meta model tensors and CPU dataloader state are constructed; no NPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from torch.distributed.checkpoint._nested_dict import flatten_state_dict

from examples.qwen3_5.train import build_argument_parser, build_config
from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict
from llmtuner.datasets.loader_builder import build_dataloader
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.utils.module_names import canonical_fqn
from tests.integration_tests.audit_dcp_checkpoint import AuditContract, TensorSpec


def prepare_contract(model_path: str, dataset_path: str) -> AuditContract:
    args = build_argument_parser().parse_args(
        [
            "--model",
            model_path,
            "--data",
            dataset_path,
            "--output",
            "unused-meta-only",
            "--steps",
            "100",
            "--batch",
            "8",
            "--sequence-length",
            "2048",
            "--dp-shard",
            "8",
            "--checkpoint-interval",
            "50",
        ]
    )
    cfg = build_config(args)
    if cfg.training.ema_config is not None:
        raise ValueError("This pinned acceptance recipe explicitly has EMA disabled")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = HFTransformerModel(build_model_config_for(cfg))
    finally:
        torch.set_default_dtype(previous)
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
        cfg, dp_rank=0, dp_world_size=8, num_tokens_per_batch=2048
    )
    try:
        data = loader.state_dict()
        cursor, _ = flatten_state_dict(data["dp_rank_0"])
    finally:
        loader.close()
    contract = AuditContract(
        world_size=8,
        step=100,
        ntokens_seen=100 * 2048,
        device_type="npu",
        parallel_degrees=dict(
            dp_replicate=1, dp_shard=8, cp=1, tp=1, pp=1, ep=1, cp_kv=1, cp_ulysses=1
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
        ema=None,
        ema_values={},
        model_prefix="",
    )
    contract.validate()
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = prepare_contract(args.model, args.data)
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
        "recipe": "examples/qwen3_5/train.py",
        "recipe_sha256": hashlib.sha256(
            Path("examples/qwen3_5/train.py").read_bytes()
        ).hexdigest(),
        "source": "independent recipe/meta model/new loader; no checkpoint opened",
        "model_tensors": len(contract.expected_fqns),
        "optimizer_tensors": len(contract.optimizer_tensors),
        "ema": False,
    }
    with args.output.with_suffix(".provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
