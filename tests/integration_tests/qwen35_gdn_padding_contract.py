"""CPU audit and staged plan for actual Qwen3.5 GDN padding/full-AC acceptance.

Does not execute recurrence, NPU, checkpointing or numerical model acceptance.
A successful audit means the documented restrictions were reproduced, not that
padding is supported. No production/reference source or guard is changed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    apply_mask_to_padding_states,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3_5.gdn_numerics import configure_gdn_numerics
from llmtuner.models.qwen.qwen3_5.linear_attention import validate_gdn_inputs


def source(value):
    path = Path(inspect.getsourcefile(value))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": inspect.getsource(value),
    }


def rejected(call):
    try:
        call()
    except UnsupportedCombinationError as error:
        assert "mask" in str(error).lower(), str(error)
        return {"rejected": True, "error": str(error)}
    raise AssertionError("Production unexpectedly accepted unverified attention_mask")


def audit(args):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert config.hidden_size == 2048 and config.num_hidden_layers == 40
    config.gdn_numerical_policy = "split_qkv_fp32_norm"
    config.qkv_projection = "separate"
    config.norm_backend = "native"
    with torch.device("meta"):
        module = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(torch.bfloat16)
    holder = torch.nn.Module()
    holder.config = config
    holder.add_module("linear_attn", module)
    configure_gdn_numerics(holder)
    observations = {}
    for batch in (1, 2):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(batch, 128, 2048, generator=generator).bfloat16()
        mask = torch.arange(128)[None, :].expand(batch, -1) < 93
        masked = apply_mask_to_padding_states(x, mask)
        expected = x if batch == 1 else x * mask[..., None]
        assert torch.equal(masked, expected)
        assert torch.equal(masked[:, :93], x[:, :93])
        observations[str(batch)] = {
            "native_hf_padding_is_zero": not bool(masked[:, 93:].count_nonzero()),
            "native_hf_unchanged": torch.equal(masked, x),
            "mask_shape": list(mask.shape),
            "positions": "Each row 0..127, no reset at padding start",
            "policy_guard": rejected(
                lambda x=x, mask=mask: module(x, attention_mask=mask)
            ),
            "npu_triton_guard": rejected(
                lambda x=x, mask=mask: validate_gdn_inputs(
                    module, (x,), {"attention_mask": mask}
                )
            ),
        }
    path = args.reference_root / "torchtitan/models/qwen3_5/model.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GatedDeltaNet"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    argument = next(a for a in forward.args.args if a.arg == "attention_masks")
    annotation = ast.unparse(argument.annotation)
    assert "VarlenMetadata" in annotation
    code = ast.get_source_segment(path.read_text(), forward)
    assert "attention_masks.cu_seq_q" in code
    return {
        "status": "CPU_CONTRACT_CONFIRMED_PADDING_FULL_AC_NOT_ACCEPTED",
        "atol": 0.005,
        "rtol": 0.02,
        "hf_behavior": observations,
        "sources": {
            "hf_mask": source(apply_mask_to_padding_states),
            "production_policy": source(configure_gdn_numerics),
            "production_backend_guard": source(validate_gdn_inputs),
            "reference": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "annotation": annotation,
                "forward": code,
            },
        },
        "reference_contract": (
            "VarlenMetadata describes sequence resets, not boolean padding. "
            "No bool mask was passed to reference or treated as equivalent."
        ),
        "next_acceptance_plan": [
            "Full published geometry/shared initial-state; no internal observation "
            "hooks or retain_grad. Production validation hooks remain intact.",
            "Production masked module/AC path is BLOCKED by current guards. "
            "Do not remove hooks or invoke unbound forward to claim acceptance.",
            "Separate research path must declare actual raw HF 2D mask and "
            "external reference input mapping. B1 HF leaves padding unchanged; "
            "B2 HF zeros it. Reference boundary adapter must match each case.",
            "Use original reference metadata builder with row-contiguous "
            "positions 0..127: offsets B1=[0,128], B2=[0,128,256]. Never "
            "replace a padding suffix with a new document at position 93.",
            "Test B1/B2, all-valid/right-padding, changed nonzero padding inputs "
            "and cross-row perturbations. Compare valid outputs, raw-input dx "
            "and all native-key parameter VJPs. Padded outputs are observed, "
            "not assumed zero. No packed/internal-hole/left-padding claim.",
            "Only after eager semantics are explicit, checkpoint the complete "
            "module via nonreentrant full AC with identical state/RNG/data. "
            "Compare each side against its eager baseline and across frameworks; "
            "record output/dx/all parameter VJPs, finite/nonempty/nonzero "
            "evidence with legitimate zero padding regions separately checked.",
            "Reference input masking is an explicit external adapter, not native "
            "reference mask support. Any research result remains distinct from "
            "production FSDP/Trainer/full40/100-step acceptance and guard release.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
