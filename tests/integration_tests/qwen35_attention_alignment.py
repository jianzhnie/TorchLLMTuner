"""Actual layer3 input/weights: HF eager/SDPA vs unchanged reference SDPA.

This isolated attention diagnostic preserves published geometry and original
modules. It does not establish whole-layer, Trainer or 100-step acceptance.
The primary VJP has no observation hooks; a separate forward captures stages.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path

# The reference plugin must register its NPU compatibility patches first.
# ruff: noqa: I001
import torch
import torchtitan_npu  # noqa: F401
import torchtitan_npu.models.qwen3_5  # noqa: F401
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan_npu.models.qwen3_5 import model_registry

from llmtuner.models.hf.qkv_projection import (
    apply_fp32_rotary_projection,
    apply_qwen35_fp32_rotary,
)
from tests.integration_tests.qwen35_cross_framework import boundary

ATOL, RTOL = 0.005, 0.02


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def compare(candidate, reference):
    candidate, reference = candidate.float().cpu(), reference.float().cpu()
    assert candidate.shape == reference.shape and candidate.numel()
    assert candidate.isfinite().all() and reference.isfinite().all()
    delta = (candidate - reference).abs()
    bad = delta > ATOL + RTOL * candidate.abs()
    return {
        "exact": torch.equal(candidate, reference),
        "max_abs": float(delta.max()),
        "rms": float(delta.square().mean().sqrt()),
        "failed_elements": int(bad.sum()),
        "passed": not bool(bad.any()),
        "elements": candidate.numel(),
    }


def run(args, report):
    assert args.execute, "Require --execute"
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    set_spmd_backend("partial_dtensor")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    assert (
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
    ) == (40, 2048, 16)
    assert config.layer_types[3] == "full_attention"
    config.rotary_compute_dtype = config.rotary_cache_dtype = "float32"
    spec = model_registry("35B-A3B")
    spec.model.layers[3].attention.inner_attention = ScaledDotProductAttention.Config()
    adapter = Qwen35StateDictAdapter(spec.model, None)
    source = load(args.artifacts / "initial-3.pt")
    selected = {
        k: v
        for k, v in source.items()
        if k.startswith("model.layers.3.self_attn.")
        or k == "model.layers.3.input_layernorm.weight"
    }
    head = load(args.artifacts / "initial-outer.pt")["lm_head.weight"]
    native = adapter.from_hf(
        {**boundary(selected, to_reference=True), "lm_head.weight": head}
    )
    native.pop("lm_head.weight")
    restored = boundary(adapter.to_hf(native), to_reference=False)
    assert restored.keys() == selected.keys()
    assert all(torch.equal(restored[k], v) for k, v in selected.items())
    del source, head, restored
    ref = spec.model.layers[3].attention.build().bfloat16().to(device)
    ref.init_states(buffer_device=device)
    ref.load_state_dict(
        {
            k.removeprefix("layers.3.attn."): v
            for k, v in native.items()
            if k.startswith("layers.3.attn.")
        },
        strict=True,
    )
    norm = Qwen3_5MoeRMSNorm(2048, eps=config.rms_norm_eps).bfloat16().to(device)
    norm.load_state_dict({"weight": selected["model.layers.3.input_layernorm.weight"]})
    actual = load(args.capture / "hf-0-layers-rank0.pt")[2]
    other = load(args.capture / "reference-0-layers-rank0.pt")[2]
    assert torch.equal(actual, other) and actual.shape == (1, 128, 2048)
    with torch.no_grad():
        x = norm(actual.to(device)).detach()
    assert x.isfinite().all() and x.count_nonzero()
    positions = torch.arange(128, device=device)[None]
    generator = torch.Generator(device="cpu").manual_seed(937)
    dy = torch.randn((1, 128, 2048), generator=generator).bfloat16().to(device)
    report["input"] = {
        "source": "full40 layer2 output then native input norm",
        "shape": list(x.shape),
        "norm_nonzero": bool(x.count_nonzero()),
    }
    report["reference_rope_cache_dtype"] = str(ref.rope.cache.dtype)
    results = {}
    stages = {}

    def execute(name, module, forward):
        module.zero_grad(set_to_none=True)
        inputs = x.detach().requires_grad_()
        output = forward(inputs)
        output.backward(dy)
        assert output.isfinite().all() and inputs.grad.isfinite().all()
        gradients = {k: p.grad.detach().cpu() for k, p in module.named_parameters()}
        assert all(v.isfinite().all() for v in gradients.values())
        results[name] = {
            "output": output.detach().cpu(),
            "dx": inputs.grad.detach().cpu(),
            "gradients": gradients,
        }

    execute("reference", ref, lambda v: ref(v, None, positions))
    ref_stages = {}
    handles = []
    for name in ("wq", "wk", "wv", "q_norm", "k_norm"):

        def observe(_m, _a, value, key=name):
            ref_stages[key] = value.detach().cpu()

        handles.append(getattr(ref, name).register_forward_hook(observe))

    def observe_inner(_m, values, kwargs):
        for name, value in zip(("rotated_q", "rotated_k", "v"), values, strict=True):
            ref_stages[name] = value.detach().cpu()

    handles.append(
        ref.inner_attention.register_forward_pre_hook(observe_inner, with_kwargs=True)
    )
    try:
        with torch.no_grad():
            assert torch.equal(
                ref(x, None, positions).cpu(), results["reference"]["output"]
            )
    finally:
        for handle in handles:
            handle.remove()
    for backend in ("eager", "sdpa"):
        cfg = copy.deepcopy(config)
        cfg._attn_implementation = backend
        holder = nn.Module()
        holder.config = cfg
        holder.model = nn.Module()
        holder.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(cfg, device=device)
        holder.attention = Qwen3_5MoeAttention(cfg, 3).bfloat16().to(device)
        holder.attention.load_state_dict(
            {
                k.removeprefix("model.layers.3.self_attn."): v
                for k, v in selected.items()
                if k.startswith("model.layers.3.self_attn.")
            }
        )
        apply_fp32_rotary_projection(holder)
        cosine, sine = holder.model.rotary_emb(x, positions)
        mask = None
        if backend == "eager":
            mask = torch.full(
                (128, 128),
                torch.finfo(torch.bfloat16).min,
                dtype=torch.bfloat16,
                device=device,
            ).triu(1)[None, None]

        def forward(
            v,
            module=holder.attention,
            cache=(cosine, sine),
            attention_mask=mask,
        ):
            return module(v, cache, attention_mask, use_cache=False)[0]

        execute(backend, holder.attention, forward)
        with torch.no_grad():
            qg = holder.attention.q_proj(x).view(1, 128, 16, 512)
            q, _ = qg.chunk(2, dim=-1)
            k = holder.attention.k_proj(x).view(1, 128, 2, 256)
            q, k = holder.attention.q_norm(q), holder.attention.k_norm(k)
            qr, kr = apply_qwen35_fp32_rotary(
                q.transpose(1, 2), k.transpose(1, 2), cosine, sine
            )
            stages[backend] = {
                "q_projection": compare(qg.flatten(-2), ref_stages["wq"]),
                "k_projection": compare(holder.attention.k_proj(x), ref_stages["wk"]),
                "q_norm": compare(q, ref_stages["q_norm"]),
                "k_norm": compare(k, ref_stages["k_norm"]),
                "rotated_q": compare(qr.transpose(1, 2), ref_stages["rotated_q"]),
                "rotated_k": compare(kr.transpose(1, 2), ref_stages["rotated_k"]),
            }
        del holder
    reference_gradients = boundary(
        adapter.to_hf(
            {
                "layers.3.attn." + k: v
                for k, v in results["reference"]["gradients"].items()
            }
        ),
        to_reference=False,
    )
    report["comparisons"] = {}
    for backend in ("eager", "sdpa"):
        report["comparisons"][backend] = {
            "output": compare(
                results[backend]["output"], results["reference"]["output"]
            ),
            "dx": compare(results[backend]["dx"], results["reference"]["dx"]),
            "gradients": {
                k: compare(v, reference_gradients["model.layers.3.self_attn." + k])
                for k, v in results[backend]["gradients"].items()
            },
            "stages": stages[backend],
        }
    report["status"] = "isolated_attention_complete_not_training_acceptance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "artifacts", "capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = {
        "status": "failed",
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        run(args, report)
    except Exception as error:
        report.update(
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if report["status"] == "failed":
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
