"""Single-document FP32 Qwen alignment against the real torchtitan-npu runtime.

python tests/integration_tests/npu_qwen_alignment.py --output outputs/alignment

Set PYTHONPATH to the baseline repository, isolated runtime, and this repository.

This diagnostic uses native SDPA, not packed attention or either full Trainer.
It does not establish distributed or Qwen3-8B training equivalence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from tests.caps import require_env


def _baseline() -> tuple[torch.nn.Module, object]:
    importlib.import_module("torchtitan_npu")
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear
    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.qwen3 import _build_qwen3_layers, _qwen3_norm
    from torchtitan.models.qwen3.model import Qwen3Model
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

    layers = _build_qwen3_layers(
        n_layers=2,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        hidden_dim=128,
        fuse_qkv=False,
        attn_backend="flex",
        rope=CosSinRoPE.Config(dim=16, max_seq_len=32, theta=1000000.0),
    )
    for layer in layers:
        layer.attention.inner_attention = ScaledDotProductAttention.Config()
    config = Qwen3Model.Config(
        vocab_size=128,
        dim=64,
        norm=_qwen3_norm(64),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(num_embeddings=128, embedding_dim=64),
        lm_head=Linear.Config(in_features=64, out_features=128),
        layers=layers,
    )
    model = config.build().to("npu")
    model.init_states(buffer_device=torch.device("npu"))
    return model, Qwen3StateDictAdapter(config, None)


def _candidate() -> HFTransformerModel:
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            arch_overrides={
                "head_dim": 16,
                "tie_word_embeddings": False,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            },
        ),
        training=TrainingConfig(max_seq_len=32, steps=4),
    )
    return HFTransformerModel(build_model_config_for(config)).to("npu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("../torchtitan-npu"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    require_env("dtensor")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Requires a real NPU and torch_npu")
    args.output.mkdir(parents=True, exist_ok=False)
    metric_path = (
        args.baseline / "scripts/numerical_alignment/utils/metrics.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("alignment_metrics", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline metrics: {metric_path}")
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    reference, adapter = _baseline()
    candidate = _candidate()
    initial = adapter.to_hf(reference.state_dict())
    candidate.model.load_state_dict(initial, strict=True)
    for name, parameter in candidate.model.state_dict().items():
        torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
    optimizer_options = {"lr": 1e-3, "foreach": False}
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_options)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), **optimizer_options)
    generator = torch.Generator().manual_seed(1234)
    manifest = {
        "scope": "diagnostic single-document SDPA FP32, no Trainer or distribution",
        "torch": torch.__version__,
        "seed": 42,
        "input_seed": 1234,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "loss_reduction": "mean over 32 explicitly supplied targets, no shift",
        "candidate_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline_commit": subprocess.check_output(
            ["git", "-C", str(args.baseline), "rev-parse", "HEAD"], text=True
        ).strip(),
        "metric_source": str(metric_path),
        "candidate_config": candidate.model.config.to_dict(),
        "optimizer": reference_optimizer.defaults,
        "steps": args.steps,
        "clip_norm": 1.0,
        "weight_check": {
            "atol": 1e-5,
            "rtol": 1e-4,
            "basis": "FP32 diagnostic regression only, not formal P4 acceptance",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    batches = [
        (
            torch.randint(128, (1, 32), generator=generator),
            torch.randint(128, (32,), generator=generator),
        )
        for _ in range(args.steps)
    ]
    torch.save(
        {"initial_weights": initial, "batches": batches}, args.output / "inputs.pt"
    )
    for step, (tokens, targets) in enumerate(batches):
        tokens, targets = tokens.to("npu"), targets.to("npu")
        reference_optimizer.zero_grad()
        candidate_optimizer.zero_grad()
        expected = reference(tokens)
        actual = candidate(tokens.flatten()).reshape_as(expected)
        result = {
            "step": step,
            "logits": metrics.compare_tensors(
                actual.detach().cpu(), expected.detach().cpu()
            ),
        }
        assert not metrics.failed_logits_metrics(result["logits"]), result
        losses = [
            torch.nn.functional.cross_entropy(logits.reshape(-1, 128), targets)
            for logits in (actual, expected)
        ]
        for loss in losses:
            loss.backward()
        reference_gradients = adapter.to_hf(
            {name: parameter.grad for name, parameter in reference.named_parameters()}
        )
        result["gradients"] = {}
        for name, parameter in candidate.model.named_parameters():
            assert parameter.grad is not None, name
            observed = parameter.grad.detach().cpu()
            expected = reference_gradients[name]
            assert expected is not None, name
            expected = expected.detach().cpu()
            result["gradients"][name] = metrics.compare_tensors(observed, expected)
            torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-4)
        norms = [
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for model in (candidate, reference)
        ]
        for name, values in (("loss", losses), ("grad_norm", norms)):
            compared = metrics.compare_tensors(
                values[0].detach().cpu(), values[1].detach().cpu()
            )
            assert compared["mean_ae"] <= 0.02, compared
            assert compared["max_re"] <= 0.005, compared
            assert compared["mean_re"] <= 0.005, compared
            result[name] = compared
        reference_optimizer.step()
        candidate_optimizer.step()
        mapped = adapter.to_hf(reference.state_dict())
        result["weights"] = {}
        for name, parameter in candidate.model.state_dict().items():
            observed, expected = parameter.detach().cpu(), mapped[name].detach().cpu()
            result["weights"][name] = metrics.compare_tensors(observed, expected)
            torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-4)
        (args.output / f"step-{step}.json").write_text(json.dumps(result, indent=2))
        print(f"PASS step={step} logits/loss/grad_norm/all weights", flush=True)
    print(f"PASS diagnostic alignment; artifacts={args.output}", flush=True)


if __name__ == "__main__":
    main()
