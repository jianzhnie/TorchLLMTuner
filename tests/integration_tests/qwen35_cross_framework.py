"""Full 40-layer native HF / TorchTitan v0.3.0 + NPU diagnostic.

Run only in the existing reference environment, with repository PYTHONPATH.
Three explicit sequential invocations share --model, --artifacts, --output,
--expected-reference-commit and --expected-npu-commit (full revision hashes):
  python .../qwen35_cross_framework.py --phase prepare ... --execute
  torchrun --standalone --nproc-per-node=8 ... --phase hf ... --execute
  torchrun --standalone --nproc-per-node=8 ... --phase reference ... --execute

CPU prepare needs ~140GB host RAM for native FP32 construction and invokes
initialize_random_hf_model before BF16 conversion. Runtime uses only sharded
model storage; canonical diagnostics stage at most one decoder layer on CPU
per rank (plus the real LM head for initial adapter calls), then save dim-0
shards. Disk usage is large: all gradients and updated weights per step.
All ranks use the same fixed unpadded token document.
Reference uses original NPU GDN override, causal SDPA, and original parallelizer.
HF uses native math + direct FSDP, NOT TorchLLMTuner trainer assembly. A pass
therefore does not validate the candidate framework training path.
HF-only production router/experts/rotary flags install existing adapters before
FSDP. Supply the same flags in reference phase for contract verification only;
the reference model is not adapted. These runs require a NEW --capture-artifacts
directory, leaving --artifacts initial weights and all prior captures untouched.
Imports require the dedicated reference environment; even --help imports the
NPU plugin. Use source compile/lint for accelerator-free validation.

Fresh SGD from the original random state is NOT LIMO or full-scope 100-step
training acceptance. No step-50 resume, checkpoints, reduced layers, or vision
training. Runtime API/shape mismatches fail with per-rank JSON, never fake pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

# NPU patches must load before upstream reference configuration modules.
# ruff: noqa: I001
import torch
import torch.distributed as dist
import torch.nn.functional as functional
import torchtitan_npu
import torchtitan_npu.models.qwen3_5 as reference
import torchtitan
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torchtitan.config import (
    CompileConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
    apply_overrides,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from transformers import AutoConfig, AutoModelForCausalLM

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.models.hf.materialization import (
    initialize_random_hf_model,
    materialize_meta_model,
)
from llmtuner.models.hf.expert_kernels import (
    stabilize_moe_experts,
    stabilize_moe_router,
)
from llmtuner.models.hf.gdn_numerics import configure_gdn_numerics
from llmtuner.models.hf.linear_attention import apply_linear_attention_backend
from llmtuner.models.hf.model_setup import configure_experts_implementation
from llmtuner.models.hf.qkv_projection import apply_fp32_rotary_projection

from tests.integration_tests.qwen35_backward_capture import (
    backward_capture_path,
    capture_layer_cotangents,
    ordered_layers,
)
from tests.integration_tests.qwen35_conv_alignment import (
    capture_hf_conv,
    capture_reference_conv,
    conv_capture_path,
)

EXPERT_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")


def boundary(state: dict, *, to_reference: bool) -> dict:
    """Native [E,2F,D]/[E,D,F] <-> adapter [E,D,2F]/[E,F,D]."""
    result = {}
    for key, value in state.items():
        if to_reference and key.startswith("model."):
            key = "model.language_model." + key[6:]
        elif not to_reference and key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
        if key.endswith(EXPERT_SUFFIXES):
            value = value.transpose(-1, -2).contiguous()
        result[key] = value
    return result


def provenance(root: Path, expected: str) -> dict:
    """Archives have no git identity: record a claim separately from file hashes."""
    assert len(expected) == 40 and all(c in "0123456789abcdef" for c in expected), (
        "Supply the full expected commit hash; archives record it as unverified"
    )
    files = sorted(root.rglob("*.py"))
    assert files, f"No Python sources in {root}"
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    actual = None
    if (root / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        assert actual == expected, f"Expected {expected}, found {actual}"
    return {
        "path": str(root),
        "expected_commit": expected,
        "git_commit": actual,
        "python_sources_sha256": digest.hexdigest(),
        "identity_verified_by_git": actual is not None,
    }


def compare(a: Any, b: Any, atol: float, rtol: float) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "elements": a.numel(),
            "passed": False,
            "error": "shape or dtype mismatch",
            "left": {"shape": list(a.shape), "dtype": str(a.dtype)},
            "right": {"shape": list(b.shape), "dtype": str(b.dtype)},
        }
    if a.numel() == 0:
        # A scalar gate's dim-0 shard is empty on all but one rank. This is
        # valid ownership, but does not count as nonempty numerical evidence.
        return {
            "elements": 0,
            "max_abs": 0.0,
            "rms": 0.0,
            "passed": True,
            "empty_shard": True,
        }
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a - b).abs()
    return {
        "elements": a.numel(),
        "max_abs": delta.max().item() if finite else None,
        "rms": delta.square().mean().sqrt().item() if finite else None,
        "passed": finite and bool((delta <= atol + rtol * a.abs()).all()),
    }


def compare_state(a: dict, b: dict, atol: float, rtol: float) -> dict:
    assert a.keys() == b.keys(), {
        "missing": sorted(a.keys() - b.keys()),
        "extra": sorted(b.keys() - a.keys()),
    }
    metrics = {key: compare(a[key], b[key], atol, rtol) for key in a}
    return {"passed": all(m["passed"] for m in metrics.values()), "tensors": metrics}


def config(args: argparse.Namespace) -> Any:
    native = AutoConfig.from_pretrained(args.model, local_files_only=True)
    native = getattr(native, "text_config", native)
    assert (native.num_hidden_layers, native.hidden_size, native.num_experts) == (
        40,
        2048,
        256,
    )
    assert native.moe_intermediate_size == 512 and not native.tie_word_embeddings
    native._attn_implementation = getattr(args, "hf_attention_backend", "eager")
    return native


def groups() -> list[str]:
    return ["outer"] + [str(i) for i in range(40)]


def group_key(key: str) -> str:
    parts = key.split(".")
    return parts[parts.index("layers") + 1] if "layers" in parts else "outer"


def prepare(args: argparse.Namespace, report: dict) -> None:
    """CPU-only native staging; reference conversion happens later per layer."""
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1
    args.artifacts.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    # Avoid initializing every large tensor once in the HF constructor and
    # then immediately replacing those samples in our explicit initializer.
    # CPU storage still needs ~140 GB; no full model is allocated on NPU.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config(args), dtype=torch.float32)
    materialize_meta_model(model, torch.device("cpu"))
    initialize_random_hf_model(model)
    model.to(dtype=torch.bfloat16)
    for group in groups():
        state = {
            k: v.detach()
            for k, v in model.state_dict().items()
            if group_key(k) == group
        }
        assert all(bool(torch.isfinite(v).all()) for v in state.values())
        torch.save(state, args.artifacts / f"initial-{group}.pt")
    manifest = {
        "source": "HF native random CPU FP32 then BF16",
        "initializer": "initialize_random_hf_model",
        "initializer_source_sha256": file_hash(
            Path(initialize_random_hf_model.__code__.co_filename)
        ),
        "seed": args.seed,
        "resume": False,
        "original_step": 0,
        "config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "files": {
            path.name: file_hash(path)
            for path in sorted(args.artifacts.glob("initial-*.pt"))
        },
    }
    (args.artifacts / "initial.json").write_text(json.dumps(manifest, indent=2))
    report["initialization"] = manifest
    report["status"] = "prepared_only"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def attach_reference_gdn(model, kernel) -> int:
    """Diagnostic intervention, never a candidate Trainer/backend acceptance."""

    def recurrence(query, key, value, *, g, beta, **kwargs):
        assert kwargs.get("initial_state") is None
        assert not kwargs.get("output_final_state", False)
        assert kwargs.get("use_qk_l2norm_in_kernel", False)
        return kernel(query, key, value, g, beta), None

    count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5MoeGatedDeltaNet":
            module.chunk_gated_delta_rule = recurrence
            count += 1
    assert count == 30, f"Expected all 30 published GDN layers, got {count}"
    return count


def hf_production_options(args) -> dict:
    """Only opt-in options enter the contract, preserving legacy captures."""
    return {
        key: value
        for key, value in {
            "router_compute_dtype": args.hf_router_compute_dtype,
            "experts_implementation": args.hf_experts_implementation,
            "gdn_numerical_policy": getattr(args, "hf_gdn_numerical_policy", "native"),
            "linear_attention_backend": getattr(
                args, "hf_linear_attention_backend", "native"
            ),
            "gdn_deterministic_reduction": getattr(
                args, "hf_gdn_deterministic_reduction", False
            ),
            "rotary_compute_dtype": args.hf_rotary_compute_dtype,
            "rotary_cache_dtype": args.hf_rotary_cache_dtype,
            "attention_backend": (
                "sdpa"
                if getattr(args, "hf_attention_backend", "eager") == "sdpa"
                else None
            ),
        }.items()
        if value not in (None, "native", False)
    }


def install_hf_production_adapters(model, args, report) -> None:
    """Use existing production entry points before FSDP; retain native params."""
    options = hf_production_options(args)
    if not options:
        return
    parameters = dict(model.named_parameters())
    keys = set(model.state_dict())
    for key, value in options.items():
        setattr(model.config, key, value)
    configure_experts_implementation(model.config, model_cls=type(model))
    stabilize_moe_router(model)
    stabilize_moe_experts(model)
    configure_gdn_numerics(model)
    apply_linear_attention_backend(model)
    if args.hf_rotary_compute_dtype is not None:
        apply_fp32_rotary_projection(model)
    counts = {
        "routers": sum(
            type(m).__name__ == "Qwen3_5MoeTopKRouter" for m in model.modules()
        ),
        "experts": sum(
            type(m).__name__ == "Qwen3_5MoeExperts" for m in model.modules()
        ),
        "full_attention": sum(
            type(m).__name__ == "Qwen3_5MoeAttention" for m in model.modules()
        ),
    }
    assert counts == {"routers": 40, "experts": 40, "full_attention": 10}, counts
    assert set(model.state_dict()) == keys, "Adapters changed state keys"
    after = dict(model.named_parameters())
    assert parameters.keys() == after.keys()
    assert all(after[key] is value for key, value in parameters.items())
    report["hf_production_adapters_installed"] = options
    report["hf_production_module_counts"] = counts


def run(args: argparse.Namespace, report: dict) -> None:
    assert args.execute, "Require --execute; no numerical result produced"
    assert importlib.metadata.version("transformers").startswith("5.13.")
    report["versions"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers")
    }
    if args.phase == "prepare":
        prepare(args, report)
        return
    assert int(os.environ.get("WORLD_SIZE", "1")) == 8, "Require all 8 ranks"
    manifest = json.loads((args.artifacts / "initial.json").read_text())
    assert manifest["original_step"] == 0 and not manifest["resume"]
    assert manifest["config_sha256"] == file_hash(args.model / "config.json")
    report["initialization"] = manifest
    contract = {
        "tokens": args.tokens,
        "steps": args.steps,
        "lr": args.lr,
        "initial_manifest_sha256": file_hash(args.artifacts / "initial.json"),
    }
    if args.deterministic_algorithms:
        contract["deterministic_algorithms"] = True
    if args.hf_gdn_kernel != "native":
        contract["hf_gdn_kernel"] = args.hf_gdn_kernel
    if args.trace_layer_outputs:
        contract["trace_layer_outputs"] = True
    if args.trace_layer_cotangents is not None:
        contract["trace_layer_cotangents"] = True
        contract["layer_cotangent_observer_sha256"] = file_hash(
            Path(capture_layer_cotangents.__code__.co_filename)
        )
    options = hf_production_options(args)
    capture_dir = args.capture_artifacts or args.artifacts
    if options:
        assert args.capture_artifacts is not None, (
            "Production interventions require a separate --capture-artifacts directory"
        )
        assert capture_dir.resolve() != args.artifacts.resolve()
        contract["hf_production_options"] = options
        if args.hf_rotary_compute_dtype == "float32":
            # Match production FSDP: decoder kwargs include externally built
            # FP32 position_embeddings, which must never roundtrip via BF16.
            contract["hf_fsdp_cast_forward_inputs"] = False
        contract["hf_production_sources"] = {
            str(Path(function.__code__.co_filename).resolve()): file_hash(
                Path(function.__code__.co_filename)
            )
            for function in (
                stabilize_moe_router,
                configure_experts_implementation,
                apply_fp32_rotary_projection,
                configure_gdn_numerics,
                apply_linear_attention_backend,
            )
        }
        if options.get("linear_attention_backend") == "npu_triton":
            kernel_source = (
                Path(apply_linear_attention_backend.__code__.co_filename)
                .resolve()
                .parents[2]
                / "accelerator"
                / "gated_delta.py"
            )
            contract["hf_production_sources"][str(kernel_source)] = file_hash(
                kernel_source
            )
    report["capture_contract"] = contract
    report["capture_artifacts"] = str(capture_dir)
    completion = capture_dir / "hf-complete.json"
    if args.phase == "reference":
        assert json.loads(completion.read_text()) == contract, "HF run contract differs"
    else:
        assert not completion.exists(), "HF capture already exists; use fresh artifacts"
        if args.capture_artifacts is not None and int(os.environ["RANK"]) == 0:
            assert not capture_dir.exists(), "Capture directory must be NEW"
    if args.phase == "reference":
        assert not any(capture_dir.glob("reference-*.pt")), (
            "Refuse overwriting an existing reference capture"
        )
    for package, root, expected in (
        (torchtitan, args.reference_root, args.expected_reference_commit),
        (torchtitan_npu, args.npu_root, args.expected_npu_commit),
    ):
        actual_root = Path(package.__file__).resolve().parents[1]
        assert actual_root == root.resolve(), f"Wrong {package} import: {actual_root}"
        report[package.__name__] = provenance(root, expected)
    rank = int(os.environ["RANK"])
    device = torch.device(f"npu:{os.environ['LOCAL_RANK']}")
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    if rank == 0:
        capture_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    set_spmd_backend("partial_dtensor")
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=8,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=8,
        spmd_backend="partial_dtensor",
    )
    dims.build_mesh()
    spec = reference.model_registry("35B-A3B", attn_backend="flex")
    assert isinstance(spec.model.layers, list) and len(spec.model.layers) == 40
    assert not spec.model.enable_weight_tying
    assert (
        inspect.signature(ScaledDotProductAttention.forward)
        .parameters["is_causal"]
        .default
        is True
    )
    for layer in spec.model.layers:
        layer.moe.load_balance_coeff = None
        if layer.attention is not None:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()
    report["overrides"] = apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.qwen3_5.gated_delta.npu"]),
        spec.model,
    )
    assert report["overrides"], "No original NPU override applied"
    # This is text-only state. The NPU adapter additionally tries to recover
    # untrained vision/MTP weights; use its original upstream text mapping.
    adapter = Qwen35StateDictAdapter(spec.model, None)
    report["state_adapter"] = type(adapter).__module__ + "." + type(adapter).__name__
    is_reference = args.phase == "reference"
    with torch.device("meta"):
        model = (
            spec.model.build()
            if is_reference
            else AutoModelForCausalLM.from_config(config(args))
        )
    model.to(dtype=torch.bfloat16)
    if is_reference:
        model.vision_encoder = None
    else:
        install_hf_production_adapters(model, args, report)
        if args.hf_gdn_kernel == "reference-npu":
            report["gdn_layers_replaced"] = attach_reference_gdn(
                model, spec.model.layers[0].delta_net.kernel.build()
            )
    if is_reference:
        model = spec.parallelize_fn(
            model,
            parallel_dims=dims,
            training=TrainingConfig(
                mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
            ),
            parallelism=ParallelismConfig(
                spmd_backend="partial_dtensor", fsdp_reshard_after_forward="always"
            ),
            compile_config=CompileConfig(enable=False),
            ac_config=None,
            dump_folder=str(args.artifacts),
        )
        model.to_empty(device=device)
        model.init_states(buffer_device=device)
    else:
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=args.hf_rotary_compute_dtype != "float32",
        )
        report["hf_fsdp_cast_forward_inputs"] = policy.cast_forward_inputs
        for layer in model.model.layers:
            fully_shard(
                layer,
                mesh=dims.get_mesh("fsdp"),
                mp_policy=policy,
                reshard_after_forward=True,
            )
        for module in (model.model.embed_tokens, model.lm_head):
            fully_shard(
                module,
                mesh=dims.get_mesh("fsdp"),
                mp_policy=policy,
                reshard_after_forward=True,
            )
        fully_shard(
            model,
            mesh=dims.get_mesh("fsdp"),
            mp_policy=policy,
            reshard_after_forward=True,
        )
        materialize_meta_model(model, device)
    parameters = dict(model.named_parameters())
    report["text_parameters"] = sum(p.numel() for p in parameters.values())
    assert 34_000_000_000 < report["text_parameters"] < 36_000_000_000
    report["initial_tensor_count"] = len(parameters)
    # Use the genuine shared head, never a synthetic adapter placeholder.
    head = load(args.artifacts / "initial-outer.pt")["lm_head.weight"]
    for group in groups():
        path = args.artifacts / f"initial-{group}.pt"
        assert file_hash(path) == manifest["files"][path.name], "Changed initial state"
        native = load(path)
        state = native
        if is_reference:
            # Adapter requires lm_head even when converting an isolated layer.
            state = boundary(native, to_reference=True)
            state.setdefault("lm_head.weight", head)
            state = adapter.from_hf(state)
            if group != "outer":
                del state["lm_head.weight"]
            restored = boundary(adapter.to_hf(state), to_reference=False)
            assert compare_state(native, restored, 0, 0)["passed"], (
                f"Initial adapter roundtrip differs: {group}"
            )
            del restored
        expected = {k for k in parameters if group_key(k) == group}
        assert state.keys() == expected, "Initial layer keys differ"
        with torch.no_grad():
            for key, value in state.items():
                parameter = parameters[key]
                assert parameter.shape == value.shape
                assert len(parameter.placements) == 1
                assert parameter.placements[0].dim == 0, "Only dim-0 FSDP supported"
                shard = value.tensor_split(8, dim=0)[rank].to(device)
                assert shard.shape == parameter.to_local().shape
                parameter.to_local().copy_(shard)
                assert torch.equal(parameter.to_local(), shard), "Initial shard differs"
        del native, state
    del head
    tokens = torch.tensor([args.tokens], device=device)
    assert tokens.shape[1] >= 2 and tokens.min() >= 0
    assert tokens.max() < config(args).vocab_size
    positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    report["steps"] = []
    report["tokens"] = args.tokens

    def capture(kind: str, step: int) -> dict:
        summary = {"passed": True, "tensors": {}}
        for group in groups():
            state = {}
            # Validate every local source before any peer begins gathering.
            # A missing gradient on one rank must not strand peers in all-gather.
            with synchronize_local_errors(f"Qwen3.5 {kind} source group {group}"):
                sources = {}
                for key, parameter in parameters.items():
                    if group_key(key) != group:
                        continue
                    value = parameter.grad if kind == "gradients" else parameter
                    assert isinstance(value, DTensor), f"Missing DTensor {kind}: {key}"
                    sources[key] = value
            for key, value in sources.items():
                # One layer of CPU staging, never gather a full model on NPU.
                state[key] = value.full_tensor().detach().cpu()
            # Local validation/I/O must finish on every peer before the next
            # group's full_tensor collectives. A rank-local error must not
            # leave another rank issuing an unmatched all-gather.
            with synchronize_local_errors(f"Qwen3.5 {kind} capture group {group}"):
                if is_reference:
                    state = boundary(adapter.to_hf(state), to_reference=False)
                local = {
                    k: v.tensor_split(8, dim=0)[rank].clone() for k, v in state.items()
                }
                path = capture_dir / f"hf-{step}-{kind}-{group}-rank{rank}.pt"
                if not is_reference:
                    assert not path.exists(), f"Refuse stale capture: {path}"
                    torch.save(local, path)
                else:
                    torch.save(
                        local, path.with_name(path.name.replace("hf-", "reference-", 1))
                    )
                    metrics = compare_state(
                        load(path), local, **report["tolerances"][kind]
                    )
                    summary["passed"] &= metrics["passed"]
                    summary["tensors"].update(metrics["tensors"])
                del state, local
        if is_reference:
            summary["compared_elements"] = sum(
                item["elements"] for item in summary["tensors"].values()
            )
            summary["passed"] &= summary["compared_elements"] > 0
        return summary

    conv_path = None
    if args.capture_first_conv is not None:
        with synchronize_local_errors("First conv capture preflight"):
            conv_path = conv_capture_path(
                args.capture_first_conv, phase=args.phase, steps=args.steps, rank=rank
            )
            if conv_path is not None:
                assert not conv_path.parent.exists(), (
                    "Conv capture directory must be NEW"
                )
                assert not any(
                    "checkpoint" in type(module).__name__.lower()
                    for module in model.modules()
                ), "Conv capture requires no activation checkpointing"
                report["first_conv_capture"] = str(conv_path)

    backward_path = None
    if args.trace_layer_cotangents is not None:
        with synchronize_local_errors("Layer cotangent capture preflight"):
            backward_path = backward_capture_path(
                args.trace_layer_cotangents,
                phase=args.phase,
                steps=args.steps,
                rank=rank,
            )
            assert not backward_path.parent.exists(), "Cotangent directory must be NEW"
            assert not any(
                "checkpoint" in type(module).__name__.lower()
                for module in model.modules()
            ), "No AC in cotangent capture"
            assert not getattr(model, "is_gradient_checkpointing", False)
            backward_layers = ordered_layers(model, reference=is_reference)
            report["layer_cotangent_capture"] = str(backward_path)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        layer_outputs, handles = {}, []
        if args.trace_layer_outputs:
            layers = model.layers if is_reference else model.model.layers
            # TorchTitan uses ModuleDict; HF uses ModuleList. Iterating the
            # former yields names rather than the decoder modules.
            if isinstance(layers, torch.nn.ModuleDict):
                assert list(layers) == [str(index) for index in range(40)]
                layers = layers.values()

            def observe_layer(index, outputs):
                def observe(_module, _inputs, output):
                    assert index not in outputs, "Repeated layer forward"
                    outputs[index] = output.detach().cpu()

                return observe

            for index, layer in enumerate(layers):
                handles.append(
                    layer.register_forward_hook(observe_layer(index, layer_outputs))
                )
        conv_observer = nullcontext()
        if conv_path is not None:
            with synchronize_local_errors("First conv capture provenance/setup"):
                observer = capture_reference_conv if is_reference else capture_hf_conv
                target_gdn = (
                    model.layers["0"].attn
                    if is_reference
                    else model.model.layers[0].linear_attn
                )
                if is_reference:
                    assert not model.layers["0"].full_attn
                    assert [
                        tuple(getattr(target_gdn, "conv_" + name).weight.shape)
                        for name in ("q", "k", "v")
                    ] == [(2048, 1, 4), (2048, 1, 4), (4096, 1, 4)]
                else:
                    assert tuple(target_gdn.conv1d.weight.shape) == (8192, 1, 4)
                extra = (
                    {"loss_scale": 1.0 / dist.get_world_size()} if is_reference else {}
                )
                conv_observer = observer(
                    target_gdn,
                    conv_path,
                    provenance={
                        "scope": "actual full40 loss raw-conv cotangent; observed",
                        "phase": args.phase,
                        "backward_loss_scale": 1.0 / dist.get_world_size()
                        if is_reference
                        else 1.0,
                        "step": step,
                        "rank": rank,
                        "initial_manifest_sha256": contract["initial_manifest_sha256"],
                        "initial_layer0_sha256": manifest["files"]["initial-0.pt"],
                        "capture_contract": contract,
                        "entry_sha256": file_hash(Path(__file__)),
                        "conv_helper_sha256": file_hash(
                            Path(capture_hf_conv.__code__.co_filename)
                        ),
                        "tokens": args.tokens,
                        "loss": (
                            "cross_entropy(logits[:,:-1].float(),tokens[:,1:]); "
                            "local mean"
                        ),
                        "activation_checkpoint": False,
                    },
                    **extra,
                    local_errors=lambda: synchronize_local_errors(
                        "First conv capture local IO"
                    ),
                )
        backward_observer = nullcontext()
        if backward_path is not None:
            with synchronize_local_errors("Layer cotangent provenance/setup"):
                backward_observer = capture_layer_cotangents(
                    backward_layers,
                    backward_path,
                    loss_scale=1.0 / dist.get_world_size() if is_reference else 1.0,
                    provenance={
                        "scope": "actual full40 layer-output cotangents; observed",
                        "phase": args.phase,
                        "rank": rank,
                        "step": step,
                        "initial_manifest_sha256": contract["initial_manifest_sha256"],
                        "initial_layer0_sha256": manifest["files"]["initial-0.pt"],
                        "tokens": args.tokens,
                        "capture_contract": contract,
                        "entry_sha256": file_hash(Path(__file__)),
                        "observer_sha256": file_hash(
                            Path(capture_layer_cotangents.__code__.co_filename)
                        ),
                        "loss": "causal token mean CE; reference backward /8",
                        "activation_checkpoint": False,
                    },
                    local_errors=lambda: synchronize_local_errors("Layer cotangent IO"),
                )
        with conv_observer, backward_observer:
            try:
                logits = (
                    model(
                        tokens,
                        positions=positions,
                        special_tokens={"image_id": -1, "video_id": -1},
                    )
                    if is_reference
                    else model(input_ids=tokens, use_cache=False).logits
                )
            finally:
                for handle in handles:
                    handle.remove()
            if args.trace_layer_outputs:
                with synchronize_local_errors("Qwen3.5 layer-output trace save"):
                    assert set(layer_outputs) == set(range(40))
                    assert all(torch.isfinite(v).all() for v in layer_outputs.values())
                    side = "reference" if is_reference else "hf"
                    path = capture_dir / f"{side}-{step}-layers-rank{rank}.pt"
                    assert not path.exists(), f"Refuse stale layer trace: {path}"
                    torch.save(layer_outputs, path)
                del layer_outputs
            loss = functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
            with synchronize_local_errors("Qwen3.5 diagnostic loss validation"):
                assert bool(torch.isfinite(loss)), "Nonfinite loss"
            # The original reference parallelizer disables FSDP division (SUM),
            # whereas native fully_shard keeps its default DP average. Every rank
            # uses this same document, so dividing the reference local mean by
            # world size gives the same global target mean on both paths.
            backward_loss = loss / dist.get_world_size() if is_reference else loss
            backward_loss.backward()
        metrics = {"gradients": capture("gradients", step)}
        optimizer.step()
        metrics["weights"] = capture("weights", step)
        with synchronize_local_errors("Qwen3.5 diagnostic output comparison"):
            path = capture_dir / f"hf-{step}-outputs-rank{rank}.pt"
            outputs = {"logits": logits.detach().cpu(), "loss": loss.detach().cpu()}
            if not is_reference:
                torch.save(outputs, path)
            else:
                reference_path = capture_dir / f"reference-{step}-outputs-rank{rank}.pt"
                assert not reference_path.exists(), (
                    f"Refuse stale reference outputs: {reference_path}"
                )
                # Preserve actual reference outputs even when comparison fails,
                # so rank variation can be checked without rerunning training.
                torch.save(outputs, reference_path)
                native = load(path)
                metrics.update(
                    {
                        k: compare(native[k], v, **report["tolerances"][k])
                        for k, v in outputs.items()
                    }
                )
        report["steps"].append({"step": step + 1, **metrics})
        passed = torch.tensor(
            int(all(v["passed"] for v in metrics.values())), device=device
        )
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        assert passed.item(), f"Mismatch at diagnostic step {step + 1}"
    report["runtime_verified"] = True
    dist.barrier()
    if not is_reference and rank == 0:
        completion.write_text(json.dumps(contract, indent=2))
    report["status"] = "passed_diagnostic_only" if is_reference else "hf_captured_only"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--capture-artifacts",
        type=Path,
        help="New diagnostic capture directory; initial weights remain in --artifacts",
    )
    parser.add_argument("--hf-router-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--hf-attention-backend", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--hf-experts-implementation", choices=("native", "pre_down"), default="native"
    )
    parser.add_argument("--hf-rotary-compute-dtype", choices=("float32",))
    parser.add_argument(
        "--hf-gdn-numerical-policy",
        choices=("native", "split_qkv_fp32_norm"),
        default="native",
    )
    parser.add_argument("--hf-rotary-cache-dtype", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--hf-linear-attention-backend",
        choices=("native", "npu_triton"),
        default="native",
    )
    parser.add_argument("--hf-gdn-deterministic-reduction", action="store_true")
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="Require deterministic PyTorch algorithms in both original paths",
    )
    parser.add_argument(
        "--capture-first-conv",
        type=Path,
        help="Single-step HF/reference: new raw-conv capture directory per rank",
    )
    parser.add_argument(
        "--trace-layer-cotangents",
        type=Path,
        help="Single-step HF/reference: new directory of raw layer VJPs",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--trace-layer-outputs",
        action="store_true",
        help="Save detached layer outputs for diagnostic rank/stage comparison",
    )
    parser.add_argument(
        "--phase", choices=["prepare", "hf", "reference"], required=True
    )
    parser.add_argument("--expected-reference-commit", required=True)
    parser.add_argument("--expected-npu-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(range(100, 228)))
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument(
        "--hf-gdn-kernel",
        choices=("native", "reference-npu"),
        default="native",
        help="Diagnostic intervention only; not the candidate framework backend",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/tmp/tlt_torchtitan_v030_reference"),
    )
    parser.add_argument(
        "--npu-root",
        type=Path,
        default=Path("/home/jianzhnie/llmtuner/llm/torchtitan-npu"),
    )
    args = parser.parse_args()
    if args.capture_first_conv is not None and (
        args.phase not in ("hf", "reference") or args.steps != 1
    ):
        parser.error(
            "--capture-first-conv requires single-step HF/reference phase (no AC)"
        )
    if args.trace_layer_cotangents is not None and (
        args.phase not in ("hf", "reference") or args.steps != 1
    ):
        parser.error("Layer cotangent capture requires single-step HF/reference")
    if args.hf_gdn_deterministic_reduction and (
        args.hf_linear_attention_backend != "npu_triton"
    ):
        parser.error("Deterministic GDN reduction requires npu_triton")
    if args.hf_linear_attention_backend != "native" and args.hf_gdn_kernel != "native":
        parser.error(
            "Production GDN backend cannot be overwritten by a reference kernel"
        )
    report = {
        "status": "failed",
        "scope": "full 40-layer text fixed-token diagnostic",
        "acceptance": "NOT LIMO; NOT full-scope cross-framework 100-step acceptance",
        "load_balance_coeff": None,
        "vision": "published model built; encoder detached; no vision/MTP training",
        "execution": "CPU prepare, then HF 8-rank FSDP, then reference 8-rank FSDP",
        "phase": args.phase,
        "hf_gdn_kernel": args.hf_gdn_kernel,
        "hf_path": "native HF math + direct FSDP; NOT candidate trainer assembly",
        "runtime_verified": False,
        "attention": (
            f"HF {args.hf_attention_backend} / original reference causal SDPA; "
            "single document"
        ),
        "optimizer": {
            "name": "SGD",
            "lr": args.lr,
            "momentum": 0,
            "weight_decay": 0,
            "fresh_at_original_step": 0,
        },
        "gradient_normalization": (
            "global target mean: HF FSDP average; original reference FSDP SUM "
            "with local mean divided by world size; identical document per rank"
        ),
        "tolerances": {
            name: {"atol": args.atol, "rtol": args.rtol}
            for name in ("logits", "loss", "gradients", "weights")
        },
        "criterion": "all finite, every element abs(ref-HF) <= atol+rtol*abs(HF)",
    }
    try:
        assert args.hf_rotary_cache_dtype is None or (
            args.hf_rotary_compute_dtype == "float32"
        ), "HF rotary cache intervention requires FP32 compute"
        if args.phase == "prepare":
            assert not hf_production_options(args) and args.capture_artifacts is None, (
                "Prepare remains native; adapter flags belong to capture/comparison"
            )
        assert args.steps > 0 and math.isfinite(args.lr) and args.lr > 0
        assert all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol))
        if args.phase != "prepare" and (
            not args.tokens or len(args.tokens) % 64 or min(args.tokens) < 0
        ):
            raise ValueError(
                "The original NPU GDN kernel requires a positive multiple of "
                "64 tokens with nonnegative token IDs"
            )
        torch.use_deterministic_algorithms(args.deterministic_algorithms)
        report["deterministic_algorithms"] = (
            torch.are_deterministic_algorithms_enabled()
        )
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        rank = os.environ.get("RANK", "0")
        output = args.output.with_name(args.output.stem + f"-rank{rank}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] == "failed":
        raise SystemExit(report.get("error", "Diagnostic failed"))


if __name__ == "__main__":
    main()
