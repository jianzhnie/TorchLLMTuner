"""Real CPU serialization contracts for the standalone artifact comparator."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

ENTRY = (
    Path(__file__).resolve().parents[4]
    / "tests/integration_tests/qwen35_training_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("training_comparison", ENTRY)
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def write_json(path, data):
    path.write_text(json.dumps(data))


def test_original_tolerance_is_asymmetric():
    # .02525 > .005+.02*1 but <= .005+.02*1.02525; swapping matters.
    a = torch.tensor([1.0], dtype=torch.float64)
    b = torch.tensor([1.02525], dtype=torch.float64)
    assert not COMPARATOR.metric(a, b)["passed"]
    assert COMPARATOR.metric(b, a)["passed"]
    assert not COMPARATOR.metric(b, a, exact=True)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rejected(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        COMPARATOR.metric(torch.tensor([value]), torch.ones(1))


def test_empty_requires_explicit_local_shard_permission():
    with pytest.raises(ValueError, match="Empty"):
        COMPARATOR.metric(torch.empty(0), torch.empty(0))
    assert COMPARATOR.metric(torch.empty(0), torch.empty(0), empty=True)["passed"]
    with pytest.raises(ValueError, match="Shape/dtype"):
        COMPARATOR.metric(torch.ones(1), torch.ones(1, dtype=torch.bfloat16))


def test_duplicate_json_and_escape_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        COMPARATOR.read_json(path)
    with pytest.raises(ValueError, match="Unsafe"):
        COMPARATOR.child(tmp_path, "../outside")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Escaping"):
        COMPARATOR.child(tmp_path, "escape/file")


def test_native_key_boundary():
    assert (
        COMPARATOR.canonical(
            "model.model.layers.0._checkpoint_wrapped_module._orig_mod.weight", True
        )
        == "model.layers.0.weight"
    )
    assert COMPARATOR.canonical("lm_head.weight", False) == "lm_head.weight"
    with pytest.raises(ValueError, match="Unexpected"):
        COMPARATOR.canonical("lm_head.weight", True)


def test_batch_hash_matches_actual_exporter_definitions():
    # Read the real pure helpers without importing Trainer/NPU dependencies.
    # No replacement tensor operations: serialize actual CPU tensors and compare.
    batch = {
        "input": torch.tensor([[7, 2, 11]], dtype=torch.int64),
        "labels": torch.tensor([[-100, 11, 2]], dtype=torch.int64),
        "padding_mask": torch.tensor([[False, False, True]]),
    }
    for name in ("qwen35_candidate_alignment.py", "qwen35_reference_training.py"):
        tree = ast.parse((ENTRY.parent / name).read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"tensor_info", "batch_hash"}
        ]
        assert len(nodes) == 2
        namespace = {"torch": torch, "hashlib": hashlib, "json": json, "Any": Any}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
        assert COMPARATOR.batch_hash(batch) == namespace["batch_hash"](batch)


def snapshot(tmp_path, *, candidate=True, rank=0, global_rows=9):
    folder = tmp_path / "step-00001-weights-after-update"
    folder.mkdir()
    native = "model.layers.0.weight"
    key = "model." + native if candidate else native
    size = (global_rows + 7) // 8
    offset = min(rank * size, global_rows)
    value = torch.arange(
        min(size, global_rows - offset) * 3, dtype=torch.bfloat16
    ).reshape(-1, 3)
    path = folder / "0.pt"
    torch.save({key: value}, path)
    descriptor = {
        **COMPARATOR.info(value),
        "global_shape": [global_rows, 3],
        "shard_dim": 0,
        "offset": offset,
    }
    files = {f"{i}.pt": {"tensors": {}} for i in ("outer", *map(str, range(40)))}
    files["0.pt"] = {
        "sha256": COMPARATOR.file_hash(path),
        "tensors": {key: descriptor},
    }
    index = {"files": files, "kind": "weights-after-update", "step": 1}
    if not candidate:
        index["format"] = "HF5.13 native text"
    write_json(folder / "index.json", index)
    return folder, index, native


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("rank", [0, 7])
def test_real_local_snapshot_roundtrip_and_legal_empty(tmp_path, candidate, rank):
    _, _, key = snapshot(tmp_path, candidate=candidate, rank=rank)
    data = COMPARATOR.load_snapshot(
        tmp_path, 1, "weights-after-update", "0", candidate, rank, {key}
    )
    assert data.keys() == {key}
    assert data[key][0].shape == (2 if rank == 0 else 0, 3)


@pytest.mark.parametrize("corruption", ["hash", "offset", "keys", "missing_layer"])
def test_snapshot_corruption_rejected(tmp_path, corruption):
    folder, index, key = snapshot(tmp_path)
    entry = index["files"]["0.pt"]
    if corruption == "hash":
        entry["sha256"] = "f" * 64
    elif corruption == "offset":
        entry["tensors"]["model." + key]["offset"] = 1
    elif corruption == "keys":
        entry["tensors"] = {}
    else:
        del index["files"]["39.pt"]
    write_json(folder / "index.json", index)
    with pytest.raises(ValueError):
        COMPARATOR.load_snapshot(
            tmp_path, 1, "weights-after-update", "0", True, 0, {key}
        )


def test_missing_adam_state_rejected(tmp_path):
    write_json(
        tmp_path / "checkpoint-step-00050.json",
        {
            "step": 50,
            "boundary": "Actual state_dict payloads during synchronous DCP save",
            "state_fingerprints": dict.fromkeys(
                ["lr_scheduler", "dataloader", "train_state", "rng_state"], "a" * 64
            ),
            "model_adam": {"model": {"weight": {}}, "adam": {}, "param_groups": []},
        },
    )
    with pytest.raises(ValueError, match="Model/Adam"):
        COMPARATOR.candidate_manifest(tmp_path)


def test_legacy_reference_manifest_is_not_resume_proof(tmp_path):
    write_json(
        tmp_path / "step-00050-checkpoint-state.json",
        {"cursor": 50, "rng_sha256": "a" * 64},
    )
    with pytest.raises(KeyError, match="schema"):
        COMPARATOR.reference_manifest(tmp_path, {"weight"})


def test_missing_checkpoint_statekey_rejected():
    with pytest.raises(ValueError, match="statekeys"):
        COMPARATOR.validate_fingerprints(
            {"lr_scheduler": "a" * 64, "rng_state": "b" * 64}
        )


def test_rng_contract_does_not_compare_cross_framework_values():
    # Valid independently different seeds/state bytes are reported, not compared.
    for digest in ("a" * 64, "b" * 64):
        COMPARATOR.validate_fingerprints(
            dict.fromkeys(
                ["lr_scheduler", "rng_state", "dataloader", "train_state"], digest
            )
        )
    with pytest.raises(ValueError, match="SHA256"):
        COMPARATOR.validate_fingerprints(
            dict.fromkeys(
                ["lr_scheduler", "rng_state", "dataloader", "train_state"], "invalid"
            )
        )


def test_resume_requires_exact_51_to_100_coverage(tmp_path):
    for step in range(52, 101):
        write_json(tmp_path / f"step-{step:05d}.json", {})
    with pytest.raises(ValueError, match="missing/extra"):
        COMPARATOR.verify_resume_records(tmp_path, tmp_path, "candidate")


def test_nonconstant_or_empty_lr_rejected():
    for rates in ({}, {"lr": 0.0}, {"lr": float("nan")}, {"lr": 2e-4}):
        with pytest.raises(ValueError, match="LR"):
            COMPARATOR.validate_lr({"applied_lr": rates})
    COMPARATOR.validate_lr({"applied_lr": {"lr": 3e-4}})


@pytest.mark.parametrize("rank", [0, 7])
def test_initial_proof_rehashes_immutable_hf_local_slice(tmp_path, rank):
    value = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.bfloat16)
    save_file({"model.norm.weight": value}, tmp_path / "initial.safetensors")
    local = value if rank == 0 else value[:0]
    descriptor = {
        **COMPARATOR.info(local),
        "global_shape": [1, 3],
        "shard_dim": 0,
        "offset": 0 if rank == 0 else 1,
    }
    proof = {"exact": True, "tensors": {"model.norm.weight": descriptor}}
    index = {"weight_map": {"model.norm.weight": "initial.safetensors"}}
    COMPARATOR.verify_initial_proof(tmp_path, index, proof, rank)
    descriptor["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor mismatch"):
        COMPARATOR.verify_initial_proof(tmp_path, index, proof, rank)


def test_real_adam_bf16_serialization_and_checkpoint_step(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW(
        [parameter], lr=3e-4, weight_decay=0, foreach=False, fused=False
    )
    for _ in range(50):
        parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
        optimizer.step()
    path = tmp_path / "adam.pt"
    torch.save(optimizer.state_dict(), path)
    restored = torch.optim.AdamW(
        [parameter], lr=3e-4, weight_decay=0, foreach=False, fused=False
    )
    restored.load_state_dict(torch.load(path, weights_only=True))
    state = restored.state[parameter]
    assert state["exp_avg"].dtype == torch.bfloat16
    descriptor = {**COMPARATOR.info(parameter), "global_shape": [2]}
    manifest = {
        "model": {"weight": descriptor},
        "adam": {
            "0:weight": {
                key: {**COMPARATOR.info(value), "global_shape": list(value.shape)}
                for key, value in state.items()
            }
        },
        "param_groups": [
            {"optimizer": 0, "parameters": ["weight"], "options_sha256": "a" * 64}
        ],
    }
    COMPARATOR.validate_model_adam(manifest, {"weight"}, parameter_count=2)
    with pytest.raises(ValueError, match="checkpoint step"):
        COMPARATOR.validate_model_adam(manifest, parameter_count=2, step=100)
    manifest["adam"]["0:weight"]["unknown"] = COMPARATOR.info(torch.ones(1))
    with pytest.raises(ValueError, match="extra Adam"):
        COMPARATOR.validate_model_adam(manifest, parameter_count=2)


def test_cli_missing_artifacts_reports_error_and_preserves_existing_output(tmp_path):
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        str(ENTRY),
        "--candidate",
        str(tmp_path / "candidate"),
        "--reference",
        str(tmp_path / "reference"),
        "--shared-hf",
        str(tmp_path / "shared"),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    report = COMPARATOR.read_json(output)
    assert report["status"] == "ERROR" and report["passed"] is False
    assert "FileNotFoundError" in report["error"]
    assert report["atol"] == 0.005 and report["rtol"] == 0.02
    original = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 1
    assert "Output must be new" in repeated.stderr
    assert output.read_bytes() == original


def replay_positions_helper():
    path = ENTRY.parent / "qwen35_reference_training.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "replay_positions"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["replay_positions"]


@pytest.mark.parametrize("bad", [None, "prefix", "hole", "label", "dtype", "negative"])
def test_padding_positions_reference_and_comparator_same_policy(bad):
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
    padding = torch.tensor([[False, False, False, True, True, True]])
    labels = torch.tensor([[7, 8, 9, -100, -100, -100]])
    if bad == "prefix":
        positions[0, 1] = 0
    elif bad == "hole":
        padding[0, 4] = False
    elif bad == "label":
        labels[0, 4] = 9
    elif bad == "dtype":
        padding = padding.int()
    elif bad == "negative":
        positions[0, 5] = -1
    batch = {"positions": positions, "padding_mask": padding}
    if bad is None:
        assert torch.equal(replay_positions_helper()(batch, labels), positions)
        COMPARATOR.validate_padding_positions(positions, padding, labels)
    else:
        with pytest.raises(AssertionError):
            replay_positions_helper()(batch, labels)
        with pytest.raises(ValueError):
            COMPARATOR.validate_padding_positions(positions, padding, labels)


def test_legacy_source_contract_requires_explicit_scope():
    with pytest.raises(ValueError, match="Legacy source coverage"):
        COMPARATOR.validate_contract({})
    # Opt-in does not waive the original training contract.
    with pytest.raises(ValueError, match="Unsupported/missing"):
        COMPARATOR.validate_contract({}, allow_legacy=True)


def test_source_identity_actual_file_mutation_detected(tmp_path):
    candidate = ENTRY.parent / "qwen35_candidate_alignment.py"
    node = next(
        n
        for n in ast.parse(candidate.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "verify_source_identity"
    )
    root = tmp_path / "repo"
    entry = root / "tests/integration_tests/entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("entry")
    helper = root / "llmtuner/helper.py"
    helper.parent.mkdir()
    helper.write_text("helper")
    external = tmp_path / "hf.py"
    external.write_text("hf")
    trainer = tmp_path / "trainer.py"
    trainer.write_text("trainer")
    recipe = tmp_path / "recipe.py"
    recipe.write_text("recipe")

    def function_at(path):
        namespace = {}
        exec(compile("def f(): pass", str(path), "exec"), namespace)
        return namespace["f"]

    trainer_type = type("Trainer", (), {"__init__": function_at(trainer)})

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    contract = {
        "entry_sha256": digest(entry),
        "trainer_sha256": digest(trainer),
        "recipe_sha256": digest(recipe),
        "numerical_sources_sha256": {"llmtuner/helper.py": digest(helper)},
        "hf_sources_sha256": {"hf": digest(external)},
    }
    namespace = {
        "Path": Path,
        "__file__": str(entry),
        "file_hash": digest,
        "ENTRY_STARTUP_SHA256": digest(entry),
        "Trainer": trainer_type,
        "build_config": function_at(recipe),
        "hashlib": hashlib,
        "json": json,
        "hf_source_paths": lambda: {"hf": external},
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(candidate), "exec"),
        namespace,
    )
    verify = namespace["verify_source_identity"]
    assert verify(contract)["verified"]
    for path in (helper, external, trainer, recipe, entry):
        old = path.read_bytes()
        path.write_bytes(old + b"changed")
        with pytest.raises(AssertionError):
            verify(contract)
        path.write_bytes(old)


def test_v2_source_boundary_evidence_required_at_load_and_completion(tmp_path):
    contract = {"source_contract_version": 2, "resolved_attention_backend": "sdpa"}
    identity = {
        "source_contract_version": 2,
        "verified": True,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
    }
    complete = {"source_identity": identity, "resolved_attention_backend": "sdpa"}
    write_json(tmp_path / "loaded-source-identity.json", identity)
    COMPARATOR.validate_source_evidence(tmp_path, contract, complete)
    with pytest.raises(ValueError, match="source verification"):
        COMPARATOR.validate_source_evidence(tmp_path, contract, {})
    with pytest.raises(ValueError, match="backend"):
        COMPARATOR.validate_source_evidence(
            tmp_path, contract, {**complete, "resolved_attention_backend": "eager"}
        )
    write_json(
        tmp_path / "loaded-source-identity.json", {**identity, "verified": False}
    )
    with pytest.raises(ValueError, match="source verification"):
        COMPARATOR.validate_source_evidence(tmp_path, contract, complete)


def test_actual_installed_hf_source_coverage_without_trainer_import():
    candidate = ENTRY.parent / "qwen35_candidate_alignment.py"
    node = next(
        n
        for n in ast.parse(candidate.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "hf_source_paths"
    )
    namespace = {"Path": Path, "importlib": importlib}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(candidate), "exec"),
        namespace,
    )
    paths = namespace["hf_source_paths"]()
    assert "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe" in paths
    assert "transformers.modeling_rope_utils" in paths
    assert "transformers.integrations.sdpa_attention" in paths
    assert "transformers.integrations/moe.py" in paths
    assert all(path.is_file() and path.suffix == ".py" for path in paths.values())
