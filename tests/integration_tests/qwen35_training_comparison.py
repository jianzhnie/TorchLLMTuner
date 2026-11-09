"""CPU comparator for actual Qwen3.5 candidate/reference 100-step artifacts.

Fixed asymmetric criterion: abs(candidate-reference) <= .005+.02*abs(candidate).
Reads one local layer shard at a time; no reference/Trainer/NPU imports.
Every-step scalars/batches/contracts plus declared full snapshots are mandatory.
Updates mean post-update weights, NOT deltas (candidate has no pre-update dump).
RNG is exact within each side's resume boundary, never across frameworks.
--candidate-resume/--reference-resume require strong model+Adam manifests on BOTH
sides; older reference artifacts without these are rejected.
Only trusted local torch.save artifacts should be supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

ATOL, RTOL = 0.005, 0.02
RANKS, STEPS = 8, 100
SNAPSHOTS = (1, 50, 100)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key}: {path}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def child(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe path {name}")
    target = (root / path).resolve()
    require(target.is_relative_to(root.resolve()), f"Escaping artifact {name}")
    return target


def info(value):
    require(isinstance(value, torch.Tensor), "Expected tensor")
    value = value.detach().cpu().contiguous()
    require(bool(torch.isfinite(value).all()), "Nonfinite tensor")
    return {
        "sha256": hashlib.sha256(
            memoryview(value.reshape(-1).view(torch.uint8).numpy())
        ).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nonzero": bool(value.count_nonzero()),
        "bytes": value.numel() * value.element_size(),
        "empty_local_shard": not bool(value.numel()),
    }


def batch_hash(value):
    def describe(item):
        if isinstance(item, torch.Tensor):
            return {"tensor": info(item)}
        if isinstance(item, dict):
            return {k: describe(item[k]) for k in sorted(item)}
        if isinstance(item, tuple | list):
            return {type(item).__name__: [describe(v) for v in item]}
        require(
            item is None or isinstance(item, str | bool | int | float),
            f"Unknown batch type {type(item)}",
        )
        return {type(item).__name__: item}

    return hashlib.sha256(
        json.dumps(describe(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def metric(candidate, reference, *, exact=False, empty=False):
    a, b = torch.as_tensor(candidate), torch.as_tensor(reference)
    require(a.shape == b.shape and a.dtype == b.dtype, "Shape/dtype mismatch")
    require(a.numel() or empty, "Empty numerical evidence")
    require(
        bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "Nonfinite comparison",
    )
    if not a.numel():
        return {"passed": True, "exact": True, "empty_local_shard": True, "elements": 0}
    same = torch.equal(a, b)
    # Chunk float64 comparisons to preserve BF16/FP32 input values and avoid
    # tensor-size dependent loss of precision in error summaries.
    failed, maximum, square = 0, 0.0, 0.0
    for x, y in zip(
        a.reshape(-1).split(1024 * 1024), b.reshape(-1).split(1024 * 1024), strict=True
    ):
        x, y = x.double(), y.double()
        error = (x - y).abs()
        failed += int((error > ATOL + RTOL * x.abs()).sum())
        maximum = max(maximum, float(error.max()))
        square += float(error.square().sum())
    return {
        "elements": a.numel(),
        "max_abs": maximum,
        "rms": math.sqrt(square / a.numel()),
        "failed_elements": failed,
        "exact": same,
        "passed": same if exact else failed == 0,
    }


def canonical(key, candidate):
    parts = [
        p
        for p in key.split(".")
        if p not in ("_checkpoint_wrapped_module", "_orig_mod")
    ]
    key = ".".join(parts)
    if candidate:
        require(key.startswith("model."), f"Unexpected candidate key {key}")
        key = key[len("model.") :]
    require(
        key.startswith("model.") or key == "lm_head.weight", f"Not native HF key {key}"
    )
    return key


def validate_descriptor(value, descriptor):
    actual = info(value)
    for key, expected in actual.items():
        require(descriptor.get(key) == expected, f"Tensor descriptor mismatch: {key}")
    return actual


def load_snapshot(root, step, kind, group, candidate, rank, expected_keys):
    folder = root / f"step-{step:05d}-{kind}"
    index = read_json(folder / "index.json")
    if candidate:
        require(
            index["kind"] == kind and index["step"] == step,
            "Snapshot identity mismatch",
        )
    else:
        require(index["format"] == "HF5.13 native text", "Reference format mismatch")
    expected_files = {f"{g}.pt" for g in ("outer", *map(str, range(40)))}
    tensor_files = {k for k, v in index["files"].items() if "tensors" in v}
    require(tensor_files == expected_files, "Missing/extra layer snapshot files")
    entry = index["files"][f"{group}.pt"]
    path = child(folder, f"{group}.pt")
    require(file_hash(path) == entry["sha256"], f"Corrupt snapshot {path}")
    tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(
        tensors.keys() == entry["tensors"].keys(),
        "Snapshot payload/index statekeys differ",
    )
    result = {}
    for key, tensor in tensors.items():
        native = canonical(key, candidate)
        require(
            native in expected_keys and native not in result,
            f"Unknown/duplicate statekey {native}",
        )
        layout = entry["tensors"][key]
        validate_descriptor(tensor, layout)
        shape = layout["global_shape"]
        require(
            shape and all(isinstance(d, int) and d > 0 for d in shape),
            "Bad global shape",
        )
        size = (shape[0] + 7) // 8
        offset = min(rank * size, shape[0])
        require(
            layout.get("shard_dim") == 0 and layout.get("offset") == offset,
            f"Unexpected/replicated shard layout: {native}",
        )
        require(
            list(tensor.shape) == [min(size, shape[0] - offset), *shape[1:]],
            "Shard shape mismatch",
        )
        if isinstance(expected_keys, dict):
            require(shape == expected_keys[native], "Shared checkpoint shape differs")
        require(
            tensor.dtype == torch.bfloat16,
            "Expected BF16 local parameter/gradient storage",
        )
        result[native] = (tensor, shape)
    return result


def verify_initial_proof(shared, index, proof, rank):
    require(
        proof["exact"] and set(proof["tensors"]) == set(index["weight_map"]),
        "Initial native HF key coverage differs",
    )
    for filename in set(index["weight_map"].values()):
        with safe_open(child(shared, filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                descriptor = proof["tensors"][key]
                view = f.get_slice(key)
                shape = view.get_shape()
                size = (shape[0] + 7) // 8
                offset = min(rank * size, shape[0])
                require(
                    descriptor["global_shape"] == shape
                    and descriptor["shard_dim"] == 0
                    and descriptor["offset"] == offset,
                    "Initial shard layout differs",
                )
                validate_descriptor(
                    view[offset : offset + min(size, shape[0] - offset)], descriptor
                )


def scalar(value):
    require(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"Invalid scalar {value}",
    )
    return torch.tensor(value, dtype=torch.float64)


def validate_source_evidence(root, contract, complete):
    if contract.get("source_contract_version", 1) != 2:
        return
    loaded = read_json(root / "loaded-source-identity.json")
    expected = {
        "source_contract_version": 2,
        "verified": True,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
    }
    require(
        loaded == complete.get("source_identity") == expected,
        "Missing/mismatched load/completion source verification",
    )
    require(
        complete.get("resolved_attention_backend") == "sdpa",
        "Completion backend changed",
    )


def validate_contract(contract, *, allow_legacy=False):
    version = contract.get("source_contract_version", 1)
    require(version in (1, 2), "Unknown source contract version")
    require(
        version == 2 or allow_legacy,
        "Legacy source coverage requires --allow-legacy-source-contract",
    )
    if version == 2:
        require(
            contract.get("resolved_attention_backend") == "sdpa",
            "Expected resolved SDPA",
        )
        require(
            bool(contract.get("hf_sources_sha256")), "Missing actual HF source hashes"
        )
        required = {
            "llmtuner/models/hf/attention_backend.py",
            "llmtuner/models/hf/padding_metadata.py",
            "llmtuner/parallel/expert_parallel/token_dispatch.py",
            "llmtuner/models/common/scatter_add.py",
        }
        require(
            required <= contract["numerical_sources_sha256"].keys(),
            "Incomplete source coverage",
        )
        for digest in (
            *contract["hf_sources_sha256"].values(),
            *contract["numerical_sources_sha256"].values(),
        ):
            check_digest(digest)
    expected = {
        "horizon": 100,
        "world_size": 8,
        "geometry": "35B-A3B full 40 layers",
        "global_batch_size_per_accumulation": 8,
        "accumulation": 1,
        "sequence_length": 2048,
        "seed": 42,
        "max_num_documents": 1,
        "activation_checkpoint": "full",
        "max_norm": 1.0,
        "deterministic": True,
        "compile": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "scheduler": "constant; warmup=0; decay_ratio=0; horizon=100",
        "optimizer": {
            "name": "AdamW",
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "implementation": "for-loop",
        },
        "normalization": "sum token CE / global valid targets; FSDP gradient divide=1",
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"Unsupported/missing contract {key}")
    for key in (
        "shared_index_sha256",
        "data_sha256",
        "tokenizer_hashes",
        "numerical_sources_sha256",
        "versions",
        "linear_attention_backend",
        "gdn_numerical_policy",
        "gdn_allow_right_padding",
        "gdn_deterministic_reduction",
        "router_compute_dtype",
        "experts_implementation",
        "rotary_compute_dtype",
        "rotary_cache_dtype",
    ):
        require(key in contract, f"Missing declared contract {key}")
    require(
        bool(contract["tokenizer_hashes"])
        and bool(contract["numerical_sources_sha256"]),
        "Empty provenance",
    )


def validate_lr(record):
    rates = record["applied_lr"]
    require(
        isinstance(rates, dict) and rates and all(v == 3e-4 for v in rates.values()),
        "Missing/nonconstant applied LR",
    )


def validate_padding_positions(positions, padding, labels):
    """Single causal document: validate real prefix, preserve pad metadata."""
    length = labels.numel()
    valid_length = length
    if padding is not None:
        require(
            padding.dtype == torch.bool and padding.shape in ((length,), (1, length)),
            "Invalid right-padding metadata",
        )
        padding = padding.reshape(-1)
        require(
            not bool(padding[0]) and not bool((padding[:-1] & ~padding[1:]).any()),
            "Invalid right-padding metadata",
        )
        require(
            bool((labels.reshape(-1)[padding] == -100).all()),
            "Unmasked padding targets",
        )
        valid_length = int((~padding).sum())
    if positions is not None:
        require(
            positions.dtype in (torch.int32, torch.int64)
            and positions.shape in ((length,), (1, length))
            and not bool((positions < 0).any()),
            "Invalid positions",
        )
        require(
            torch.equal(
                positions.reshape(-1)[:valid_length], torch.arange(valid_length)
            ),
            "Packed/reset valid-prefix positions",
        )


def verify_batch(root, record, reference_record, step):
    require(
        len(record["batches"]) == len(reference_record["batches"]) == 1,
        "Expected exactly one microbatch",
    )
    c, r = record["batches"][0], reference_record["batches"][0]
    path = child(root, c["file"])
    require(
        path == root.resolve() / "batches" / f"step-{step:05d}-batch-000.pt",
        "Batch step/path mismatch",
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved.keys() == {"batch", "sha256"}, "Unknown batch payload keys")
    digest = batch_hash(saved["batch"])
    require(
        digest == saved["sha256"] == c["sha256"] == r["sha256"], "Batch hash mismatch"
    )
    batch = saved["batch"]
    require(
        {"input", "labels"}
        <= batch.keys()
        <= {"input", "labels", "positions", "padding_mask", "num_valid_tokens"},
        "Unknown batch fields",
    )
    ids, labels = batch["input"], batch["labels"]
    require(
        ids.dtype == labels.dtype == torch.int64
        and ids.numel() == labels.numel() == 2048,
        "Batch token geometry/dtype",
    )
    require(
        bool(ids.count_nonzero())
        and bool((ids.reshape(-1)[1:] != ids.reshape(-1)[:-1]).any())
        and not torch.equal(ids.reshape(-1), ids.reshape(-1).flip(0)),
        "Degenerate/symmetric token input",
    )
    require(bool(((ids >= 0) & (ids < 248320)).all()), "Invalid token IDs")
    require(
        bool(((labels == -100) | ((labels >= 0) & (labels < 248320))).all()),
        "Invalid labels",
    )
    valid = int((labels != -100).sum())
    if batch.get("num_valid_tokens") is not None:
        require(
            type(batch["num_valid_tokens"]) is int
            and batch["num_valid_tokens"] == valid,
            "Loader valid-token metadata differs",
        )
    require(
        valid > 0
        and valid
        == c["valid_targets"]
        == r["valid_targets"]
        == record["local_valid_targets"],
        "Valid-target mismatch",
    )
    require(
        r["step"] == step and reference_record["loader_cursor"] == step,
        "Reference loader cursor mismatch",
    )
    validate_padding_positions(
        batch.get("positions"), batch.get("padding_mask"), labels
    )
    return valid


def compare_logits(croot, rroot, c, r, step, scheduled, *, exact=False):
    require(
        len(c["logits"]) == len(r["logits"]) == 1, "Missing/extra logits observation"
    )
    a, b = c["logits"][0], r["logits"][0]
    require(
        a["shape"] == [2048, 248320]
        and b["shape"] == [1, 2048, 248320]
        and a["dtype"] in ("torch.bfloat16", "torch.float32"),
        "Unexpected full logits geometry/dtype",
    )
    require(a["dtype"] == b["dtype"], "Logits layout mismatch")
    require(
        a["nonzero"]
        and b["nonzero"]
        and not a["empty_local_shard"]
        and not b["empty_local_shard"],
        "Invalid logits evidence",
    )
    if step not in scheduled:
        require("file" not in a and "file" not in b, "Undeclared full logits dump")
        return {
            "compared": False,
            "reason": "Hashes observed; full logits not scheduled",
            "candidate_sha256": a["sha256"],
            "reference_sha256": b["sha256"],
        }
    x = torch.load(
        child(croot, a["file"]), map_location="cpu", weights_only=True, mmap=True
    )
    y = torch.load(
        child(rroot, b["file"]), map_location="cpu", weights_only=True, mmap=True
    )
    validate_descriptor(x, a)
    validate_descriptor(y, b)
    return {
        "compared": True,
        "layout_mapping": "reference singleton B removed; token/vocab axes unchanged",
        **metric(x, y.squeeze(0), exact=exact),
    }


def check_digest(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256 evidence",
    )


def validate_model_adam(
    model, expected_keys=None, *, step=50, parameter_count=34_660_610_688
):
    require(
        model.keys() == {"model", "adam", "param_groups"},
        "Model/Adam manifest fields differ",
    )
    names = set(model["model"])
    require(
        names and (expected_keys is None or names == expected_keys),
        "Model manifest statekey coverage differs",
    )
    owners = []
    for group in model["param_groups"]:
        check_digest(group["options_sha256"])
        owners.extend(f"{group['optimizer']}:{name}" for name in group["parameters"])
    require(
        len(owners) == len(set(owners)) == len(names)
        and set(owners) == set(model["adam"])
        and {key.split(":", 1)[1] for key in owners} == names,
        "Model/Adam statekey mismatch",
    )
    require(
        sum(math.prod(v["global_shape"]) for v in model["model"].values())
        == parameter_count,
        "Incomplete model geometry",
    )
    for descriptor in model["model"].values():
        check_digest(descriptor["sha256"])
        require(descriptor["dtype"] == "torch.bfloat16", "Non-BF16 model storage")
    for name, state in model["adam"].items():
        require(
            {"step", "exp_avg", "exp_avg_sq"} == state.keys(),
            "Missing/extra Adam moments",
        )
        step_descriptor = state["step"]
        require(
            step_descriptor["dtype"] in ("torch.float32", "torch.float64")
            and step_descriptor["shape"] == [],
            "Invalid Adam step tensor",
        )
        expected_step = info(
            torch.tensor(
                step,
                dtype=getattr(torch, step_descriptor["dtype"].removeprefix("torch.")),
            )
        )
        require(
            all(step_descriptor.get(k) == v for k, v in expected_step.items()),
            "Adam step is not the declared checkpoint step",
        )
        parameter = model["model"][name.split(":", 1)[1]]
        for key, descriptor in state.items():
            check_digest(descriptor.get("sha256", descriptor.get("value_sha256")))
            if key in ("exp_avg", "exp_avg_sq"):
                require(
                    descriptor["shape"] == parameter["shape"]
                    and descriptor["global_shape"] == parameter["global_shape"]
                    and descriptor["dtype"] in ("torch.bfloat16", "torch.float32"),
                    "Adam moment geometry/dtype differs",
                )


def validate_fingerprints(saved):
    require(
        set(saved) == {"lr_scheduler", "dataloader", "train_state", "rng_state"},
        "Missing checkpoint statekeys",
    )
    for value in saved.values():
        check_digest(value)


def candidate_manifest(root):
    saved = read_json(root / "checkpoint-step-00050.json")
    require(saved["step"] == 50, "Wrong saved boundary")
    require(
        saved["boundary"] == "Actual state_dict payloads during synchronous DCP save",
        "Unverified save boundary",
    )
    validate_fingerprints(saved["state_fingerprints"])
    validate_model_adam(saved["model_adam"])
    require(
        len(saved["model_adam"]["model"]) == 693,
        "Incomplete candidate model/Adam manifest",
    )
    check_digest(saved["metadata_sha256"])
    return saved


def reference_manifest(root, initial_keys, step=50):
    saved = read_json(root / f"step-{step:05d}-checkpoint-state.json")
    require(
        saved["schema"] == "qwen35-native-checkpoint-evidence-v1"
        and saved["step"] == saved["cursor"] == step,
        "Wrong reference checkpoint boundary/schema",
    )
    validate_fingerprints(saved["state_fingerprints"])
    validate_model_adam(saved["model_adam"], initial_keys, step=step)
    check_digest(saved["rng_sha256"])
    check_digest(saved["metadata_sha256"])
    return saved


def run(args, report):
    index = read_json(args.shared_hf / "model.safetensors.index.json")
    provenance_path = args.shared_hf / "initialization_provenance.json"
    require(
        index["metadata"]["total_size"] == 69_321_221_376
        and index["metadata"]["initialization_provenance_sha256"]
        == file_hash(provenance_path),
        "Shared initialization provenance differs",
    )
    provenance = read_json(provenance_path)
    require(
        provenance["config"]["text_config_sha256"]
        == file_hash(args.shared_hf / "config.json"),
        "Shared text config hash differs",
    )
    require(
        set(index["weight_map"].values()) == set(provenance["shards"]),
        "Shared checkpoint shard set differs",
    )
    for name, metadata in provenance["shards"].items():
        require(
            file_hash(child(args.shared_hf, name)) == metadata["sha256"],
            f"Shared checkpoint shard changed: {name}",
        )
    keys = {}
    for name in provenance["shards"]:
        with safe_open(child(args.shared_hf, name), framework="pt", device="cpu") as f:
            for key in f.keys():
                require(
                    key not in keys and index["weight_map"].get(key) == name,
                    "Duplicate/unknown shared checkpoint tensor",
                )
                view = f.get_slice(key)
                require(view.get_dtype() == "BF16", "Shared checkpoint dtype differs")
                keys[key] = view.get_shape()
    require(set(keys) == set(index["weight_map"]), "Shared checkpoint key set differs")
    require(
        sum(math.prod(shape) for shape in keys.values()) == 34_660_610_688,
        "Shared checkpoint parameter geometry differs",
    )
    require(len(keys) == 693, "Expected complete 693-tensor model")
    contract = None
    reference_contract = None
    rows = {}
    for root in (args.candidate, args.reference):
        require(
            {p.name for p in root.glob("rank-*")}
            == {f"rank-{r:05d}" for r in range(8)},
            "Missing/extra rank directories",
        )
    for rank in range(RANKS):
        croot = args.candidate / f"rank-{rank:05d}"
        rroot = args.reference / f"rank-{rank:05d}"
        c, r = read_json(croot / "run.json"), read_json(rroot / "run.json")
        validate_contract(
            c["contract"],
            allow_legacy=getattr(args, "allow_legacy_source_contract", False),
        )
        if reference_contract is None:
            reference_contract = r["contract"]
        require(
            r["contract"] == reference_contract,
            "Reference rank source contracts differ",
        )
        require(
            r["contract"]["provenance_sha256"] == file_hash(provenance_path)
            and r["contract"]["original_config_sha256"]
            == provenance["config"]["original_sha256"],
            "Reference initialization provenance differs",
        )
        if contract is None:
            contract = c["contract"]
        require(
            contract == c["contract"] == r["contract"]["candidate_contract"],
            "Rank/reference replay contract mismatch",
        )
        require(
            r["contract"]["shared_index_sha256"]
            == contract["shared_index_sha256"]
            == file_hash(args.shared_hf / "model.safetensors.index.json"),
            "Shared initialization differs",
        )
        require(
            c["resume"] is None and r["resume"] is None, "Continuous roots required"
        )
        require(
            c["snapshot_steps"] == list(SNAPSHOTS)
            and c["logits_steps"] == list(SNAPSHOTS)
            and r["snapshots"] == list(SNAPSHOTS),
            "Comparator requires declared snapshots/logits 1/50/100",
        )
        for root, side in ((croot, "candidate"), (rroot, "reference")):
            complete = read_json(root / "complete.json")
            require(complete["step"] == 100, "Incomplete run")
            if side == "candidate":
                validate_source_evidence(root, c["contract"], complete)
            require(
                complete["resume_step" if side == "candidate" else "resumed_from"] == 0,
                "Fresh run has a resume boundary",
            )
            require(
                complete["status"]
                == (
                    "candidate_complete_reference_pending"
                    if side == "candidate"
                    else "reference_complete_comparison_pending"
                ),
                "Unknown completion status",
            )
            expected = {f"step-{s:05d}.json" for s in range(1, 101)}
            actual = {
                p.name
                for p in root.glob("step-*.json")
                if p.name[5:10].isdigit() and p.name[10:] == ".json"
            }
            require(actual == expected, "Missing/extra step records")
        loaded = read_json(croot / "loaded.json")
        require(
            loaded["loaded"]
            and loaded["step"] == 0
            and loaded["scheduler"]["last_epoch"] == 0
            and loaded["initial_shards_exact"]
            and loaded["original_loader_checkpointed"],
            "Candidate initialization verification missing",
        )
        precision = loaded["runtime_precision"]
        require(
            precision["parameter_dtype"] == "torch.bfloat16"
            and precision["reduce_dtype"] == "torch.float32"
            and precision["cast_forward_inputs"] is False
            and precision["gradient_divide_factor"] == 1.0
            and precision["fsdp_states"] > 0,
            "Runtime precision contract differs",
        )
        initial = read_json(croot / "initial-shards-exact.json")
        require(
            initial["exact"]
            and initial["tensor_count"] == 693
            and set(initial["tensors"]) == set(keys),
            "Incomplete initial shard proof",
        )
        verify_initial_proof(args.shared_hf, index, initial, rank)
        rloaded = read_json(rroot / "loaded.json")
        require(
            rloaded["step"] == rloaded["cursor"] == 0
            and rloaded["rng_restored"] is False,
            "Reference fresh load contract",
        )
        saved = candidate_manifest(croot)
        require(
            {canonical(k, True) for k in saved["model_adam"]["model"]} == set(keys),
            "Candidate checkpoint native key coverage differs",
        )
        refinitial = read_json(rroot / "initial-shards-exact.json")
        require(
            rloaded["initial_shards_exact"] and refinitial,
            "Reference fresh exact initial proof missing",
        )
        require(
            sum(math.prod(v["global_shape"]) for v in refinitial.values())
            == 34_660_610_688,
            "Reference initial geometry incomplete",
        )
        for descriptor in refinitial.values():
            check_digest(descriptor["sha256"])
            require(
                descriptor["dtype"] == "torch.bfloat16",
                "Reference initial storage dtype differs",
            )
        refstate = reference_manifest(rroot, set(refinitial))
        reference_manifest(rroot, set(refinitial), step=100)
        require(
            refstate["cursor"] == 50
            and isinstance(refstate["rng_sha256"], str)
            and len(refstate["rng_sha256"]) == 64,
            "Reference saved loader/RNG missing",
        )
        rows[rank] = (
            croot,
            rroot,
            {},
            [],
            {
                "candidate_saved_state": saved["state_fingerprints"],
                "reference_loaded_rng": rloaded["rng_sha256"],
                "reference_saved_rng": refstate["rng_sha256"],
            },
        )
    report["contract"] = contract
    report["reference_contract"] = reference_contract
    report["rng_contract"] = {
        "rule": (
            "seed42; preserve own RNG on checkpoint/resume; "
            "no cross-framework RNG byte equality"
        ),
        "actual": {str(k): v[4] for k, v in rows.items()},
    }
    for step in range(1, 101):
        local = [], []
        cvalid, rvalid = local
        for cr, rr, metrics, _, _ in rows.values():
            c, r = (
                read_json(cr / f"step-{step:05d}.json"),
                read_json(rr / f"step-{step:05d}.json"),
            )
            require(c["step"] == r["step"] == step, "Step identity mismatch")
            validate_lr(c)
            validate_lr(r)
            require(
                c["preclip_hook_seen"]
                and not c["entry_source"]["changed_since_startup"],
                "Missing capture or source changed",
            )
            valid = verify_batch(cr, c, r, step)
            cvalid.append(valid)
            rvalid.append(r["batches"][0]["valid_targets"])
            require(
                c["global_valid_targets"] == r["global_valid_targets"] > 0,
                "Global target denominator mismatch",
            )
            loss = metric(scalar(c["metrics"]["loss"]), scalar(r["global_loss"]))
            norm = metric(
                scalar(c["metrics"]["grad_norm"]), scalar(r["grad_norm_preclip"])
            )
            require(
                float(c["metrics"]["loss"]) > 0
                and float(r["global_loss"]) > 0
                and float(c["metrics"]["grad_norm"]) > 0
                and float(r["grad_norm_preclip"]) > 0,
                "Zero/negative scalar evidence",
            )
            metrics[str(step)] = {
                "loss": loss,
                "gradnorm": norm,
                "logits": compare_logits(cr, rr, c, r, step, SNAPSHOTS),
            }
        require(sum(cvalid) == sum(rvalid), "Global batch targets differ")
        for cr, _, _, _, _ in rows.values():
            require(
                read_json(cr / f"step-{step:05d}.json")["global_valid_targets"]
                == sum(cvalid),
                "Incorrect global normalization denominator",
            )
    report["steps"] = {str(k): v[2] for k, v in rows.items()}
    report["snapshots"] = {}
    for rank, (cr, rr, _, _, _) in rows.items():
        report["snapshots"][str(rank)] = {}
        for step in SNAPSHOTS:
            for kind in ("gradients-preclip", "weights-after-update"):
                metrics = {}
                nonzero = [False, False]
                for group in ("outer", *map(str, range(40))):
                    a = load_snapshot(cr, step, kind, group, True, rank, keys)
                    b = load_snapshot(rr, step, kind, group, False, rank, keys)
                    require(a.keys() == b.keys(), "Canonical snapshot keys differ")
                    for key, (x, shape) in a.items():
                        y, other = b[key]
                        require(
                            shape == other and key not in metrics,
                            "Global shape/duplicate snapshot key",
                        )
                        nonzero[0] |= bool(x.count_nonzero())
                        nonzero[1] |= bool(y.count_nonzero())
                        metrics[key] = metric(x, y, empty=True)
                require(
                    metrics.keys() == keys.keys() and all(nonzero),
                    "Missing keys/all-zero snapshot",
                )
                report["snapshots"][str(rank)][f"{step}:{kind}"] = metrics
    report["limitations"] = [
        (
            "Every-step loss/globaltargets/LR/gradnorm; full logits/gradients/"
            "post-update weights only at 1/50/100."
        ),
        (
            "Candidate has no pre-update weights; "
            "cross-framework delta metrics unavailable."
        ),
        (
            "Reference initial and resume proofs are required; "
            "absent artifacts are not inferred from later observations."
        ),
    ]
    report["limitations"].append(
        "Candidate has no final step100 Adam manifest: resume evidence covers "
        "exact step50 restoration and subsequent exported observations, "
        "not unobserved final optimizer states. Reference native initial proof "
        "is an exporter attestation; candidate HF slices are independently hashed."
    )
    report["resume"] = (
        compare_resumes(args, rows, report)
        if args.candidate_resume
        else {
            "requested": False,
            "accepted": False,
            "reason": "Continuous-only comparison; no resume acceptance claim",
        }
    )

    def passed(value):
        if isinstance(value, dict):
            return all((v if k == "passed" else passed(v)) for k, v in value.items())
        if isinstance(value, list):
            return all(passed(v) for v in value)
        return True

    report["passed"] = (
        passed(report["steps"])
        and passed(report["snapshots"])
        and (not args.candidate_resume or report["resume"]["accepted"])
    )
    report["status"] = "PASS" if report["passed"] else "FAIL"
    report["acceptance_scope"] = (
        "100-step observations plus independent own-state resume evidence"
        if args.candidate_resume
        else "Continuous 100-step observations only"
    )


def verify_resume_records(root, old, side):
    expected = {f"step-{s:05d}.json" for s in range(51, 101)}
    actual = {
        p.name
        for p in root.glob("step-*.json")
        if p.name[5:10].isdigit() and p.name[10:] == ".json"
    }
    require(actual == expected, "Resume missing/extra step records")
    loaded = read_json(root / "loaded.json")
    require(loaded["step"] == 50, "Resume did not load step50")
    if side == "candidate":
        require(
            loaded["loaded"] and loaded["scheduler"]["last_epoch"] == 50,
            "Candidate resume scheduler/load differs",
        )
    for step in range(51, 101):
        a = read_json(root / f"step-{step:05d}.json")
        b = read_json(old / f"step-{step:05d}.json")
        fields = (
            "step",
            "global_valid_targets",
            "local_valid_targets",
            "local_loss_sum",
            "metrics",
            "applied_lr",
            "batches",
            "logits",
        )
        require(
            all(a[key] == b[key] for key in fields) if side == "candidate" else a == b,
            f"Resume record differs step{step}",
        )
        if side == "candidate":
            require(
                a["preclip_hook_seen"]
                and not a["entry_source"]["changed_since_startup"],
                "Resume capture/source invariant failed",
            )
            for record in a["batches"]:
                saved = torch.load(
                    child(root, record["file"]), map_location="cpu", weights_only=True
                )
                require(
                    batch_hash(saved["batch"]) == saved["sha256"] == record["sha256"],
                    "Resume batch payload changed",
                )
        for left, right in zip(a["logits"], b["logits"], strict=True):
            if step == 100:
                x = torch.load(
                    child(root, left["file"]),
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                y = torch.load(
                    child(old, right["file"]),
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                validate_descriptor(x, left)
                validate_descriptor(y, right)
                require(
                    metric(x, y, exact=True)["passed"],
                    "Resume logits payload not exact",
                )


def compare_resumes(args, rows, report):
    evidence = {}
    for rank, (cr, rr, _, _, _) in rows.items():
        cs = args.candidate_resume / cr.name
        rs = args.reference_resume / rr.name
        candidate_contract = read_json(cr / "run.json")["contract"]
        validate_source_evidence(
            cs, candidate_contract, read_json(cs / "complete.json")
        )
        cproof = read_json(cs / "resume-step50-exact.json")
        saved = candidate_manifest(cr)
        require(
            cproof["exact"]
            and cproof["before_data_iterator"]
            and cproof["baseline_sha256"]
            == file_hash(cr / "checkpoint-step-00050.json")
            and cproof["state_fingerprints"] == saved["state_fingerprints"],
            "Candidate resume state proof differs",
        )
        for root, old, side in ((cs, cr, "candidate"), (rs, rr, "reference")):
            run = read_json(root / "run.json")
            base = read_json(old / "run.json")
            require(
                run["contract"] == base["contract"] and run["resume"] is not None,
                "Resume contract mismatch",
            )
            require(
                read_json(root / "complete.json")["step"] == 100, "Resume incomplete"
            )
            complete = read_json(root / "complete.json")
            field = "resume_step" if side == "candidate" else "resumed_from"
            require(complete[field] == 50, "Resume completion boundary differs")
            boundary = (
                candidate_manifest(old)
                if side == "candidate"
                else read_json(old / "step-00050-checkpoint-state.json")
            )
            require(
                Path(run["resume"]).resolve() == Path(boundary["checkpoint"]).resolve(),
                "Resume is not this side's original step50 checkpoint",
            )
            verify_resume_records(root, old, side)
            # Compare payloads, not just inherited descriptors, after resume.
            for step in (100,):
                for kind in ("gradients-preclip", "weights-after-update"):
                    for group in ("outer", *map(str, range(40))):
                        a = load_snapshot(
                            root,
                            step,
                            kind,
                            group,
                            side == "candidate",
                            rank,
                            set(report["snapshots"][str(rank)]["100:" + kind]),
                        )
                        b = load_snapshot(
                            old, step, kind, group, side == "candidate", rank, set(a)
                        )
                        require(a.keys() == b.keys(), "Resume statekeys differ")
                        require(
                            all(torch.equal(a[k][0], b[k][0]) for k in a),
                            "Resume snapshot not exact",
                        )
        loaded = read_json(rs / "loaded.json")
        boundary = read_json(rr / "step-00050-checkpoint-state.json")
        require(
            loaded["step"] == loaded["cursor"] == 50
            and loaded["rng_restored"]
            and loaded["rng_sha256"] == boundary["rng_sha256"],
            "Reference resume RNG/cursor differs",
        )
        saved_ref = reference_manifest(rr, set(boundary["model_adam"]["model"]))
        restored = read_json(rs / "restored-state.json")
        require(
            restored.keys()
            == {"cursor", "rng_sha256", "model_adam", "state_fingerprints"},
            "Missing restored reference statekeys",
        )
        require(
            all(saved_ref[key] == value for key, value in restored.items()),
            "Restored reference model/Adam/scheduler/RNG/loader differs",
        )
        require(loaded["model_adam_restored_exact"], "Reference resume unverified")
        final_ref = reference_manifest(rr, set(boundary["model_adam"]["model"]), 100)
        resumed_final = reference_manifest(
            rs, set(boundary["model_adam"]["model"]), 100
        )
        for key in ("model_adam", "state_fingerprints", "cursor", "rng_sha256"):
            require(
                final_ref[key] == resumed_final[key],
                f"Reference resumed step100 state differs: {key}",
            )
        evidence[str(rank)] = {"candidate_exact": True, "reference_exact": True}

    return {
        "requested": True,
        "accepted": True,
        "boundary_evidence": evidence,
        "available_step51_100_records_and_step100_snapshots_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "reference", "shared-hf", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-source-contract",
        action="store_true",
        help="Allow old partial coverage; no v2 provenance claim",
    )
    parser.add_argument("--candidate-resume", type=Path)
    parser.add_argument("--reference-resume", type=Path)
    args = parser.parse_args()
    require(
        bool(args.candidate_resume) == bool(args.reference_resume),
        "Both independent resume roots required",
    )
    require(not args.output.exists(), "Output must be new")
    report = {
        "legacy_source_contract_allowed": args.allow_legacy_source_contract,
        "source_provenance_scope": (
            "legacy partial coverage allowed; no v2 provenance claim"
            if args.allow_legacy_source_contract
            else "v2 load/completion source verification"
        ),
        "status": "ERROR",
        "passed": False,
        "atol": ATOL,
        "rtol": RTOL,
        "asymmetry": "candidate is relative-tolerance baseline; never swap baseline",
        "scope": (
            "Actual 100-step observation comparison, not proof of unobserved states"
        ),
    }
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
