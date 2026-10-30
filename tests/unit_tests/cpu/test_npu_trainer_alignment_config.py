from dataclasses import dataclass

import pytest
import torch

from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    _disable_moe_balancing,
    alignment_runtime_manifest,
    baseline_allow_hf32,
    baseline_cli_args,
    baseline_config_snapshot,
    baseline_learning_rate,
    build_candidate_30b_config,
    call_after_load,
    construction_dtype,
    debug_candidate_config,
    effective_forward_positions,
    expected_dtensor_shard,
    input_fingerprints,
    iter_local_state,
    map_candidate_to_canonical_hf,
    map_capture_to_candidate_hf,
    mesh_manifest,
    optimizer_manifest,
    selected_model,
    summed_cross_entropy,
    validate_baseline_30b_dict,
    validate_capture_runtime,
)


@pytest.mark.parametrize("runtime", [{}, {"diagnostic_timeout_patch": True}])
def test_formal_capture_rejects_missing_or_diagnostic_provenance(runtime):
    with pytest.raises(ValueError, match="unpatched runtime provenance"):
        validate_capture_runtime({"runtime": runtime})


def test_formal_capture_accepts_explicit_unpatched_provenance():
    validate_capture_runtime({"runtime": {"diagnostic_timeout_patch": False}})


def test_runtime_manifest_does_not_require_or_patch_set_timeout(monkeypatch):
    import torch.distributed as dist

    monkeypatch.delattr(dist, "set_timeout", raising=False)
    manifest = alignment_runtime_manifest((__name__,))
    assert manifest["diagnostic_timeout_patch"] is False
    assert not hasattr(dist, "set_timeout")


def test_baseline_snapshot_retains_model_arithmetic():
    from types import SimpleNamespace

    class ScaledDotProductAttention:
        class Config:
            pass

    config = SimpleNamespace(
        to_dict=lambda: {"model_spec": {"name": "qwen3", "flavor": "30B-A3B"}},
        model_spec=SimpleNamespace(
            model=SimpleNamespace(
                to_dict=lambda: {"layers": [{"moe": {"aux_loss": None}}]},
                layers=[
                    SimpleNamespace(
                        attention=SimpleNamespace(
                            inner_attention=ScaledDotProductAttention.Config(),
                            mask_type="causal",
                        )
                    )
                ],
            )
        ),
    )
    snapshot = baseline_config_snapshot(config)
    assert snapshot["model_spec"]["model"]["layers"][0]["moe"]["aux_loss"] is None
    assert snapshot["model_spec"]["flavor"] == "30B-A3B"
    assert snapshot["alignment_attention"] == {"backend": "sdpa", "mask_type": "causal"}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_baseline_hf32_reads_current_and_extension_layout(enabled, nested):
    training = {"allow_hf32": enabled}
    if nested:
        training = {"extension": training}
    assert baseline_allow_hf32({"training": training}) is enabled


@pytest.mark.parametrize(
    "training",
    [{}, {"allow_hf32": 1}, {"allow_hf32": True, "extension": {"allow_hf32": False}}],
)
def test_baseline_hf32_rejects_missing_invalid_or_conflicting_settings(training):
    with pytest.raises(ValueError, match="allow_hf32"):
        baseline_allow_hf32({"training": training})


@pytest.mark.parametrize("key", ["lr", "lr/AdamW"])
def test_baseline_learning_rate_reads_both_metric_layouts(key):
    assert baseline_learning_rate({key: 1e-5}) == 1e-5


@pytest.mark.parametrize("metrics", [{}, {"lr": 1e-5, "lr/AdamW": 2e-5}])
def test_baseline_learning_rate_rejects_missing_or_conflicting_values(metrics):
    with pytest.raises(ValueError, match="learning rate"):
        baseline_learning_rate(metrics)


def test_optimizer_manifest_supports_scheduler_without_get_metrics():
    from types import SimpleNamespace

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    manifest = optimizer_manifest(
        SimpleNamespace(
            optimizers=SimpleNamespace(optimizers=[optimizer]), lr_schedulers=scheduler
        )
    )
    assert manifest["lr_metrics"] is None
    assert manifest["scheduler"]["last_epoch"] == 0
    assert manifest["optimizers"][0]["param_groups"][0]["lr"] == 1e-5


def test_mesh_manifest_reads_existing_process_groups(monkeypatch):
    from types import SimpleNamespace

    import torch.distributed as dist

    class Mesh:
        shape = (2, 2)
        mesh_dim_names = ("dp", "tp")
        mesh = torch.tensor([[0, 1], [2, 3]])

        def get_group(self, axis):
            return {"dp": "actual-dp-handle", "tp": "actual-tp-handle"}[axis]

    class TPView:
        mesh_dim_names = ("tp",)

        def get_group(self, axis):
            assert axis == "tp"
            return "actual-tp-handle"

    topology = SimpleNamespace(
        dp_replicate=1,
        dp_shard=2,
        tp=2,
        ep=1,
        pp=1,
        cp=1,
        world_size=4,
        _global_meshes={"dense": Mesh()},
        _single_axis_meshes={"tp": TPView()},
        _multi_axis_meshes={("dp", "tp"): Mesh()},
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(
        dist,
        "get_process_group_ranks",
        lambda group: {"actual-dp-handle": [1, 3], "actual-tp-handle": [2, 3]}[group],
    )
    manifest = mesh_manifest(topology)
    assert manifest["rank"] == 3
    assert manifest["meshes"]["dense"]["ranks"] == [[0, 1], [2, 3]]
    assert manifest["local_process_groups"]["dense"] == {
        "dp": [1, 3],
        "tp": [2, 3],
    }
    assert manifest["local_process_groups"]["tp"] == {"tp": [2, 3]}
    assert manifest["local_process_groups"]["dp/tp"] == {
        "dp": [1, 3],
        "tp": [2, 3],
    }


def test_runtime_manifest_identifies_loaded_source():
    import hashlib
    from pathlib import Path

    manifest = alignment_runtime_manifest((__name__,))
    source = manifest["sources"][__name__]
    assert source["path"] == str(Path(__file__).resolve())
    assert source["sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert manifest["versions"]["torch"]
    with pytest.raises(ValueError, match="not loaded"):
        alignment_runtime_manifest(("nonexistent_alignment_source_module",))


def test_default_debugmodel_selection_and_config_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("LLMTUNER_ALIGNMENT_MODEL", raising=False)
    argv = ["--module", "qwen3", "--config", "qwen3_debugmodel"]
    assert selected_model() is None
    assert baseline_cli_args(argv) == argv
    config = debug_candidate_config(
        {"optimizer": {"implementation": "for-loop"}}, tmp_path, 1, 1
    )
    assert config.model.model_name_or_path == "qwen3"
    assert config.model.vocab_size == 2048
    assert config.model.arch_overrides["tie_word_embeddings"] is True
    assert config.optimizer.learning_rate == 8e-4
    assert config.optimizer.lr_scheduler_config.warmup_steps == 2
    assert config.training.max_seq_len == 32
    assert construction_dtype(False) is None


def test_30b_requires_explicit_opt_in_and_uses_real_baseline_recipe():
    argv = ["--dump_folder", "/tmp/capture"]
    environ = {"LLMTUNER_ALIGNMENT_MODEL": QWEN3_30B}
    assert baseline_cli_args(argv, environ) == [
        "--module",
        "torchtitan_npu.models.qwen3.config_registry",
        "--config",
        "sft_qwen3_30ba3b_gsm8k",
        *argv,
    ]


def _hf_directory(tmp_path):
    (tmp_path / "config.json").write_text(
        """{
            "model_type": "qwen3_moe",
            "vocab_size": 17,
            "hidden_size": 8,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "tie_word_embeddings": false,
            "router_aux_loss_coef": 0.001
        }"""
    )
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    return tmp_path


def _baseline_30b(tmp_path):
    return {
        "alignment_attention": {"backend": "flex", "mask_type": "block_causal"},
        "model_spec": {
            "flavor": "30B-A3B",
            "model": {
                "layers": [{"moe": {"load_balance_coeff": None, "aux_loss": None}}]
            },
        },
        "hf_assets_path": str(_hf_directory(tmp_path)),
        "parallelism": {
            "data_parallel_replicate_degree": 1,
            "data_parallel_shard_degree": 8,
            "tensor_parallel_degree": 1,
            "expert_parallel_degree": 1,
            "pipeline_parallel_degree": 1,
            "context_parallel_degree": 1,
            "spmd_backend": "partial_dtensor",
        },
        "training": {
            "local_batch_size": 1,
            "global_batch_size": 8,
            "seq_len": 4096,
            "steps": 1,
            "dtype": "bfloat16",
            "mixed_precision_param": "bfloat16",
            "mixed_precision_reduce": "float32",
            "max_norm": 0.5,
        },
        "debug": {"seed": 42, "deterministic": True},
        "optimizer": {
            "implementation": "for-loop",
            "param_groups": [
                {
                    "pattern": ".*",
                    "optimizer_name": "AdamW",
                    "optimizer_kwargs": {
                        "lr": 2e-5,
                        "weight_decay": 0.01,
                        "betas": [0.8, 0.9],
                        "eps": 1e-7,
                    },
                }
            ],
        },
        "lr_scheduler": {
            "warmup_steps": 7,
            "total_steps": None,
            "decay_ratio": 0.0,
            "decay_type": "cosine",
            "min_lr_factor": 0.1,
        },
        "loss": {"global_vocab_size": 17},
        "dataloader": {
            "load_dataset_kwargs": {"data_files": "/tmp/limo.jsonl", "split": "train"}
        },
    }


def test_30b_candidate_has_no_debugmodel_assumptions(tmp_path):
    baseline = _baseline_30b(tmp_path)
    config = build_candidate_30b_config(baseline, tmp_path / "output", 8)
    assert config.model.model_name_or_path == str(tmp_path)
    assert config.model.vocab_size == 17
    assert config.model.arch_overrides == {}
    assert config.model.experts_implementation == "torchtitan"
    assert config.model.norm_backend == "npu"
    assert config.model.qkv_projection == "separate"
    assert config.model.rotary_compute_dtype == "float32"
    assert config.model.attention_backend == "flex"
    assert config.optimizer.learning_rate == 2e-5
    assert config.optimizer.lr_scheduler_config.warmup_steps == 7
    assert config.optimizer.lr_scheduler_config.decay_type == "cosine"
    assert config.parallel.data_parallel_replicate_size == 1
    assert config.parallel.data_parallel_shard_size == 8
    assert config.parallel.ep_config.router_aux_loss_coef == 0.0
    assert config.training.max_seq_len == 4096
    assert config.training.global_batch_size == 8
    assert config.training.activation_checkpoint_mode == "full"
    assert config.training.chunked_loss_num_chunks == 1
    assert config.training.checkpoint_config.load_only
    assert config.training.checkpoint_config.initial_load_in_hf
    assert config.training.checkpoint_config.initial_load_path == str(tmp_path)
    assert config.training.dataloader_config.dataset == "local_jsonl_sft"
    assert config.training.dataloader_config.dataset_path == "/tmp/limo.jsonl"
    assert config.training.max_norm == 0.5
    assert construction_dtype(True) is torch.bfloat16

    from llmtuner.models.hf.factory import build_model_config_for

    hf_config = build_model_config_for(config)
    assert hf_config.attn_mask_type == "block_causal"
    assert hf_config.rotary_compute_dtype == "float32"


def test_30b_candidate_preserves_current_baseline_causal_sdpa(tmp_path):
    baseline = _baseline_30b(tmp_path)
    baseline["alignment_attention"] = {"backend": "sdpa", "mask_type": "causal"}
    config = build_candidate_30b_config(baseline, tmp_path / "output", 8)
    assert config.model.attention_backend == "auto"
    assert config.training.dataloader_config.attention_mask_type() == "causal"


def test_30b_replay_rejects_unknown_attention_contract(tmp_path):
    baseline = _baseline_30b(tmp_path)
    del baseline["alignment_attention"]
    with pytest.raises(ValueError, match="attention contract"):
        build_candidate_30b_config(baseline, tmp_path / "output", 8)


def test_effective_positions_follow_implicit_sequential_rope():
    forward = {"args": (torch.zeros(2, 4, dtype=torch.int64),), "kwargs": {}}
    expected = torch.arange(4).expand(2, 4)
    torch.testing.assert_close(effective_forward_positions(forward), expected)
    forward["kwargs"]["positions"] = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 2]])
    assert effective_forward_positions(forward) is forward["kwargs"]["positions"]


def test_replay_rejects_enabled_baseline_balancing(tmp_path):
    baseline = _baseline_30b(tmp_path)
    baseline["model_spec"]["model"]["layers"][0]["moe"]["load_balance_coeff"] = 1e-3
    with pytest.raises(ValueError, match="was not disabled"):
        validate_baseline_30b_dict(baseline, 8)


def test_baseline_balancing_disable_is_recursive_and_recorded():
    @dataclass
    class MoE:
        load_balance_coeff: float | None
        aux_loss: object | None

    @dataclass
    class Model:
        layers: list[MoE]

    model = Model([MoE(1e-3, object()), MoE(None, None)])
    disabled = _disable_moe_balancing(model)
    assert len(disabled) == 2
    assert model.layers[0].load_balance_coeff is None
    assert model.layers[0].aux_loss is None
    assert disabled[0]["previous"] == 1e-3


def test_initial_snapshot_runs_after_real_load_returns():
    events = []
    result = call_after_load(
        lambda step: events.append(("load", step)) or True,
        lambda: events.append(("snapshot", 0)),
        -1,
    )
    assert result is True
    assert events == [("load", -1), ("snapshot", 0)]


def test_aligned_cross_entropy_uses_actual_vocab_without_shifting():
    logits = torch.tensor([[[1.0, 0.5, -0.5, 2.0], [0.25, 1.5, 0.0, -1.0]]])
    labels = torch.tensor([[3, 1]])
    expected = -logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).sum()
    actual = summed_cross_entropy(logits, labels)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_native_fused_hf_experts_round_trip_canonical_names():
    canonical = {"lm_head.weight": torch.ones(1, 2)}
    prefix = "model.layers.0.mlp.experts."
    for expert in range(2):
        for role in ("gate", "up"):
            canonical[f"{prefix}{expert}.{role}_proj.weight"] = torch.full(
                (3, 2), float(expert * 10 + len(canonical))
            )
        canonical[f"{prefix}{expert}.down_proj.weight"] = torch.full(
            (2, 3), float(expert + 20)
        )
    candidate = dict(canonical)
    candidate_keys = {
        "lm_head.weight",
        f"{prefix}gate_up_proj",
        f"{prefix}down_proj",
    }
    mapped = map_capture_to_candidate_hf(candidate, candidate_keys, 2)
    assert set(mapped) == candidate_keys
    assert mapped[f"{prefix}gate_up_proj"].shape == (2, 6, 2)
    assert mapped[f"{prefix}down_proj"].shape == (2, 2, 3)
    restored = map_candidate_to_canonical_hf(mapped)
    assert set(restored) == set(canonical)
    for name, expected in canonical.items():
        torch.testing.assert_close(restored[name], expected, atol=0, rtol=0)


def test_input_fingerprints_include_scalar_global_valid_tokens():
    batches = [
        {
            "kwargs": {
                "input_dict": {"input": torch.tensor([1, 2])},
                "labels": torch.tensor([2, 3]),
                "global_valid_tokens": torch.tensor(2),
            }
        }
    ]
    forwards = [
        {
            "args": (torch.tensor([1, 2]),),
            "kwargs": {"positions": torch.tensor([0, 1])},
        }
    ]
    result = input_fingerprints(batches, forwards)
    assert result[0]["global_valid_tokens"]["shape"] == []
    assert result[0]["global_valid_tokens"]["dtype"] == "torch.int64"
    assert len(result[0]["global_valid_tokens"]["sha256"]) == 64


def test_expected_dtensor_shard_matches_local_coordinate() -> None:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    expected = torch.arange(24).reshape(8, 3)
    for rank in (0, 1):
        mesh = DeviceMesh(
            "cpu", [0, 1], mesh_dim_names=("dp",), _init_backend=False, _rank=rank
        )
        local = expected.chunk(2, dim=0)[rank]
        value = DTensor.from_local(local, mesh, [Shard(0)])
        actual = expected_dtensor_shard(value, expected)
        torch.testing.assert_close(actual, local, atol=0, rtol=0)
        [(name, actual_value, actual_expected)] = list(
            iter_local_state({"weight": value}, {"weight": expected})
        )
        assert name == "weight"
        torch.testing.assert_close(actual_value, local, atol=0, rtol=0)
        torch.testing.assert_close(actual_expected, local, atol=0, rtol=0)


def test_iter_local_state_rejects_incomplete_baseline() -> None:
    with pytest.raises(ValueError, match="unexpected=\\['bias'\\]"):
        list(
            iter_local_state(
                {"weight": torch.ones(2), "bias": torch.zeros(2)},
                {"weight": torch.ones(2)},
            )
        )


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_expected_dtensor_shard_handles_empty_ownership(rows) -> None:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    expected = torch.arange(rows * 3).reshape(rows, 3)
    chunks = expected.chunk(4, dim=0)
    for rank in range(4):
        mesh = DeviceMesh(
            "cpu",
            list(range(4)),
            mesh_dim_names=("dp",),
            _init_backend=False,
            _rank=rank,
        )
        local = chunks[rank] if rank < len(chunks) else expected.narrow(0, rows, 0)
        value = DTensor.from_local(
            local, mesh, [Shard(0)], shape=expected.shape, stride=expected.stride()
        )
        actual = expected_dtensor_shard(value, expected)
        torch.testing.assert_close(actual, local, atol=0, rtol=0)


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_alignment_steps_rejects_invalid_update_counts(value):
    from tests.integration_tests.npu_trainer_alignment_config import alignment_steps

    with pytest.raises(ValueError, match="positive integer"):
        alignment_steps({"LLMTUNER_ALIGNMENT_STEPS": value})


def test_alignment_steps_preserves_default_and_allows_multistep():
    from tests.integration_tests.npu_trainer_alignment_config import alignment_steps

    assert alignment_steps({}) == 1
    assert alignment_steps({"LLMTUNER_ALIGNMENT_STEPS": "100"}) == 100


@pytest.mark.parametrize("steps", [1, 100])
def test_30b_multistep_capture_reaches_candidate(tmp_path, steps):
    baseline = _baseline_30b(tmp_path)
    baseline["training"]["steps"] = steps
    config = build_candidate_30b_config(baseline, tmp_path / "output", 8)
    assert config.training.steps == steps


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_30b_capture_rejects_invalid_update_counts(tmp_path, steps):
    baseline = _baseline_30b(tmp_path)
    baseline["training"]["steps"] = steps
    with pytest.raises(ValueError, match="positive integer"):
        validate_baseline_30b_dict(baseline, 8)


@pytest.mark.parametrize(
    "settings", [{}, {"matmul": True}, {"matmul": False, "conv": True, "aclnn": True}]
)
def test_capture_rejects_missing_or_conflicting_actual_hf32(settings):
    from tests.integration_tests.npu_trainer_alignment_config import (
        validate_capture_hf32,
    )

    with pytest.raises(ValueError):
        validate_capture_hf32({"runtime": {"npu_allow_hf32": settings}}, True)


def test_capture_accepts_matching_actual_hf32():
    from tests.integration_tests.npu_trainer_alignment_config import (
        validate_capture_hf32,
    )

    validate_capture_hf32(
        {
            "runtime": {
                "npu_allow_hf32": dict.fromkeys(("matmul", "conv", "aclnn"), True)
            }
        },
        True,
    )


@pytest.mark.parametrize("value", ["0", "1"])
def test_baseline_hf32_override_updates_both_config_layouts(value):
    from types import SimpleNamespace

    from tests.integration_tests.npu_trainer_alignment_config import (
        configure_baseline_hf32,
    )

    training = SimpleNamespace(
        allow_hf32=True, extension=SimpleNamespace(allow_hf32=True)
    )
    configure_baseline_hf32(training, {"LLMTUNER_ALIGNMENT_ALLOW_HF32": value})
    assert training.allow_hf32 is (value == "1")
    assert training.extension.allow_hf32 is (value == "1")


@pytest.mark.parametrize("value", ["false", "", "2"])
def test_baseline_hf32_override_rejects_invalid_values(value):
    from types import SimpleNamespace

    from tests.integration_tests.npu_trainer_alignment_config import (
        configure_baseline_hf32,
    )

    with pytest.raises(ValueError, match="must be 0 or 1"):
        configure_baseline_hf32(
            SimpleNamespace(allow_hf32=True), {"LLMTUNER_ALIGNMENT_ALLOW_HF32": value}
        )


@pytest.mark.parametrize(
    "layout,dp,tp,ep",
    [
        ("dp8", 8, 1, 1),
        ("dp4-tp2", 4, 2, 1),
        ("tp2-ep4", 4, 2, 4),
        ("dp8-ep8", 8, 1, 8),
    ],
)
def test_target_layout_replay_preserves_topology_and_batch(
    tmp_path, layout, dp, tp, ep
):
    from tests.integration_tests.npu_trainer_alignment_config import target_parallelism

    baseline = _baseline_30b(tmp_path)
    topology = target_parallelism({"LLMTUNER_ALIGNMENT_LAYOUT": layout})
    baseline["parallelism"].update(topology)
    baseline["training"]["local_batch_size"] = 8 // dp
    validate_baseline_30b_dict(baseline, 8)
    config = build_candidate_30b_config(baseline, tmp_path / "run", 8)
    assert config.parallel.data_parallel_shard_size == dp
    assert config.parallel.tensor_parallel_size == tp
    assert config.parallel.expert_parallel_size == ep
    assert config.training.global_batch_size == 8
    assert config.checkpoint.sharded_tensor_parallel == (tp > 1)
    assert config.checkpoint.sharded_expert_parallel == (ep > 1)


def test_native_ep_reference_mapping_keeps_all_experts_and_router():
    source = {}
    for expert in range(4):
        for index, role in enumerate(("gate", "up", "down")):
            source[f"model.layers.0.mlp.experts.{expert}.{role}_proj.weight"] = (
                torch.full((2, 3), expert * 10 + index)
            )
    source["model.layers.0.mlp.gate.weight"] = torch.ones(4, 3)
    prefix = "model.layers.0.mlp."
    keys = {
        prefix + "routed_experts.inner_experts." + r
        for r in ("w1_EFD", "w3_EFD", "w2_EDF")
    }
    keys.add(prefix + "router.gate.weight")
    result = map_capture_to_candidate_hf(source, keys, 4)
    assert result.keys() == keys
    for index, role in enumerate(("w1_EFD", "w3_EFD", "w2_EDF")):
        value = result[prefix + "routed_experts.inner_experts." + role]
        assert value.shape == (4, 2, 3)
        assert value[:, 0, 0].tolist() == [expert * 10 + index for expert in range(4)]


def test_reference_localization_slices_packed_tp_and_ep_without_mutation(monkeypatch):
    import torch.distributed as dist
    from torch import nn

    from llmtuner.parallel.expert_parallel.checkpointing import EP_GROUP_ATTR
    from tests.integration_tests.npu_trainer_alignment_config import (
        localize_parallel_reference,
    )

    model = nn.Module()
    model.experts = nn.Module()
    model.experts.weight = nn.Parameter(torch.zeros(2, 4, 3))
    setattr(model.experts, EP_GROUP_ATTR, "ep")
    model._tp_checkpoint_layouts = {"experts.weight": (1, True, "tp")}
    monkeypatch.setattr(dist, "get_rank", lambda group: {"tp": 1, "ep": 0}[group])
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    full = torch.arange(96).reshape(4, 8, 3)
    state = {"experts.weight": full}
    result = localize_parallel_reference(state, model)
    expected = torch.cat((full[:2, 2:4], full[:2, 6:8]), dim=1)
    torch.testing.assert_close(result["experts.weight"], expected)
    assert state["experts.weight"] is full
    assert torch.count_nonzero(model.experts.weight) == 0


def test_reference_localization_reads_wrapper_owned_tp_layout(monkeypatch):
    import torch.distributed as dist
    from torch import nn

    from tests.integration_tests.npu_trainer_alignment_config import (
        localize_parallel_reference,
    )

    wrapper = nn.Module()
    wrapper.model = nn.Module()
    wrapper.model.projection = nn.Linear(3, 4, bias=False)
    wrapper._tp_checkpoint_layouts = {"model.projection.weight": (0, False, "tp")}
    monkeypatch.setattr(dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    full = torch.arange(24).reshape(8, 3)
    result = localize_parallel_reference(
        {"projection.weight": full}, wrapper, parameter_prefix="model."
    )
    torch.testing.assert_close(result["projection.weight"], full[4:])
    assert result["projection.weight"].shape == wrapper.model.projection.weight.shape
