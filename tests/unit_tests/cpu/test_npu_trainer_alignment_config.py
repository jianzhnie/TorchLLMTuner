from dataclasses import dataclass

import pytest
import torch

from tests.integration_tests.npu_trainer_alignment_config import (
    QWEN3_30B,
    _disable_moe_balancing,
    baseline_cli_args,
    build_candidate_30b_config,
    call_after_load,
    construction_dtype,
    debug_candidate_config,
    expected_dtensor_shard,
    input_fingerprints,
    iter_local_state,
    map_candidate_to_canonical_hf,
    map_capture_to_candidate_hf,
    selected_model,
    summed_cross_entropy,
    validate_baseline_30b_dict,
)


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
        "qwen3",
        "--config",
        QWEN3_30B,
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
    assert config.model.norm_backend == "torch"
    assert config.model.qkv_projection == "fused"
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
