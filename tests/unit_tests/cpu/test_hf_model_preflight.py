"""Real local HF configuration checks; no weights or network required."""

import json

from examples.hf_model_preflight import inspect_model


def test_native_text_config_does_not_claim_training_acceptance(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama"}), encoding="utf-8"
    )
    report = inspect_model(str(tmp_path), None, True)
    assert report["native_causal_lm_available"]
    assert report["native_causal_lm_class"] == "LlamaForCausalLM"
    assert report["training_verified"] is False
    assert report["npu_verified"] is False


def test_unknown_official_architecture_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unsupported_test_architecture"}), encoding="utf-8"
    )
    report = inspect_model(str(tmp_path), None, True)
    assert not report["native_causal_lm_available"]
    assert "ValueError" in report["error"]


def test_unreviewed_remote_code_is_not_executed(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "unreviewed_test_architecture",
                "auto_map": {"AutoConfig": "configuration_remote.RemoteConfig"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "configuration_remote.py").write_text(
        "raise RuntimeError('remote code executed')", encoding="utf-8"
    )
    report = inspect_model(str(tmp_path), None, True)
    assert not report["native_causal_lm_available"]
    assert "ValueError" in report["error"]


def test_misnamed_model_directory_cannot_pass_native_mapping_check(tmp_path):
    path = tmp_path / "DeepSeek-V4.1-Flash"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    report = inspect_model(str(path), None, True, ("deepseek_v41",))
    assert not report["native_causal_lm_available"]
    assert report["model_type"] == "deepseek_v3"
    assert report["advertised_architectures"] == ["DeepseekV3ForCausalLM"]
    assert "Model identity mismatch" in report["error"]


def test_expected_identity_does_not_infer_model_type_from_directory(tmp_path):
    path = tmp_path / "llama"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    report = inspect_model(str(path), None, True, ("llama",))
    assert report["model_type"] is None
    assert "Model identity mismatch" in report["error"]


def test_nested_text_config_is_used_for_identity(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe_text"},
            }
        ),
        encoding="utf-8",
    )
    report = inspect_model(str(tmp_path), None, True, ("qwen3_5_moe_text",))
    assert report["declared_model_type"] == "qwen3_5_moe"
    assert report["model_type"] == "qwen3_5_moe_text"
    assert report["native_causal_lm_available"]


def test_invalid_nested_config_returns_diagnostic(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "text_config": None}),
        encoding="utf-8",
    )
    report = inspect_model(str(tmp_path), None, True)
    assert not report["native_causal_lm_available"]
    assert "text_config must be a configuration object" in report["error"]


def test_top_level_identity_rejects_same_text_type_from_another_model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "other_family", "text_config": {"model_type": "llama"}}
        ),
        encoding="utf-8",
    )
    report = inspect_model(str(tmp_path), None, True, ("llama",), "target_family")
    assert "expected top-level" in report["error"]
    assert not report["native_causal_lm_available"]
