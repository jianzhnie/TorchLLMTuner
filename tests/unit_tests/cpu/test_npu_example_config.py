import pytest

from examples.train_qwen3_8b_npu import qwen3_8b_npu_config
from examples.train_qwen3_30b_npu import qwen3_30b_npu_config


@pytest.fixture(autouse=True)
def example_files(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text("{}\n")
    monkeypatch.setenv("LLMTUNER_QWEN3_8B_PATH", str(tmp_path))
    monkeypatch.setenv("LLMTUNER_QWEN3_30B_PATH", str(tmp_path))
    monkeypatch.setenv("LLMTUNER_DATASET_PATH", str(dataset_path))


@pytest.mark.parametrize(
    ("prompt_field", "response_field"),
    [("problem", "solution"), ("question", "answer")],
)
def test_npu_example_uses_configured_dataset_fields(
    monkeypatch, prompt_field, response_field
):
    monkeypatch.setenv("LLMTUNER_PROMPT_FIELD", prompt_field)
    monkeypatch.setenv("LLMTUNER_RESPONSE_FIELD", response_field)
    config = qwen3_8b_npu_config()
    assert config.training.dataloader_config.prompt_field == prompt_field
    assert config.training.dataloader_config.response_field == response_field


def test_npu_example_preserves_default_dataset_fields(monkeypatch):
    monkeypatch.delenv("LLMTUNER_PROMPT_FIELD", raising=False)
    monkeypatch.delenv("LLMTUNER_RESPONSE_FIELD", raising=False)
    config = qwen3_8b_npu_config()
    assert config.training.dataloader_config.prompt_field == "problem"
    assert config.training.dataloader_config.response_field == "solution"


@pytest.mark.parametrize(
    ("offload", "pin_memory", "implementation"),
    [("0", "1", "fused"), ("1", "0", "for-loop"), ("1", "1", "for-loop")],
)
def test_30b_offload_selects_cpu_optimizer(
    monkeypatch, offload, pin_memory, implementation
):
    monkeypatch.setenv("LLMTUNER_FSDP_CPU_OFFLOAD", offload)
    monkeypatch.setenv("LLMTUNER_FSDP_CPU_OFFLOAD_PIN_MEMORY", pin_memory)
    config = qwen3_30b_npu_config()
    assert config.parallel.fsdp_config.fsdp_cpu_offload == (offload == "1")
    assert config.parallel.fsdp_config.fsdp_cpu_offload_pin_memory == (
        pin_memory == "1"
    )
    assert config.optimizer.implementation == implementation
    assert config.training.checkpoint_config.last_save_model_only is False


def test_30b_example_configures_hsdp_replication(monkeypatch):
    monkeypatch.setenv("LLMTUNER_DATA_PARALLEL_REPLICATE_SIZE", "2")
    config = qwen3_30b_npu_config()
    assert config.parallel.data_parallel_replicate_size == 2
    assert config.parallel.data_parallel_shard_size == -1


def test_30b_example_configures_dp_shard8_ep8(monkeypatch):
    monkeypatch.setenv("LLMTUNER_DATA_PARALLEL_SHARD_SIZE", "8")
    monkeypatch.setenv("LLMTUNER_EXPERT_PARALLEL_SIZE", "8")
    config = qwen3_30b_npu_config()
    assert config.parallel.data_parallel_shard_size == 8
    assert config.parallel.expert_parallel_size == 8
    assert config.parallel.data_parallel_replicate_size == 1
    assert config.parallel.tensor_parallel_size == 1


def test_npu_example_configures_checkpoint_resume(monkeypatch):
    monkeypatch.setenv("LLMTUNER_CHECKPOINT_INTERVAL", "5")
    monkeypatch.setenv("LLMTUNER_CHECKPOINT_LOAD_STEP", "5")
    config = qwen3_8b_npu_config()
    assert config.training.checkpoint_config.interval == 5
    assert config.training.checkpoint_config.load_step == 5


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_npu_example_configures_tensorboard(monkeypatch, enabled):
    monkeypatch.setenv("LLMTUNER_ENABLE_TENSORBOARD", enabled)
    config = qwen3_8b_npu_config()
    assert config.training.metrics_config.enable_tensorboard == (enabled == "1")


def test_npu_example_disables_tensorboard_by_default(monkeypatch):
    monkeypatch.delenv("LLMTUNER_ENABLE_TENSORBOARD", raising=False)
    config = qwen3_8b_npu_config()
    assert not config.training.metrics_config.enable_tensorboard


def test_30b_example_uses_target_model_and_preserves_training_state(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("LLMTUNER_DUMP_FOLDER", raising=False)
    monkeypatch.delenv("LLMTUNER_PROMPT_FIELD", raising=False)
    monkeypatch.delenv("LLMTUNER_RESPONSE_FIELD", raising=False)
    config = qwen3_30b_npu_config()
    assert config.model.model_name_or_path == str(tmp_path)
    assert config.training.checkpoint_config.initial_load_path == str(tmp_path)
    assert not config.training.checkpoint_config.last_save_model_only
    assert config.training.dump_folder.endswith("outputs/qwen3-30b-npu")
    assert config.parallel.data_parallel_shard_size == -1
    assert config.training.dataloader_config.prompt_field == "question"
    assert config.training.dataloader_config.response_field == "solution"


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_30b_example_selects_global_context_rows(monkeypatch, enabled):
    monkeypatch.setenv("LLMTUNER_GLOBAL_CONTEXT_ROWS", enabled)
    config = qwen3_30b_npu_config()
    assert config.dataloader.global_context_rows == (enabled == "1")
    assert config.dataloader.attention_mask_type() == (
        "block_causal" if enabled == "1" else "causal"
    )


@pytest.mark.parametrize(("tp", "ep"), [(1, 1), (2, 1), (1, 8), (2, 4)])
def test_30b_parallel_defaults_select_sharded_hf_loading(monkeypatch, tp, ep):
    monkeypatch.setenv("LLMTUNER_TENSOR_PARALLEL_SIZE", str(tp))
    monkeypatch.setenv("LLMTUNER_EXPERT_PARALLEL_SIZE", str(ep))
    monkeypatch.delenv("LLMTUNER_SHARDED_TENSOR_CHECKPOINT", raising=False)
    monkeypatch.delenv("LLMTUNER_SHARDED_EXPERT_CHECKPOINT", raising=False)
    checkpoint = qwen3_30b_npu_config().training.checkpoint_config
    assert checkpoint.sharded_expert_parallel == (ep > 1)
    assert checkpoint.sharded_tensor_parallel == (tp > 1)


def test_30b_preserves_explicit_checkpoint_layout(monkeypatch):
    monkeypatch.setenv("LLMTUNER_EXPERT_PARALLEL_SIZE", "4")
    monkeypatch.setenv("LLMTUNER_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("LLMTUNER_SHARDED_EXPERT_CHECKPOINT", "0")
    monkeypatch.setenv("LLMTUNER_SHARDED_TENSOR_CHECKPOINT", "0")
    checkpoint = qwen3_30b_npu_config().training.checkpoint_config
    assert not checkpoint.sharded_expert_parallel
    assert not checkpoint.sharded_tensor_parallel
