"""The training recipe preserves published geometry unless reducing layers."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
)

from examples.qwen3_5.train import build_argument_parser, build_config
from llmtuner.config import DataloaderConfig
from llmtuner.datasets.corpus_identity import corpus_data_identity
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.materialization import (
    initialize_random_hf_model,
    materialize_meta_model,
)
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.qwen3_5.linear_attention import validate_linear_attention_training


def recipe_args(tmp_path, layers=None):
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    AutoConfig.for_model("qwen3_5_moe").save_pretrained(model_path)
    return Namespace(
        model=str(model_path),
        data=str(tmp_path / "data.jsonl"),
        output=str(tmp_path / "run"),
        steps=100,
        batch=8,
        sequence_length=128,
        checkpoint_interval=50,
        load_step=-1,
        layers=layers,
    )


def test_recipe_preserves_geometry_and_random_initialization(tmp_path):
    args = recipe_args(tmp_path)
    original = AutoConfig.from_pretrained(args.model).text_config
    config = build_config(args)
    actual = build_model_config_for(config)
    for name in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_experts",
        "linear_key_head_dim",
        "layer_types",
    ):
        assert getattr(actual, name) == getattr(original, name)
    checkpoint = config.training.checkpoint_config
    assert checkpoint.initial_load_path is None
    assert checkpoint.last_save_model_only is False
    assert checkpoint.sharded_tensor_parallel is False


def test_recipe_layer_reduction_keeps_other_dimensions(tmp_path):
    args = recipe_args(tmp_path, layers=4)
    config = build_config(args)
    actual = build_model_config_for(config)
    original = AutoConfig.from_pretrained(args.model).text_config
    assert actual.num_hidden_layers == 4
    assert actual.layer_types == original.layer_types[:4]
    assert actual.hidden_size == original.hidden_size
    assert actual.num_experts == original.num_experts


@pytest.mark.parametrize("backend", ["native", "npu_triton"])
def test_recipe_propagates_explicit_linear_attention_backend(tmp_path, backend):
    args = recipe_args(tmp_path)
    args.linear_attention_backend = backend
    config = build_config(args)
    assert config.model.linear_attention_backend == backend
    assert build_model_config_for(config).linear_attention_backend == backend


@pytest.mark.parametrize("implementation", ["native", "pre_down"])
def test_recipe_preserves_explicit_expert_arithmetic_policy(tmp_path, implementation):
    args = recipe_args(tmp_path)
    args.experts_implementation = implementation
    config = build_config(args)
    assert config.model.experts_implementation == implementation
    assert build_model_config_for(config).experts_implementation == implementation
    assert config.model.router_compute_dtype is None


@pytest.mark.parametrize("policy", ["native", "split_qkv_fp32_norm"])
def test_recipe_gdn_policy_is_independent_of_recurrence_backend(tmp_path, policy):
    args = recipe_args(tmp_path)
    args.gdn_numerical_policy = policy
    config = build_config(args)
    actual = build_model_config_for(config)
    assert config.model.gdn_numerical_policy == actual.gdn_numerical_policy == policy
    assert config.model.linear_attention_backend == "native"
    assert actual.linear_attention_backend == "native"


def test_recipe_right_padding_requires_both_explicit_gdn_policies(tmp_path):
    args = recipe_args(tmp_path)
    args.gdn_allow_right_padding = True
    with pytest.raises(ConfigError, match="npu_triton"):
        build_config(args)
    args.linear_attention_backend = "npu_triton"
    args.gdn_numerical_policy = "split_qkv_fp32_norm"
    config = build_config(args)
    assert config.model.gdn_allow_right_padding is True
    assert build_model_config_for(config).gdn_allow_right_padding is True


def test_recipe_fp32_router_is_explicit_and_reaches_hf_config(tmp_path):
    args = recipe_args(tmp_path)
    assert build_config(args).model.router_compute_dtype is None
    args.router_compute_dtype = "float32"
    config = build_config(args)
    assert config.model.router_compute_dtype == "float32"
    assert build_model_config_for(config).router_compute_dtype == "float32"


@pytest.mark.parametrize("layers", [0, -1, 10000])
def test_recipe_rejects_invalid_layer_reduction(tmp_path, layers):
    with pytest.raises(ConfigError, match="--layers"):
        build_config(recipe_args(tmp_path, layers=layers))


def test_meta_materialization_initializes_delta_state_and_backward():
    architecture = AutoConfig.for_model(
        "qwen3_5_moe_text",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    with torch.device("meta"):
        model = HFTransformerModel(architecture)
    materialize_meta_model(model, torch.device("cpu"))
    initialize_random_hf_model(model.model)
    delta_layers = [m for m in model.modules() if hasattr(m, "dt_bias")]
    assert len(delta_layers) == 3
    for layer in delta_layers:
        torch.testing.assert_close(layer.dt_bias, torch.ones_like(layer.dt_bias))
        assert torch.isfinite(layer.A_log).all()
    logits = model.model(
        input_ids=torch.tensor([[1, 7, 3, 11]]), use_cache=False
    ).logits
    logits.float().square().mean().backward()
    assert torch.isfinite(logits).all()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


def test_recipe_selects_sharded_initialization(tmp_path):
    config = build_config(recipe_args(tmp_path))
    assert config.model.initialization_device == "meta"


def test_recipe_exposes_topology_without_changing_model_geometry(tmp_path):
    args = recipe_args(tmp_path)
    args.dp_shard = 4
    args.dp_replicate = 1
    args.tp = 2
    args.ep = 1
    args.pp = 1
    args.cp = 1
    config = build_config(args)
    assert config.parallel.data_parallel_shard_size == 4
    assert config.parallel.tensor_parallel_size == 2
    assert config.parallel.derive_dp(8) == 4
    assert config.model.arch_overrides["num_hidden_layers"] == 40


def test_recipe_rejects_unsupported_gdn_topology_before_training(tmp_path):
    args = recipe_args(tmp_path)
    args.linear_attention_backend = "npu_triton"
    args.tp = 2
    config = build_config(args)
    with pytest.raises(UnsupportedCombinationError, match="TP|parallel|FSDP"):
        validate_linear_attention_training(
            config,
            build_model_config_for(config),
            device=SimpleNamespace(type="npu"),
            parallel_dims=SimpleNamespace(dp_shard=4, dp_replicate=1),
        )


@pytest.mark.parametrize("mode", ["none", "full"])
def test_recipe_propagates_activation_checkpoint_mode(tmp_path, mode):
    args = recipe_args(tmp_path)
    args.ac = mode
    assert build_config(args).training.activation_checkpoint_mode == mode


def test_shared_cli_retains_explicit_numerical_and_topology_choices(tmp_path):
    model = recipe_args(tmp_path).model
    args = build_argument_parser().parse_args(
        [
            "--model",
            model,
            "--data",
            "data.jsonl",
            "--output",
            "run",
            "--linear-attention-backend",
            "npu_triton",
            "--gdn-numerical-policy",
            "split_qkv_fp32_norm",
            "--gdn-allow-right-padding",
            "--gdn-deterministic-reduction",
            "--router-compute-dtype",
            "float32",
            "--dp-shard",
            "8",
        ]
    )
    config = build_config(args)
    assert config.model.linear_attention_backend == "npu_triton"
    assert config.model.gdn_numerical_policy == "split_qkv_fp32_norm"
    assert config.model.gdn_allow_right_padding
    assert config.model.gdn_deterministic_reduction
    assert config.model.router_compute_dtype == "float32"
    assert config.parallel.data_parallel_shard_size == 8


def test_recipe_can_load_shared_random_hf_initial_state(tmp_path):
    args = recipe_args(tmp_path)
    args.initial_weights = str(tmp_path / "shared-initial-state")
    checkpoint = build_config(args).training.checkpoint_config
    assert checkpoint.initial_load_path == args.initial_weights
    assert checkpoint.initial_load_in_hf
    assert checkpoint.initial_load_model_only
    assert not checkpoint.last_save_model_only


@pytest.mark.parametrize("option,value", [("load_step", 50), ("layers", 4)])
def test_shared_initial_state_cannot_hide_resume_or_geometry_changes(
    tmp_path, option, value
):
    args = recipe_args(tmp_path)
    args.initial_weights = str(tmp_path / "shared-initial-state")
    setattr(args, option, value)
    with pytest.raises(ConfigError, match="step-zero snapshot"):
        build_config(args)


@pytest.mark.parametrize("backend", ["none", "aot_eager", "inductor"])
def test_recipe_compile_option_preserves_geometry(tmp_path, backend):
    args = recipe_args(tmp_path)
    args.compile_backend = backend
    config = build_config(args)
    assert config.training.compile is (backend != "none")
    assert config.training.compile_config.per_block is (backend != "none")
    assert config.model.arch_overrides["num_hidden_layers"] == 40
    if backend != "none":
        assert config.training.compile_config.backend == backend


def test_bf16_delta_decay_initialization_excludes_log_zero():
    config = AutoConfig.for_model("qwen3_5_moe_text")
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(config)
    # Exercise the real HF initializer on many BF16 samples without allocating
    # the published model. An unfixed BF16 uniform draw reliably contains zero.
    delta = next(m for m in model.modules() if isinstance(m, Qwen3_5MoeGatedDeltaNet))
    delta.num_v_heads = 8192
    delta.A_log = torch.nn.Parameter(torch.empty(8192, dtype=torch.bfloat16))
    delta.dt_bias = torch.nn.Parameter(torch.empty(8192, dtype=torch.bfloat16))
    initialize_random_hf_model(model)
    assert torch.isfinite(delta.A_log).all()
    assert delta.A_log.float().exp().min() > 0
    assert torch.count_nonzero(delta.A_log) > 8000


@pytest.mark.parametrize(
    "options",
    [
        None,
        {1: False},
        {"tokenize": True},
        {"add_generation_prompt": True},
        {"temperature": float("nan")},
        {"custom": object()},
    ],
)
def test_template_options_reject_ambiguous_rendering_controls(options):
    with pytest.raises(ConfigError, match="chat_template_kwargs"):
        DataloaderConfig(dataset="local_jsonl_sft", chat_template_kwargs=options)


def test_template_policy_changes_checkpoint_data_identity(tmp_path):
    args = recipe_args(tmp_path)
    config = build_config(args)
    (tmp_path / "data.jsonl").write_text('{"question":"q","solution":"a"}\n')
    config.dataloader.chat_template_kwargs = {}
    default_identity = corpus_data_identity(
        config, dataset="local_jsonl_sft", repeat=True
    )
    config.dataloader.chat_template_kwargs = {"enable_thinking": False}
    explicit_identity = corpus_data_identity(
        config, dataset="local_jsonl_sft", repeat=True
    )
    assert explicit_identity != default_identity
    assert corpus_data_identity(config, dataset="local_jsonl_sft", repeat=True) == (
        explicit_identity
    )
