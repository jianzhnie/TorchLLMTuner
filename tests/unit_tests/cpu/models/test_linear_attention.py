"""CPU contracts for opt-in HF recurrence routing, never kernel acceptance."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf import linear_attention as adapter
from llmtuner.models.hf import padding_metadata
from llmtuner.models.hf.factory import apply_runtime_config, unwrap_text_config


def geometry(**overrides):
    values = dict(
        linear_attention_backend="npu_triton",
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attn_mask_type="causal",
    )
    return SimpleNamespace(**(values | overrides))


def training():
    return SimpleNamespace(
        model=ModelConfig(linear_attention_backend="npu_triton"),
        parallel=SimpleNamespace(
            tp_size=1,
            cp_size=1,
            ep_size=1,
            pp_size=1,
            fsdp_config=SimpleNamespace(fsdp_cpu_offload=False),
        ),
        training=SimpleNamespace(
            compile=False,
            activation_checkpoint_mode="full",
            activation_swap=False,
            global_batch_size=8,
            max_seq_len=2048,
        ),
        dataloader=SimpleNamespace(
            global_context_rows=False, attention_mask_type=lambda: "causal"
        ),
    )


def test_config_default_and_rejection():
    assert ModelConfig().linear_attention_backend == "native"
    with pytest.raises(ConfigError, match="linear_attention_backend"):
        ModelConfig(linear_attention_backend="automatic")


def test_runtime_config_reaches_composite_text():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            linear_attention_backend="npu_triton",
            rotary_compute_dtype=None,
            rotary_cache_dtype=None,
        ),
        dataloader=SimpleNamespace(attention_mask_type=lambda: "causal"),
        norm_backend="native",
        qkv_projection="separate",
        rotary_init_device="constructor",
        attention_backend="auto",
    )
    top = PretrainedConfig()
    top.text_config = PretrainedConfig()
    apply_runtime_config(top, cfg)
    assert unwrap_text_config(top).linear_attention_backend == "npu_triton"


def test_native_does_not_import_optional_backend(monkeypatch):
    def forbidden(*args):
        pytest.fail("native imported an optional backend")

    monkeypatch.setattr(adapter, "import_module", forbidden)
    model = nn.Module()
    model.config = SimpleNamespace(linear_attention_backend="native")
    adapter.apply_linear_attention_backend(model)


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "qwen3_5_text"},
        {"linear_key_head_dim": 64},
        {"linear_num_value_heads": 16},
        {"hidden_act": "relu"},
        {"rms_norm_eps": 1e-5},
        {"attn_mask_type": "block_causal"},
    ],
)
def test_unverified_geometry_is_rejected(override):
    with pytest.raises(UnsupportedCombinationError):
        adapter.validate_linear_attention_config(geometry(**override))


@pytest.mark.parametrize(
    "group,name,value",
    [
        ("parallel", "tp_size", 2),
        ("parallel", "cp_size", 2),
        ("parallel", "ep_size", 2),
        ("parallel", "pp_size", 2),
        ("training", "compile", True),
        ("training", "activation_checkpoint_mode", "selective"),
        ("training", "activation_swap", True),
        ("training", "global_batch_size", 16),
        ("training", "max_seq_len", 73),
        ("dataloader", "global_context_rows", True),
    ],
)
def test_training_rejects_unverified_combinations(group, name, value):
    cfg = training()
    setattr(getattr(cfg, group), name, value)
    with pytest.raises(UnsupportedCombinationError):
        adapter.validate_linear_attention_training(
            cfg,
            geometry(),
            device=SimpleNamespace(type="npu"),
            parallel_dims=SimpleNamespace(dp_shard=8, dp_replicate=1),
        )


def test_training_allows_eager_full_ac_fsdp_and_rejects_cpu(monkeypatch):
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)
    kwargs = dict(parallel_dims=SimpleNamespace(dp_shard=8, dp_replicate=1))
    adapter.validate_linear_attention_training(
        training(),
        geometry(),
        device=SimpleNamespace(type="npu"),
        **kwargs,
    )
    with pytest.raises(UnsupportedCombinationError, match="NPU"):
        adapter.validate_linear_attention_training(
            training(),
            geometry(),
            device=torch.device("cpu"),
            **kwargs,
        )


def test_chunk_bridge_preserves_inputs_and_output_contract(monkeypatch):
    inputs = [torch.randn(1, 64, 2, 128) for _ in range(5)]
    output = torch.randn(1, 64, 2, 128)

    def op(*actual):
        assert all(a is b for a, b in zip(actual, inputs, strict=True))
        return output

    monkeypatch.setattr(
        adapter, "import_module", lambda _: SimpleNamespace(gated_delta_rule=op)
    )
    actual, state = adapter.chunk_gated_delta_rule(
        *inputs,
        use_qk_l2norm_in_kernel=True,
    )
    assert actual is output and state is None


@pytest.mark.parametrize(
    "extra",
    [
        {"initial_state": torch.zeros(1)},
        {"output_final_state": True},
        {"cu_seqlens": torch.tensor([0, 64])},
        {"seq_idx": None},
        {"use_qk_l2norm_in_kernel": False},
    ],
)
def test_chunk_bridge_rejects_cache_packing_and_unknown_arguments(extra):
    kwargs = {"use_qk_l2norm_in_kernel": True} | extra
    with pytest.raises(UnsupportedCombinationError):
        adapter.chunk_gated_delta_rule(*[torch.empty(0)] * 5, **kwargs)


def test_batch_guard_rejects_rows_padding_and_resets():
    batch = {
        "input": torch.zeros(1, 64, dtype=torch.long),
        "labels": torch.zeros(1, 64, dtype=torch.long),
        "positions": torch.arange(64),
    }
    adapter.validate_linear_attention_batch(batch)
    for change in (
        {"input": torch.zeros(2, 64)},
        {"positions": torch.zeros(64)},
        {"padding_mask": torch.ones(64, dtype=torch.bool)},
        {"cu_seq_lens_q": None},
    ):
        with pytest.raises(UnsupportedCombinationError):
            adapter.validate_linear_attention_batch(batch | change)


def test_suffix_padding_checked_but_not_yet_enabled():
    tokens = torch.zeros(64, dtype=torch.long)
    for mask, message in (
        (torch.ones(63, dtype=torch.bool), "one entry"),
        (torch.zeros(64), "boolean"),
        (torch.arange(64) < 10, "suffix-only"),
        (torch.arange(64) >= 50, "awaits full-module"),
    ):
        with pytest.raises(UnsupportedCombinationError, match=message):
            adapter.validate_linear_attention_sequence(tokens, None, None, mask)
    adapter.validate_linear_attention_sequence(
        tokens, None, None, torch.zeros(64, dtype=torch.bool)
    )


def test_training_rejects_non_bf16_construction(monkeypatch):
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float32)
    with pytest.raises(UnsupportedCombinationError, match="BF16 parameter"):
        adapter.validate_linear_attention_training(
            training(),
            geometry(),
            device=SimpleNamespace(type="npu"),
            parallel_dims=SimpleNamespace(dp_shard=8, dp_replicate=1),
        )


@pytest.mark.parametrize("deterministic", [False, True])
def test_real_hf_module_only_recurrence_is_replaced(monkeypatch, deterministic):
    hf = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    config_module = pytest.importorskip(
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
    )
    config = config_module.Qwen3_5MoeTextConfig(
        hidden_size=2048,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    config.linear_attention_backend = "npu_triton"
    config.gdn_deterministic_reduction = deterministic
    config.attn_mask_type = "causal"
    with torch.device("meta"):
        model = nn.Module()
        model.config = config
        model.gdn = hf.Qwen3_5MoeGatedDeltaNet(config, 0)
    # Keep the explicitly validated native HF implementations in this test.
    model.gdn.norm = hf.Qwen3_5MoeRMSNormGated(128, device="meta")
    model.gdn.causal_conv1d_fn = None
    parameters = dict(model.named_parameters())
    before = dict(model.gdn.__dict__)
    real_import = adapter.import_module
    calls = []
    output = object()

    def op(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(
        adapter,
        "import_module",
        lambda name: (
            real_import(name)
            if name.startswith("transformers.")
            else SimpleNamespace(gated_delta_rule=op)
        ),
    )
    adapter.apply_linear_attention_backend(model)
    recurrence = getattr(model.gdn, "chunk_gated_delta_rule", None)
    if recurrence is None:
        recurrence = inspect.unwrap(model.gdn.forward).__globals__[
            "torch_chunk_gated_delta_rule"
        ]
    inputs = [object() for _ in range(5)]
    assert recurrence(*inputs, use_qk_l2norm_in_kernel=True) == (
        output,
        None,
    )
    assert calls[0][0] == tuple(inputs)
    assert calls[0][1] == ({"deterministic_reduction": True} if deterministic else {})
    for keyword in ("deterministic_reduction", "_configured_deterministic_reduction"):
        with pytest.raises(UnsupportedCombinationError, match="fixed by model"):
            recurrence(
                *inputs, use_qk_l2norm_in_kernel=True, **{keyword: not deterministic}
            )
    assert len(calls) == 1
    assert dict(model.named_parameters()).keys() == parameters.keys()
    assert all(dict(model.named_parameters())[k] is p for k, p in parameters.items())
    for name in ("in_proj_qkv", "conv1d", "norm", "out_proj"):
        assert model.gdn._modules[name] is before["_modules"][name]
    # Real decoder forwards use_cache=False; tolerated kwargs must reach the
    # activation guard rather than spuriously failing as unknown metadata.
    with pytest.raises(UnsupportedCombinationError, match="BF16 NPU"):
        adapter.validate_gdn_inputs(
            model.gdn,
            (),
            {
                "hidden_states": torch.empty(1, 64, 2048, dtype=torch.bfloat16),
                "cache_params": None,
                "attention_mask": None,
                "use_cache": False,
                "position_ids": torch.arange(64).unsqueeze(0),
            },
        )


@pytest.mark.parametrize("value", ["true", 1, None])
def test_deterministic_flag_requires_boolean(value):
    with pytest.raises(UnsupportedCombinationError, match="must be bool"):
        adapter.validate_linear_attention_config(
            geometry(gdn_deterministic_reduction=value)
        )


def test_deterministic_flag_cannot_be_ignored_on_native_backend():
    config = geometry(
        linear_attention_backend="native", gdn_deterministic_reduction=True
    )
    with pytest.raises(
        UnsupportedCombinationError, match="requires linear_attention_backend"
    ):
        adapter.validate_linear_attention_config(config)


def test_sequence_can_reuse_numerical_policy_metadata_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        padding_metadata,
        "validate_gdn_padding_metadata",
        lambda **kwargs: calls.append(kwargs),
    )
    inputs = torch.zeros(64, dtype=torch.long)
    positions = torch.arange(64)
    adapter.validate_linear_attention_sequence(inputs, positions, None, None)
    assert len(calls) == 1
    adapter.validate_linear_attention_sequence(
        inputs, positions, None, None, validate_metadata=False
    )
    assert len(calls) == 1
