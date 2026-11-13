"""Full Kimi HF state round-trip through native grouped expert execution."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
from llmtuner.models.kimi.kimi_k3.checkpoint import (
    convert_kimi_expert_state,
    convert_kimi_optimizer_state,
)
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks
from tests.unit_tests.cpu.models.test_family_execution_policy import kimi_config


@pytest.mark.parametrize(
    "convert", [convert_kimi_expert_state, convert_kimi_optimizer_state]
)
@pytest.mark.parametrize("num_experts", [0, -1, True, 2.0, "2", None])
@pytest.mark.parametrize("to_hf", [False, True])
def test_kimi_conversion_rejects_invalid_geometry_before_reading_state(
    convert, num_experts, to_hf
):
    class UnreadableState(dict):
        def items(self):
            raise AssertionError("Invalid geometry reached state traversal")

    with pytest.raises(ValueError, match="num_experts must be a positive integer"):
        convert(UnreadableState(), num_experts=num_experts, to_hf=to_hf)


@pytest.mark.parametrize(
    "convert", [convert_kimi_expert_state, convert_kimi_optimizer_state]
)
@pytest.mark.parametrize("to_hf", [0, 1, "false", None])
def test_kimi_conversion_rejects_nonboolean_direction(convert, to_hf):
    with pytest.raises(ValueError, match="to_hf must be a bool"):
        convert({}, num_experts=2, to_hf=to_hf)


@pytest.mark.parametrize("kind", ["linear", "embedding", "padded_embedding"])
def test_kimi_initializer_preserves_loaded_weight_and_rng(kind):
    model = KimiLinearForCausalLM(kimi_config())
    module = (
        torch.nn.Linear(8, 8, bias=False)
        if kind == "linear"
        else torch.nn.Embedding(
            8, 8, padding_idx=0 if kind == "padded_embedding" else None
        )
    )
    with torch.no_grad():
        module.weight.fill_(0.75)
    module.weight._is_hf_initialized = True
    identity = id(module.weight)
    rng = torch.get_rng_state().clone()
    model._init_weights(module)
    assert id(module.weight) == identity
    assert torch.equal(module.weight, torch.full_like(module.weight, 0.75))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("tied", [False, True])
def test_kimi_external_checkpoint_restores_parameters_and_logits(tmp_path, tied):
    config = kimi_config()
    config.tie_word_embeddings = tied
    model = KimiLinearForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    restored = KimiLinearForCausalLM.from_pretrained(
        tmp_path, local_files_only=True, attn_implementation="eager"
    ).eval()
    assert (restored.lm_head.weight is restored.model.embed_tokens.weight) == tied
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    tokens = torch.tensor([[1, 3, 7], [2, 4, 6]])
    with torch.no_grad():
        expected = model(tokens).logits
        actual = restored(tokens).logits
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("latent", [False, True])
def test_kimi_grouped_export_loads_strictly_in_original_hf_model(latent, tmp_path):
    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        num_shared_experts=1,
        num_expert_group=2,
        topk_group=1,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    source = KimiLinearForCausalLM(config).eval()
    original = {key: value.clone() for key, value in source.state_dict().items()}
    swap_moe_blocks(source)
    adapter = ModelStateDictAdapter(config, None, expert_parallel=True)
    exported = adapter.to_hf(source.state_dict())
    assert exported.keys() == original.keys()
    for key in original:
        torch.testing.assert_close(exported[key], original[key], atol=0, rtol=0)
    path = tmp_path / "model.safetensors"
    save_file(
        {key: value.detach().contiguous() for key, value in exported.items()}, path
    )
    exported = load_file(path)
    restored = KimiLinearForCausalLM(config).eval()
    restored.load_state_dict(exported, strict=True)
    grouped = KimiLinearForCausalLM(config).eval()
    swap_moe_blocks(grouped)
    grouped.load_state_dict(adapter.from_hf(exported), strict=True)
    ids = torch.tensor([1, 2, 3])
    torch.testing.assert_close(
        source(ids).logits, restored(ids).logits, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(source(ids).logits, grouped(ids).logits, atol=0, rtol=0)
    meta = {key: value.to("meta") for key, value in exported.items()}
    assert all(value.is_meta for value in adapter.from_hf(meta).values())


def test_kimi_export_rejects_local_shards():
    key = "model.layers.0.block_sparse_moe.routed_experts.inner_experts.w1_EFD"
    with pytest.raises(ValueError, match="gather EP shards"):
        convert_kimi_expert_state(
            {key: torch.randn(2, 4, 8)},
            num_experts=4,
            to_hf=True,
        )


def test_kimi_import_rejects_missing_experts():
    with pytest.raises(ValueError, match="missing individual experts"):
        convert_kimi_expert_state(
            {"model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(4, 8)},
            num_experts=4,
            to_hf=False,
        )


def test_kimi_mapping_rejects_key_collisions():
    with pytest.raises(ValueError, match="collision"):
        convert_kimi_expert_state(
            {
                "model.layers.0.block_sparse_moe.router.gate.weight": torch.randn(4, 8),
                "model.layers.0.block_sparse_moe.gate.weight": torch.randn(4, 8),
            },
            num_experts=4,
            to_hf=True,
        )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_kimi_import_rejects_duplicate_numeric_expert_indices(reverse_order):
    prefix = "model.layers.0.block_sparse_moe.experts."
    entries = [
        (prefix + "0.w1.weight", torch.ones(2, 3)),
        (prefix + "00.w1.weight", torch.zeros(2, 3)),
        (prefix + "1.w1.weight", torch.ones(2, 3)),
    ]
    if reverse_order:
        entries.reverse()
    with pytest.raises(ValueError, match="duplicate expert index"):
        convert_kimi_expert_state(dict(entries), num_experts=2, to_hf=False)


def test_kimi_import_rejects_inconsistent_shapes_before_stacking():
    prefix = "model.layers.0.block_sparse_moe.experts."
    with pytest.raises(ValueError, match="expert weight shapes disagree"):
        convert_kimi_expert_state(
            {
                prefix + "0.w1.weight": torch.ones(2, 3),
                prefix + "1.w1.weight": torch.ones(4, 3),
            },
            num_experts=2,
            to_hf=False,
        )


@pytest.mark.parametrize("mismatch", ["dtype", "device"])
def test_kimi_import_rejects_mixed_metadata_before_stacking(monkeypatch, mismatch):
    prefix = "model.layers.0.block_sparse_moe.experts."
    first = torch.ones(2, 3)
    second = first.double() if mismatch == "dtype" else first.to("meta")

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid metadata reached tensor stacking")

    monkeypatch.setattr(torch, "stack", forbidden)
    with pytest.raises(ValueError, match="dtype/device disagree"):
        convert_kimi_expert_state(
            {prefix + "0.w1.weight": first, prefix + "1.w1.weight": second},
            num_experts=2,
            to_hf=False,
        )
