"""Qwen3.5 initialization keeps default RoPE's materialized-device values."""

from copy import deepcopy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_initializer_keeps_rotary_lifecycle_and_parameter_rng(dtype, shared, dynamic):
    model = Qwen3_5MoeForCausalLM(text_config(Qwen3_5MoeTextConfig, dynamic=dynamic))
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype)
    rotary = model.model.rotary_emb
    frequencies = torch.nextafter(
        rotary.inv_freq, torch.full_like(rotary.inv_freq, float("inf"))
    )
    rotary.inv_freq.copy_(frequencies)
    rotary.original_inv_freq.copy_(frequencies)
    if shared:
        rotary.original_inv_freq = rotary.inv_freq
    model.register_buffer("frequency_alias", rotary.inv_freq, persistent=False)
    identities = {name: id(p) for name, p in model.named_parameters()}
    aliases = {name: id(b) for name, b in model.named_buffers(remove_duplicate=False)}
    keys = tuple(model.state_dict())
    reference = deepcopy(model)
    torch.manual_seed(919)
    reference.apply(reference._init_weights)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(919)
    model.initialize_parameters()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert tuple(model.state_dict()) == keys
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert aliases == {
        name: id(b) for name, b in model.named_buffers(remove_duplicate=False)
    }
    assert model.frequency_alias is rotary.inv_freq
    assert (rotary.original_inv_freq is rotary.inv_freq) == shared
    expected_frequency = reference.model.rotary_emb.inv_freq if dynamic else frequencies
    torch.testing.assert_close(rotary.inv_freq, expected_frequency, rtol=0, atol=0)
    torch.testing.assert_close(
        rotary.original_inv_freq, expected_frequency, rtol=0, atol=0
    )
    actual, expected = (
        dict(model.named_parameters()),
        dict(reference.named_parameters()),
    )
    assert actual.keys() == expected.keys()
    assert any(p.count_nonzero() > 0 for p in actual.values())
    for name, parameter in actual.items():
        assert parameter.isfinite().all(), name
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)


def test_meta_materialization_and_native_dcp_rebuild_nonpersistent_rope(tmp_path):
    config = text_config(Qwen3_5MoeTextConfig)
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(deepcopy(config))
    model.register_buffer(
        "frequency_alias", model.model.rotary_emb.inv_freq, persistent=False
    )
    parameters = {name: id(p) for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    buffers = model.materialization_buffers(torch.device("cpu"))
    materialize_meta_model(model, torch.device("cpu"), runtime_buffers=buffers)
    assert torch.equal(rng, torch.get_rng_state())
    assert parameters == {name: id(p) for name, p in model.named_parameters()}
    assert model.frequency_alias is model.model.rotary_emb.inv_freq
    frequencies = model.frequency_alias.clone()
    model.initialize_parameters()
    torch.testing.assert_close(model.frequency_alias, frequencies, rtol=0, atol=0)
    state = model.state_dict()
    assert not any("inv_freq" in name or "frequency_alias" in name for name in state)
    checkpoint = tmp_path / "model"
    dcp.save(state, checkpoint_id=checkpoint)
    with torch.device("meta"):
        restored = Qwen3_5MoeForCausalLM(deepcopy(config))
    materialize_meta_model(
        restored,
        torch.device("cpu"),
        runtime_buffers=restored.materialization_buffers(torch.device("cpu")),
    )
    restored_state = restored.state_dict()
    dcp.load(restored_state, checkpoint_id=checkpoint)
    restored.load_state_dict(restored_state)
    assert restored_state.keys() == state.keys()
    for name, value in restored_state.items():
        assert value.isfinite().all(), name
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(
        restored.model.rotary_emb.inv_freq, frequencies, rtol=0, atol=0
    )
    tokens = torch.tensor([[1, 3, 7, 9]])
    with torch.no_grad():
        expected = model(tokens).logits
        actual = restored(tokens).logits
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_external_checkpoint_resume_preserves_next_update(tmp_path, dtype, tied):
    config = text_config(Qwen3_5MoeTextConfig)
    config.tie_word_embeddings = tied
    config._attn_implementation = "eager"
    config.experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(config).to(dtype)
    model.initialize_parameters()
    tokens = torch.tensor([[1, 3, 7, 9], [2, 8, 4, 6]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    model(tokens).logits.float().square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "external"
    model.save_pretrained(checkpoint)
    restored = Qwen3_5MoeForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    assert type(restored) is Qwen3_5MoeForCausalLM
    assert (restored.lm_head.weight is restored.model.embed_tokens.weight) == tied
    assert model.state_dict().keys() == restored.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    for buffer in restored.model.rotary_emb.buffers():
        assert not buffer.is_meta and buffer.isfinite().all()
    model.train()
    restored.train()
    resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, foreach=False)
    expected = model(tokens).logits
    actual = restored(tokens).logits
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.autograd.set_multithreading_enabled(False):
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
    expected_parameters = dict(model.named_parameters())
    for name, parameter in restored.named_parameters():
        other = expected_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, other.grad, rtol=0, atol=0, msg=name
            )
    assert restored.lm_head.weight.grad.count_nonzero() > 0
    optimizer.step()
    resumed_optimizer.step()
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)
