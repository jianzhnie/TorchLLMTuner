"""Production Qwen3 stage ownership preserves forward, gradients and updates."""

import pytest
import torch
from torch import nn

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.pipeline_parallel import partition
from llmtuner.parallel.pipeline_parallel.partition import split_model_into_stages
from llmtuner.parallel.pipeline_parallel.pipeline_parallel import scalar_loss_fn
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


class LocalMesh:
    def size(self):
        return 1

    def get_local_rank(self):
        return 0

    def get_group(self):
        return None


def assignments():
    return [
        ["model.embed_tokens", "model.layers.0", "model.rotary_emb"],
        ["model.layers.1", "model.norm", "lm_head", "model.rotary_emb"],
    ]


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("role", ["parameter", "buffer"])
def test_partition_rejects_same_stage_parameter_storage_views(
    monkeypatch, device, role
):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    source = model.layers[0].input_layernorm.weight
    view = source.detach().view_as(source)
    if role == "parameter":
        model.layers[0].post_attention_layernorm.weight = nn.Parameter(view)
    else:
        model.layers[0].register_buffer("storage_view", view)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        partition.copy, "deepcopy", lambda *a: pytest.fail("unsafe alias reached copy")
    )
    with pytest.raises(UnsupportedCombinationError, match="distinct tensors.*storage"):
        split_model_into_stages(
            model,
            LocalMesh(),
            "1F1B",
            torch.device("cpu"),
            assignments(),
            replicated_module_names=model.pipeline_runtime_modules,
        )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("alias_kind", ["parameter_identity", "buffer_view"])
def test_partition_preserves_supported_same_stage_aliases(monkeypatch, alias_kind):
    model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    block = model.layers[0]
    if alias_kind == "parameter_identity":
        block.post_attention_layernorm.weight = block.input_layernorm.weight
    else:
        block.register_buffer("base", torch.arange(8.0))
        block.register_buffer("view", block.base[2:6])
    monkeypatch.setattr(partition, "ComposablePipelineStage", lambda *a, **kw: object())
    _, parts = split_model_into_stages(
        model,
        LocalMesh(),
        "1F1B",
        torch.device("cpu"),
        assignments(),
        replicated_module_names=model.pipeline_runtime_modules,
    )
    copied = parts[0].layers[0]
    if alias_kind == "parameter_identity":
        assert copied.post_attention_layernorm.weight is copied.input_layernorm.weight
        assert copied.input_layernorm.weight is not block.input_layernorm.weight
    else:
        assert (
            copied.base.untyped_storage()._cdata == copied.view.untyped_storage()._cdata
        )
        assert (
            copied.base.untyped_storage()._cdata != block.base.untyped_storage()._cdata
        )
        assert copied.view.storage_offset() == 2
        copied.base[2] = 42
        assert copied.view[0] == 42
        assert block.base[2] == 2


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("role", ["parameter", "buffer"])
def test_partition_rejects_distinct_tensor_storage_alias_before_copy(
    monkeypatch, device, role
):
    with torch.device(device):
        model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    source = model.layers[0].input_layernorm.weight
    alias = source.detach().view_as(source)
    if role == "parameter":
        model.layers[1].post_attention_layernorm.weight = nn.Parameter(alias)
        alias = model.layers[1].post_attention_layernorm.weight
    else:
        model.layers[1].register_buffer("storage_alias", alias)
    assert alias is not source
    assert alias.untyped_storage()._cdata == source.untyped_storage()._cdata
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        partition.copy, "deepcopy", lambda *a: pytest.fail("alias reached copy")
    )
    with pytest.raises(
        UnsupportedCombinationError, match="shared storage.*spans stages"
    ):
        split_model_into_stages(
            model,
            LocalMesh(),
            "1F1B",
            torch.device("cpu"),
            assignments(),
            replicated_module_names=model.pipeline_runtime_modules,
        )
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("parameter_stage", [0, 1])
def test_explicit_partition_rejects_cross_role_tensor_alias_before_copy(
    monkeypatch, parameter_stage
):
    model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    source = model.layers[parameter_stage].input_layernorm.weight
    model.layers[1 - parameter_stage].register_buffer("cross_role_alias", source)
    assert model.layers[1 - parameter_stage].cross_role_alias is source
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cross-stage alias reached stage copying")

    monkeypatch.setattr(partition.copy, "deepcopy", forbidden)
    with pytest.raises(
        UnsupportedCombinationError, match="shared (parameter|buffer).*(spans|stages)"
    ):
        split_model_into_stages(
            model,
            LocalMesh(),
            "1F1B",
            torch.device("cpu"),
            assignments(),
            replicated_module_names=model.pipeline_runtime_modules,
        )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("extra", ["parameter", "buffer", "module", "alias"])
def test_explicit_partition_rejects_unassigned_state_before_copy(monkeypatch, extra):
    model = Qwen3MoeForCausalLM(model_config(tie_word_embeddings=False))
    if extra == "parameter":
        model.register_parameter("unowned", nn.Parameter(torch.ones(3)))
    elif extra == "buffer":
        model.register_buffer("unowned", torch.ones(3))
    elif extra == "alias":
        model.register_parameter("unowned", model.lm_head.weight)
    else:
        model.unowned = nn.Linear(4, 4, bias=False)
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unassigned state reached stage copying")

    monkeypatch.setattr(partition.copy, "deepcopy", forbidden)
    with pytest.raises(ValueError, match="unassigned state paths.*unowned"):
        split_model_into_stages(
            model,
            LocalMesh(),
            "1F1B",
            torch.device("cpu"),
            assignments(),
            replicated_module_names=model.pipeline_runtime_modules,
        )
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_explicit_stage_chain_matches_model_gradients_and_updates(monkeypatch):
    torch.manual_seed(53)
    config = model_config(tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config)
    monkeypatch.setattr(partition, "ComposablePipelineStage", lambda *a, **kw: object())
    _, parts = split_model_into_stages(
        model,
        LocalMesh(),
        "1F1B",
        torch.device("cpu"),
        assignments(),
        replicated_module_names=model.pipeline_runtime_modules,
    )
    assert isinstance(parts[0].lm_head, nn.Identity)
    assert isinstance(parts[1].model.embed_tokens, nn.Identity)
    assert list(parts[0].model.layers._modules) == ["0"]
    assert list(parts[1].model.layers._modules) == ["1"]
    assert "lm_head" not in parts[0].tp_config
    assert "lm_head" in parts[1].tp_config
    originals = dict(model.named_parameters())
    stage_parameters = {
        name: parameter for part in parts for name, parameter in part.named_parameters()
    }
    assert stage_parameters.keys() == originals.keys()
    assert sum(len(part.state_dict()) for part in parts) == len(model.state_dict())
    optimizers = [
        torch.optim.SGD(part.parameters(), lr=0.01, foreach=False) for part in parts
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    for step in range(3):
        tokens = (torch.arange(7) * 3 + step) % config.vocab_size
        positions = torch.arange(7)
        expected = model(tokens, positions=positions).logits
        hidden = parts[0](tokens, positions=positions)
        actual = parts[1](hidden, positions=positions)
        assert hidden.shape == (7, config.hidden_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for name, parameter in stage_parameters.items():
            torch.testing.assert_close(
                parameter.grad, originals[name].grad, rtol=0, atol=0
            )
        assert stage_parameters["model.embed_tokens.weight"].grad.count_nonzero() > 0
        for opt in (*optimizers, optimizer):
            opt.step()
            opt.zero_grad(set_to_none=True)
        for name, parameter in stage_parameters.items():
            torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["missing_block", "wrong_stage", "tied"])
def test_explicit_partition_rejects_before_model_copy(monkeypatch, invalid):
    config = model_config(tie_word_embeddings=invalid == "tied")
    model = Qwen3MoeForCausalLM(config)
    targets = assignments()
    if invalid == "missing_block":
        targets[1].remove("model.layers.1")
    elif invalid == "wrong_stage":
        targets[0].append("lm_head")
    monkeypatch.setattr(
        partition.copy,
        "deepcopy",
        lambda *a: pytest.fail("invalid partition reached copy"),
    )
    with pytest.raises(
        (ValueError, NotImplementedError), match="assign every|ownership|untied"
    ):
        split_model_into_stages(
            model,
            LocalMesh(),
            "1F1B",
            torch.device("cpu"),
            targets,
            replicated_module_names=model.pipeline_runtime_modules,
        )


@pytest.mark.parametrize("batched", [False, True])
def test_pipeline_loss_normalizes_flat_and_batched_tokens_equally(batched):
    logits = torch.linspace(-1, 1, 30).reshape(2, 3, 5).requires_grad_()
    labels = torch.tensor([[0, 1, -100], [3, 4, 2]])
    denominator = torch.tensor(5.0)
    expected = (
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, 5), labels.flatten(), reduction="sum"
        )
        / denominator
    )
    actual = scalar_loss_fn(
        logits if batched else logits.reshape(-1, 5),
        labels if batched else labels.flatten(),
        global_valid_tokens=denominator,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    assert actual_gradient.count_nonzero() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
