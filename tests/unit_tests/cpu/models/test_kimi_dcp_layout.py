"""Real DCP storage migrations between Kimi individual and grouped experts."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from llmtuner.components.checkpointer.kimi_layout import load_changed_kimi_layout
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.state_dict import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.models.kimi.checkpoint import convert_kimi_optimizer_state
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.unit_tests.cpu.parallel.test_kimi_conversion import make_model


class OptimizerState:
    def __init__(self, model, *, amsgrad=True):
        self.optimizer = torch.optim.AdamW(
            list(model.named_parameters()), lr=0.01, amsgrad=amsgrad
        )
        init_optim_state(self.optimizer)

    def state_dict(self):
        return get_flat_optim_state_dict(self.optimizer)

    def load_state_dict(self, state):
        load_flat_optim_state_dict(self.optimizer, flat_state=state)


@pytest.mark.parametrize("source_grouped", [False, True])
def test_dcp_kimi_layout_restores_weights_moments_and_next_update(
    tmp_path, source_grouped
):
    # Register under model so FQNs match the real HF wrapper schema.
    def build():
        wrapper = torch.nn.Module()
        wrapper.model = make_model(latent=True, shared=1)
        return wrapper

    source = build()
    config = source.model.layers[0].block_sparse_moe.config
    if source_grouped:
        swap_hf_moe_blocks(source.model)
    source_optimizer = OptimizerState(source)
    inputs = torch.randn(1, 3, 8)
    source.model.layers[0].block_sparse_moe(inputs).square().sum().backward()
    source_optimizer.optimizer.step()
    source_optimizer.optimizer.zero_grad()
    source_ema = EMA(model_parts=[source], decay=0.5)
    source_ema.step(1)
    state = dict(source.state_dict())
    state["optimizer"] = source_optimizer
    state["ema"] = source_ema
    dcp.save(state, checkpoint_id=tmp_path)
    target = build()
    if not source_grouped:
        swap_hf_moe_blocks(target.model)
    target_optimizer = OptimizerState(target)
    target_ema = EMA(model_parts=[target], decay=0.5)
    adapter = HFTransformerStateDictAdapter(
        config, None, kimi_expert_parallel=not source_grouped
    )
    assert load_changed_kimi_layout(
        str(tmp_path),
        states={
            "model": ModelCheckpointState(target),
            "optimizer": target_optimizer,
            "ema": target_ema,
        },
        adapter=adapter,
    )
    expected_ema = convert_kimi_optimizer_state(
        source_ema.state_dict(), num_experts=4, to_hf=source_grouped
    )
    actual_ema = target_ema.state_dict()
    assert actual_ema.keys() == expected_ema.keys()
    for key in expected_ema:
        torch.testing.assert_close(actual_ema[key], expected_ema[key], atol=0, rtol=0)
    source_output = source.model.layers[0].block_sparse_moe(inputs)
    target_output = target.model.layers[0].block_sparse_moe(inputs)
    torch.testing.assert_close(target_output, source_output, atol=1e-6, rtol=1e-6)
    # Compare the next real update through each independent layout.
    source_output.square().sum().backward()
    target_output.square().sum().backward()
    source_optimizer.optimizer.step()
    target_optimizer.optimizer.step()
    torch.testing.assert_close(
        target.model.layers[0].block_sparse_moe(inputs),
        source.model.layers[0].block_sparse_moe(inputs),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_invalid_cross_layout_model_leaves_target_weights_intact(tmp_path, corruption):
    source = torch.nn.Module()
    source.model = make_model(latent=True, shared=1)
    config = source.model.layers[0].block_sparse_moe.config
    saved = dict(source.state_dict())
    key = "model.layers.0.block_sparse_moe.shared_experts.down_proj.weight"
    if corruption == "missing":
        del saved[key]
    else:
        saved[key] = torch.zeros(1)
    dcp.save(saved, checkpoint_id=tmp_path)
    target = torch.nn.Module()
    target.model = make_model(latent=True, shared=1)
    swap_hf_moe_blocks(target.model)
    snapshot = {key: value.clone() for key, value in target.state_dict().items()}
    adapter = HFTransformerStateDictAdapter(config, None, kimi_expert_parallel=True)
    with pytest.raises(ValueError, match="missing keys|has shape"):
        load_changed_kimi_layout(
            str(tmp_path),
            states={"model": ModelCheckpointState(target)},
            adapter=adapter,
        )
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)


@pytest.mark.parametrize("category", ["optimizer", "ema"])
@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_invalid_cross_layout_training_state_preserves_model(
    tmp_path, category, corruption
):
    source = torch.nn.Module()
    source.model = make_model(latent=True, shared=1)
    config = source.model.layers[0].block_sparse_moe.config
    training_state = (
        OptimizerState(source)
        if category == "optimizer"
        else EMA(model_parts=[source], decay=0.5)
    ).state_dict()
    suffix = "exp_avg" if category == "optimizer" else "ema_params"
    key = (
        "state.model.layers.0.block_sparse_moe.shared_experts.down_proj.weight."
        + suffix
    )
    if corruption == "missing":
        del training_state[key]
    else:
        training_state[key] = torch.zeros(1)
    saved = dict(source.state_dict())
    saved[category] = training_state
    dcp.save(saved, checkpoint_id=tmp_path)
    target = torch.nn.Module()
    target.model = make_model(latent=True, shared=1)
    swap_hf_moe_blocks(target.model)
    target_state = (
        OptimizerState(target)
        if category == "optimizer"
        else EMA(model_parts=[target], decay=0.5)
    )
    snapshot = {key: value.clone() for key, value in target.state_dict().items()}
    adapter = HFTransformerStateDictAdapter(config, None, kimi_expert_parallel=True)
    with pytest.raises(ValueError, match="missing required key|has shape"):
        load_changed_kimi_layout(
            str(tmp_path),
            states={"model": ModelCheckpointState(target), category: target_state},
            adapter=adapter,
        )
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_saved_amsgrad_validated_before_model_with_fresh_non_amsgrad(
    tmp_path, corruption
):
    source = torch.nn.Module()
    source.model = make_model(latent=True, shared=1)
    config = source.model.layers[0].block_sparse_moe.config
    training_state = OptimizerState(source).state_dict()
    key = (
        "state.model.layers.0.block_sparse_moe.shared_experts.down_proj.weight."
        "max_exp_avg_sq"
    )
    if corruption == "missing":
        del training_state[key]
    else:
        training_state[key] = torch.zeros(1)
    saved = dict(source.state_dict())
    saved["optimizer"] = training_state
    dcp.save(saved, checkpoint_id=tmp_path)
    target = torch.nn.Module()
    target.model = make_model(latent=True, shared=1)
    swap_hf_moe_blocks(target.model)
    target_state = OptimizerState(target, amsgrad=False)
    snapshot = {key: value.clone() for key, value in target.state_dict().items()}
    adapter = HFTransformerStateDictAdapter(config, None, kimi_expert_parallel=True)
    with pytest.raises(ValueError, match="missing required key|has shape"):
        load_changed_kimi_layout(
            str(tmp_path),
            states={"model": ModelCheckpointState(target), "optimizer": target_state},
            adapter=adapter,
        )
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)


@pytest.mark.parametrize("source_grouped", [False, True])
def test_legacy_cross_layout_checkpoint_without_rng_restores_model(
    tmp_path, source_grouped
):
    from llmtuner.components.checkpointer.random_state import RNGCheckpointState

    source = torch.nn.Module()
    source.model = make_model(latent=True, shared=1)
    config = source.model.layers[0].block_sparse_moe.config
    if source_grouped:
        swap_hf_moe_blocks(source.model)
    dcp.save(dict(source.state_dict()), checkpoint_id=tmp_path)
    target = torch.nn.Module()
    target.model = make_model(latent=True, shared=1)
    if not source_grouped:
        swap_hf_moe_blocks(target.model)
    rng = RNGCheckpointState(torch.device("cpu"))
    rng_before = torch.get_rng_state().clone()
    adapter = HFTransformerStateDictAdapter(
        config, None, kimi_expert_parallel=not source_grouped
    )
    assert load_changed_kimi_layout(
        str(tmp_path),
        states={"model": ModelCheckpointState(target), "rng_state": rng},
        adapter=adapter,
    )
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
    inputs = torch.ones(1, 3, 8)
    torch.testing.assert_close(
        target.model.layers[0].block_sparse_moe(inputs),
        source.model.layers[0].block_sparse_moe(inputs),
        atol=1e-6,
        rtol=1e-6,
    )
