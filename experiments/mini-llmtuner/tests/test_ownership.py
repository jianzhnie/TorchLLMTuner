"""Layout provenance, aliases and source-storage lifetime for the prototype."""

import gc
from importlib import import_module

import pytest
import torch
from torch import nn

PACKAGE = "experiments.mini-llmtuner"
configuration = import_module(f"{PACKAGE}.config")
model_type = import_module(f"{PACKAGE}.model").Qwen3MoeModel
ownership = import_module(f"{PACKAGE}.ownership")
resolve = import_module(f"{PACKAGE}.resolver").resolve_plan
Trainer = import_module(f"{PACKAGE}.trainer").ExperimentTrainer


def make_model(tied=False):
    config = configuration.small_qwen3_moe_config()
    config.tie_word_embeddings = tied
    return model_type(config)


def test_local_mapping_preserves_all_aliases_and_optimizer_targets():
    model = make_model(tied=True)
    source = model.lm_head.weight
    with Trainer(
        configuration.ExperimentConfig(), torch.device("cpu"), model=model
    ) as trainer:
        mappings = trainer.assembly.parameter_mappings
        tied = next(x for x in mappings if "model.lm_head.weight" in x.source.names)
        assert tied.source.names == (
            "model.model.embed_tokens.weight",
            "model.lm_head.weight",
        )
        assert tied.target is source
        assert trainer.assembly.old_to_new_parameters[id(source)] is source
        assert tied.partitions == ()
        assert len(mappings) == len(list(model.parameters()))
        assert {id(x.target) for x in mappings} == {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        assert len(model.state_dict()) > len(mappings)
        trainer.train()
        assert model.lm_head.weight is model.tok_embeddings.weight


@pytest.mark.parametrize("alias", ["projection", "external", "view", "buffer"])
def test_sharded_aliases_are_rejected_without_mutation(alias):
    model = make_model()
    experts = model.layers[0].mlp.experts
    if alias == "projection":
        model.layers[1].mlp.experts.gate_up_proj = experts.gate_up_proj
    elif alias == "external":
        model.register_parameter("external", experts.gate_up_proj)
    elif alias == "view":
        model.register_parameter(
            "external", nn.Parameter(experts.gate_up_proj.view_as(experts.gate_up_proj))
        )
    else:
        model.register_buffer("external", experts.gate_up_proj.detach())
    before = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    config = configuration.ParallelConfig(
        tensor_parallel_size=2, expert_parallel_size=2
    )
    with pytest.raises(ValueError, match="aliases"):
        resolve(model, config, world_size=4)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    assert tuple(model.state_dict()) == keys
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["alias", "trainability"])
def test_plan_rejects_stale_ownership(kind):
    model = make_model()
    plan = resolve(model, configuration.ParallelConfig(), world_size=1)
    if kind == "alias":
        model.register_parameter("added", model.lm_head.weight)
    else:
        model.lm_head.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="Stale plan parameter"):
        plan.validate_binding(model)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_packed_tp_ep_map_selects_roles_without_retaining_source(tp_rank, ep_rank):
    model = make_model()
    plan = resolve(
        model,
        configuration.ParallelConfig(tensor_parallel_size=2, expert_parallel_size=2),
        world_size=4,
    )
    originals = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    # Apply the declared slices directly: the test's independent packed indexing
    # is compared to the ownership ranges, not another call to Partition.select.
    for spec in plan.parameter_specs:
        value = model.get_parameter(spec.names[0])
        shard = value.detach()
        if spec.tp_axis is not None:
            if spec.packed:
                gate, up = shard.chunk(2, dim=1)
                shard = torch.cat(
                    (gate.chunk(2, 1)[tp_rank], up.chunk(2, 1)[tp_rank]), dim=1
                )
            else:
                shard = shard.chunk(2, spec.tp_axis)[tp_rank]
        if spec.ep_axis is not None:
            shard = shard.chunk(2, 0)[ep_rank]
        if spec.tp_axis is not None or spec.ep_axis is not None:
            parent_path, name = spec.names[0].rsplit(".", 1)
            setattr(model.get_submodule(parent_path), name, nn.Parameter(shard.clone()))
    del value, shard
    mappings = ownership.bind_parameter_mappings(
        model,
        plan.parameter_specs,
        tp=2,
        ep=2,
        fsdp=1,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
    )
    gc.collect()
    for entry in mappings:
        torch.testing.assert_close(
            entry.select_local(originals[entry.canonical_name]),
            entry.target,
            rtol=0,
            atol=0,
        )
        if entry.partitions:
            assert entry.source.source() is None
        else:
            assert entry.source.source() is entry.target


@pytest.mark.parametrize(
    "rank,expected", [(0, (0, 1)), (1, (1, 2)), (2, (2, 2)), (3, (2, 2))]
)
def test_uneven_and_empty_fsdp_partition(rank, expected):
    entry = ownership.partition("fsdp", (2, 3), 0, 4, rank)
    assert entry.ranges == (expected,)
    full = torch.arange(6).reshape(2, 3)
    torch.testing.assert_close(entry.select(full), full[expected[0] : expected[1]])


def test_final_binding_rejects_broken_shared_identity():
    model = make_model(tied=True)
    plan = resolve(model, configuration.ParallelConfig(), world_size=1)
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="broke parameter aliases"):
        ownership.bind_parameter_mappings(
            model, plan.parameter_specs, tp=1, ep=1, fsdp=1, tp_rank=0, ep_rank=0
        )
