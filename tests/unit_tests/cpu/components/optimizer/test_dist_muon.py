import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from llmtuner.components.optimizer.flex_shard import BucketConfig, ComputeLayout, Owned
from llmtuner.components.optimizer.flex_shard.dist_muon import DistMuon
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import OptimizerConfig, ParamGroupConfig


@pytest.fixture(autouse=True)
def process_group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


def _model():
    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(
        DTensor.from_local(
            torch.randn(4, 4),
            mesh,
            (Shard(0),),
            shape=torch.Size((4, 4)),
            stride=(4, 1),
        )
    )
    return model


def _config(residency="device"):
    return OptimizerConfig(
        state_residency=residency,
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="DistMuon",
                optimizer_kwargs={"lr": 1e-3},
            )
        ],
        compute_sharding_by_fqn={
            "weight": ComputeLayout({"dp_shard": Owned()}, compute_dtype=torch.bfloat16)
        },
        bucket_configs=[BucketConfig(patterns=("weight",))],
    )


def test_constructor_validates_names_before_compute_layout_lookup():
    model = _model()
    with pytest.raises(ValueError, match="must be aligned"):
        DistMuon(
            [{"params": [model.weight]}],
            compute_sharding_by_fqn={},
            bucket_configs=[BucketConfig(patterns=("weight",))],
        )
    with pytest.raises(ValueError, match="missing compute sharding"):
        DistMuon(
            [{"params": [model.weight], "param_names": ["weight"]}],
            compute_sharding_by_fqn={},
            bucket_configs=[BucketConfig(patterns=("weight",))],
        )


@pytest.mark.parametrize(
    ("shape", "dim", "message"),
    [
        ((4, 4), 1, "must shard tensor dimension 0"),
        ((0, 4), 0, "requires at least one matrix block"),
        ((2, 2, 4), 0, "requires a 2D row-concatenated parameter"),
    ],
)
def test_block_compute_rejects_invalid_matrix_geometry(shape, dim, message):
    from llmtuner.components.optimizer.flex_shard.dist_muon import (
        _resolve_storage_to_compute_transition,
    )
    from llmtuner.components.optimizer.flex_shard.optimizer_reshard import BlockShard

    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
    parameter = DTensor.from_local(torch.zeros(shape), mesh, (Shard(0),))
    layout = ComputeLayout({"dp_shard": BlockShard(dim=dim, block_sizes=(2,))})

    with pytest.raises(ValueError, match=message):
        _resolve_storage_to_compute_transition("weight", parameter, layout)


def test_checkpoint_state_initialization_does_not_execute_a_muon_step():
    from llmtuner.components.optimizer.state_dict import init_optim_state

    model = _model()
    container = OptimizersContainer(_config(), model_parts=[model])
    optimizer = container.optimizers[0]
    before = model.weight.to_local().clone()
    assert not optimizer._first_step_validated

    init_optim_state(optimizer)

    assert not optimizer._first_step_validated
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight.to_local(), before, rtol=0, atol=0)
    momentum = optimizer.state[model.weight]["momentum_buffer"]
    assert torch.count_nonzero(momentum.to_local()) == 0
    init_optim_state(optimizer)
    assert optimizer.state[model.weight]["momentum_buffer"] is momentum


def test_invalid_checkpoint_groups_are_rejected_before_loading_state():
    from copy import deepcopy

    model = _model()
    container = OptimizersContainer(_config(), model_parts=[model])
    original = deepcopy(container.state_dict())
    incoming = deepcopy(original)
    incoming["param_groups.weight.lr"] = 0.5
    incoming["param_groups.weight.ns_steps"] = -1

    with pytest.raises(ValueError, match="unsupported DistMuon group"):
        container.load_state_dict(incoming)

    assert container.optimizers[0].param_groups[0]["lr"] == 1e-3
    assert container.optimizers[0].param_groups[0]["ns_steps"] == 5


def _multi_group_model():
    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
    model = torch.nn.Module()
    for name, offset in (("weight", 0), ("projection", 16)):
        local = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 32 + offset
        setattr(
            model,
            name,
            torch.nn.Parameter(
                DTensor.from_local(
                    local,
                    mesh,
                    (Shard(0),),
                    shape=local.shape,
                    stride=local.stride(),
                )
            ),
        )
    return model


def _multi_group_config():
    return OptimizerConfig(
        param_groups=[
            ParamGroupConfig(
                pattern=r"^weight$",
                optimizer_name="DistMuon",
                optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.05},
            ),
            ParamGroupConfig(
                pattern=r"^projection$",
                optimizer_name="DistMuon",
                optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.2},
            ),
        ],
        compute_sharding_by_fqn={
            name: ComputeLayout({"dp_shard": Owned()}, compute_dtype=torch.bfloat16)
            for name in ("weight", "projection")
        },
        bucket_configs=[BucketConfig(patterns=("*",))],
    )


def test_dist_muon_construction_preserves_fqn_state():
    container = OptimizersContainer(_config(), model_parts=[_model()])

    assert len(container.optimizers) == 1
    assert isinstance(container.optimizers[0], DistMuon)
    state = container.state_dict()
    assert any(key.startswith("state.weight.") for key in state)


def test_dist_muon_rejects_invalid_ns_device_from_config():
    config = _config()
    config.param_groups[0].optimizer_kwargs["ns_device"] = "host"
    with pytest.raises(ValueError, match="ns_device must be"):
        OptimizersContainer(config, model_parts=[_model()])


def test_dist_muon_rejects_conflicting_ns_devices():
    config = _multi_group_config()
    config.param_groups[0].optimizer_kwargs["ns_device"] = "accelerator"
    with pytest.raises(ValueError, match="identical across parameter groups"):
        OptimizersContainer(config, model_parts=[_multi_group_model()])


def test_dist_muon_rejects_adam_virtual_residency():
    with pytest.raises(ValueError, match="Adam/AdamW moments only"):
        OptimizersContainer(_config("npu_virtual"), model_parts=[_model()])


def test_dist_muon_rejects_vector_parameters():
    shape = (4,)
    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
    model = torch.nn.Module()
    local = torch.randn(shape)
    model.weight = torch.nn.Parameter(
        DTensor.from_local(
            local,
            mesh,
            (Shard(0),),
            shape=local.shape,
            stride=local.stride(),
        )
    )
    with pytest.raises(ValueError, match="requires matrix parameters"):
        OptimizersContainer(_config(), model_parts=[model])


def test_dist_muon_accepts_batched_matrix_parameters():
    shape = (2, 4, 4)
    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp_shard",))
    model = torch.nn.Module()
    local = torch.randn(shape)
    model.weight = torch.nn.Parameter(
        DTensor.from_local(
            local,
            mesh,
            (Shard(0),),
            shape=local.shape,
            stride=local.stride(),
        )
    )
    optimizer = OptimizersContainer(_config(), model_parts=[model])
    assert len(optimizer.optimizers) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"lr": -1.0},
        {"weight_decay": float("nan")},
        {"momentum": -1.0},
        {"eps": -1.0},
        {"ns_steps": 1.5},
        {"ns_steps": 100},
        {"ns_coefficients": None},
        {"ns_coefficients": ("invalid", -4.0, 2.0)},
        {"ns_coefficients": ((3.0, -4.0, 2.0),)},
    ],
)
def test_dist_muon_rejects_invalid_update_options(options):
    config = _config()
    config.param_groups[0].optimizer_kwargs.update(options)
    with pytest.raises(ValueError, match="unsupported DistMuon group"):
        OptimizersContainer(config, model_parts=[_model()])


def test_dist_muon_rejects_invalid_checkpoint_options():
    from copy import deepcopy

    optimizer = OptimizersContainer(_config(), model_parts=[_model()]).optimizers[0]
    original_groups = deepcopy(optimizer.param_groups)
    state = deepcopy(optimizer.state_dict())
    state["param_groups"][0]["lr"] = 0.5
    state["param_groups"][0]["eps"] = -1.0
    with pytest.raises(ValueError, match="unsupported DistMuon group"):
        optimizer.load_state_dict(state)
    assert optimizer.param_groups[0]["lr"] == original_groups[0]["lr"]
    assert optimizer.param_groups[0]["eps"] == original_groups[0]["eps"]


def test_dist_muon_updates_match_torch_muon():
    model = _model()
    reference_parameter = torch.nn.Parameter(model.weight.to_local().detach().clone())
    optimizer = OptimizersContainer(_config(), model_parts=[model])
    reference = torch.optim.Muon([reference_parameter], lr=1e-3, weight_decay=0.1)
    for gradient in (torch.zeros(4, 4), torch.randn(4, 4), torch.randn(4, 4)):
        model.weight.grad = DTensor.from_local(
            gradient.clone(), model.weight.device_mesh, model.weight.placements
        )
        reference_parameter.grad = gradient.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(
            model.weight.to_local(), reference_parameter, rtol=0, atol=0
        )


def test_dist_muon_multiple_groups_update_and_resume_by_fqn():
    model = _multi_group_model()
    references = {
        name: torch.nn.Parameter(parameter.to_local().detach().clone())
        for name, parameter in model.named_parameters()
    }
    reference = torch.optim.Muon(
        [
            {"params": [references["weight"]], "lr": 1e-3, "weight_decay": 0.05},
            {
                "params": [references["projection"]],
                "lr": 5e-4,
                "weight_decay": 0.2,
            },
        ]
    )
    optimizer = OptimizersContainer(_multi_group_config(), model_parts=[model])
    assert len(optimizer.optimizers) == 1
    assert len(optimizer.optimizers[0].param_groups) == 2

    def assign_gradients(target, value):
        for index, name in enumerate(("weight", "projection")):
            gradient = torch.full((4, 4), value * (index + 1))
            parameter = getattr(target, name)
            if isinstance(parameter, DTensor):
                parameter.grad = DTensor.from_local(
                    gradient,
                    parameter.device_mesh,
                    parameter.placements,
                    shape=parameter.shape,
                    stride=parameter.stride(),
                )
            else:
                parameter.grad = gradient

    for value in (0.25, -0.5):
        assign_gradients(model, value)
        for index, name in enumerate(("weight", "projection")):
            references[name].grad = torch.full((4, 4), value * (index + 1))
        optimizer.step()
        reference.step()
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)

    checkpoint = optimizer.state_dict()
    resumed_model = _multi_group_model()
    with torch.no_grad():
        for name in ("weight", "projection"):
            getattr(resumed_model, name).to_local().copy_(
                getattr(model, name).to_local()
            )
    resumed = OptimizersContainer(_multi_group_config(), model_parts=[resumed_model])
    resumed.load_state_dict(checkpoint)

    assign_gradients(model, 0.75)
    assign_gradients(resumed_model, 0.75)
    for index, name in enumerate(("weight", "projection")):
        references[name].grad = torch.full((4, 4), 0.75 * (index + 1))
    optimizer.step()
    resumed.step()
    reference.step()

    for name in ("weight", "projection"):
        torch.testing.assert_close(
            getattr(model, name).to_local(), references[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            getattr(resumed_model, name).to_local(),
            getattr(model, name).to_local(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("as_lists", [False, True])
def test_dist_muon_accepts_per_step_ns_coefficients(as_lists):
    model = _model()
    coefficients = ((3.4445, -4.7750, 2.0315), (3.0, -4.0, 2.0))
    config = _config()
    config.param_groups[0].optimizer_kwargs.update(
        ns_steps=2,
        ns_coefficients=[list(step) for step in coefficients]
        if as_lists
        else coefficients,
    )
    optimizer = OptimizersContainer(config, model_parts=[model])
    initial = model.weight.to_local().detach().clone()
    model.weight.grad = DTensor.from_local(
        torch.ones_like(initial), model.weight.device_mesh, model.weight.placements
    )
    optimizer.step()
    assert torch.isfinite(model.weight.to_local()).all()
    assert not torch.equal(initial, model.weight.to_local())


def test_dist_muon_accepts_upstream_zero_epsilon():
    model = _model()
    config = _config()
    config.param_groups[0].optimizer_kwargs["eps"] = 0.0
    optimizer = OptimizersContainer(config, model_parts=[model])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert torch.isfinite(model.weight.to_local()).all()
