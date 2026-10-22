"""The checkpointer's own surface: FQN handling, config validation, the wrapper.

The manager's save/load behaviour is covered in ``test_trainer.py``, where it is
exercised against real DCP round-trips. This file covers what sits underneath:
the pieces that are wrong in ways a round-trip would not reveal -- a config
that accepts a combination it cannot honour, an FQN helper that strips too much
or too little, an optimizer wrapper that reports success without restoring.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("dcp", "dtensor")


import pytest
import torch
import torch.nn as nn

from llmtuner.components.checkpointer import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.base import BaseCheckpointManager
from llmtuner.components.checkpointer.storage import (
    CheckpointStorage,
    FilesystemCheckpointStorage,
)
from llmtuner.components.optimizer import init_optim_state
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.config import CheckpointConfig as Config
from llmtuner.config import LRSchedulerConfig

# -- checkpoint step validation ----------------------------------------------


@pytest.mark.parametrize(
    "dirname,expected",
    [
        ("step-0", 0),
        ("step-12", 12),
        ("step-007", None),
        ("step-1٢", None),
        ("step-1２", None),
        ("step-１２", None),
    ],
)
def test_checkpoint_directory_names_require_canonical_ascii_digits(dirname, expected):
    from types import SimpleNamespace

    manager = SimpleNamespace(_STEP_DIR_PATTERN=BaseCheckpointManager._STEP_DIR_PATTERN)
    assert BaseCheckpointManager._parse_step(manager, dirname) == expected


@pytest.mark.parametrize(
    "method,step",
    [("save", value) for value in (-1, True, 1.5, "2", None)]
    + [("load", value) for value in (-2, True, 1.5, "2", None)],
)
def test_invalid_checkpoint_steps_fail_before_storage_access(method, step):
    from types import SimpleNamespace

    def unexpected(*args):
        pytest.fail("Invalid steps must fail before storage or save policy access")

    manager = SimpleNamespace(
        _initialized=True,
        enable=True,
        _save=unexpected,
        _resolve_load_source=unexpected,
    )
    with pytest.raises(ValueError, match=f"checkpoint {method} step"):
        getattr(BaseCheckpointManager, method)(manager, step)


# -- retention directory ownership --------------------------------------------


@pytest.mark.parametrize("prefix", ["_tmp-", ".pending+"])
def test_retention_only_queues_canonical_abandoned_staging_directories(
    tmp_path, prefix
):
    import queue
    from pathlib import Path
    from types import SimpleNamespace

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = DistributedCheckpointManager(
        Config(enable=True, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
    )
    checkpoint_folder = Path(manager.folder)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    names = [
        f"{prefix}step-{suffix}"
        for suffix in ("0", "12", "20", "007", "1٢", "1２", "１２")
    ]
    names.extend(["step-20", "notes", "step-1٢"])
    for name in names:
        (checkpoint_folder / name).mkdir()
    manager.keep_latest_k = 2
    manager.purge_queue = queue.Queue()
    # Inspect the deletion queue without running destructive background work.
    manager.purge_thread = SimpleNamespace(is_alive=lambda: False)
    try:
        manager._purge_stale_checkpoints(saving_step=20, staging_dir_prefix=prefix)
        queued = set()
        while not manager.purge_queue.empty():
            queued.add(manager.purge_queue.get_nowait())
        assert queued == {
            str(checkpoint_folder / f"{prefix}step-0"),
            str(checkpoint_folder / f"{prefix}step-12"),
        }
        assert all((checkpoint_folder / name).is_dir() for name in names)
    finally:
        manager.close()


# -- config validation --------------------------------------------------------


def test_config_defaults_do_not_validate_at_import() -> None:
    """Constructing the defaults is how every run starts; it must not raise."""
    config = Config()
    assert config.enable is False
    assert config.keep_latest_k == 10


def test_config_rejects_keep_latest_k_of_one() -> None:
    """One retained slot is not a policy -- it is the slot a live save occupies."""
    try:
        Config(keep_latest_k=1)
    except ValueError as error:
        assert "at least 2 replicas" in str(error)
        return
    raise AssertionError("keep_latest_k=1 should be rejected")


def test_config_rejects_a_zero_interval() -> None:
    try:
        Config(interval=0)
    except ValueError as error:
        assert "interval must be >= 1" in str(error)
        return
    raise AssertionError("interval=0 should be rejected")


def test_config_requires_optimizer_exclusion_to_imply_lr_scheduler() -> None:
    try:
        Config(exclude_from_loading=["optimizer"])
    except ValueError as error:
        assert "lr_scheduler" in str(error)
        return
    raise AssertionError("excluding the optimizer alone should be rejected")


def test_config_rejects_a_relative_initial_load_path() -> None:
    try:
        Config(initial_load_path="weights/step-1")
    except ValueError as error:
        assert "absolute path" in str(error)
        return
    raise AssertionError("a relative initial_load_path should be rejected")


def test_config_rejects_hf_quantized_without_hf() -> None:
    try:
        Config(initial_load_in_hf_quantized=True)
    except ValueError as error:
        assert "initial_load_in_hf" in str(error)
        return
    raise AssertionError("quantized-without-hf should be rejected")


def test_config_rejects_hf_safetensors_against_a_remote_folder() -> None:
    """Remote IO supports only the native DCP format."""
    try:
        Config(last_save_in_hf=True, folder="gs://bucket/checkpoints")
    except ValueError as error:
        assert "remote" in str(error)
        return
    raise AssertionError("last_save_in_hf over a remote URI should be rejected")


def test_hf_export_rejects_remote_dump_folder_before_storage_setup():
    from types import SimpleNamespace

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="remote effective checkpoint folder"):
        DistributedCheckpointManager(
            Config(enable=True, last_save_in_hf=True),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder="memory://remote-run",
            sd_adapter=SimpleNamespace(),
        )


def _schedule(optimizer):
    """The lr schedule the manager now requires alongside the optimizer.

    A bare scheduler over the bare ``AdamW`` these cases build: the manager
    checkpoints ``last_epoch`` and nothing else, so the lambda is irrelevant.
    """
    return build_lr_scheduler(
        LRSchedulerConfig(), optimizers=[optimizer], training_steps=8
    )


def test_manager_owns_states_and_orders_core_restore_dependencies(tmp_path):
    from llmtuner.components.checkpointer.checkpoint_keys import (
        LR_SCHEDULER,
        MODEL,
        OPTIMIZER,
    )

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = _schedule(optimizer)
    extra = object()
    supplied = {LR_SCHEDULER: object(), "extra": extra, OPTIMIZER: object()}
    original = supplied.copy()
    manager = DistributedCheckpointManager(
        Config(enable=True),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states=supplied,
        folder=str(tmp_path),
    )
    try:
        assert supplied == original
        assert list(manager.states) == [MODEL, OPTIMIZER, LR_SCHEDULER, "extra"]
        assert manager.states[OPTIMIZER] is optimizer
        assert manager.states[LR_SCHEDULER] is scheduler
        supplied.clear()
        assert manager.states["extra"] is extra
    finally:
        manager.close()


# -- the manager's unwired-option guard ---------------------------------------


def test_hf_options_are_rejected_without_a_state_dict_adapter(tmp_path) -> None:
    """llmtuner ships no adapter, so the HF paths must refuse, not silently no-op."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    try:
        DistributedCheckpointManager(
            Config(enable=True, last_save_in_hf=True),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
    except ValueError as error:
        assert "last_save_in_hf" in str(error)
        return
    raise AssertionError("last_save_in_hf without an sd_adapter should raise")


def test_a_disabled_manager_returns_early_without_building_anything(tmp_path) -> None:
    """``enable=False`` is the default; it must not allocate a purge thread."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = DistributedCheckpointManager(
        Config(enable=False),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
    )
    assert not hasattr(manager, "purge_thread")
    assert manager.save(1) is False
    assert manager.load(-1) is False
    manager.close()  # must be a no-op, not an AttributeError


def test_close_and_the_public_methods_survive_a_failed_constructor(tmp_path) -> None:
    """A manager whose ``__init__`` raised is still called by ``__del__``.

    The HF-options guard raises after ``enable`` and ``_storage`` are assigned
    but before the async futures and retention policy are, so this is the one
    reachable path where a half-built manager is left for the garbage collector.
    """
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = None
    try:
        DistributedCheckpointManager(
            Config(enable=True, async_mode="async", last_save_in_hf=True),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
    except ValueError as error:
        assert "last_save_in_hf" in str(error)
    assert manager is None

    # Rebuild the same half-constructed state and drive the entry points that
    # ``__del__`` and a partially-failed setup would reach.
    half_built = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    half_built.enable = True  # assigned first, so it reads as enabled
    assert half_built.save(1) is False
    assert half_built.load(-1) is False
    half_built.maybe_wait_for_staging()
    half_built.close()


# -- storage protocol ---------------------------------------------------------


def test_filesystem_storage_satisfies_the_protocol(tmp_path) -> None:
    """``runtime_checkable`` catches a rename that would surface mid-save."""
    assert isinstance(FilesystemCheckpointStorage(), CheckpointStorage)


def test_filesystem_storage_reports_paths(tmp_path) -> None:
    storage = FilesystemCheckpointStorage()
    (tmp_path / "step-1").mkdir()
    (tmp_path / "step-1" / ".metadata").write_text("{}")

    assert storage.isdir(str(tmp_path))
    assert storage.isdir(str(tmp_path / "step-1"))
    assert storage.isfile(str(tmp_path / "step-1" / ".metadata"))
    assert not storage.isfile(str(tmp_path / "step-1"))
    assert storage.listdir(str(tmp_path)) == ["step-1"]

    storage.remove(str(tmp_path / "step-1"))
    assert not storage.isdir(str(tmp_path / "step-1"))
    # Deleting again is a no-op rather than a FileNotFoundError in the purge
    # thread, which would otherwise kill retention for the rest of the run.
    storage.remove(str(tmp_path / "step-1"))


# -- init_optim_state ---------------------------------------------------------


def test_init_optim_state_materializes_without_training() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]

    assert optimizer.state_dict()["state"] == {}
    init_optim_state(optimizer)

    assert len(optimizer.state_dict()["state"]) == len(before)
    for original, current in zip(before, model.parameters(), strict=True):
        assert torch.equal(original, current.detach())
    assert [p.grad for p in model.parameters()] == [None, None]


def test_init_optim_state_leaves_the_first_real_update_as_step_one() -> None:
    """Adam's counter must not be advanced by the materializing step."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    init_optim_state(optimizer)

    for state in optimizer.state_dict()["state"].values():
        assert int(state["step"]) == 0
        assert torch.count_nonzero(state["exp_avg"]) == 0


def test_init_optim_state_is_a_no_op_once_state_exists() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()["state"]

    init_optim_state(optimizer)

    for param_id, state in optimizer.state_dict()["state"].items():
        assert torch.equal(state["exp_avg"], saved[param_id]["exp_avg"])
        assert int(state["step"]) == int(saved[param_id]["step"])


def test_init_optim_state_preserves_existing_gradients() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(2, 4)).sum().backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]

    init_optim_state(optimizer)

    for original, param in zip(grads, model.parameters(), strict=True):
        assert torch.equal(original, param.grad)


def test_init_optim_state_completes_a_partially_initialized_optimizer() -> None:
    """A parameter that has not received a gradient still needs checkpoint state.

    Adam initializes state lazily per parameter. A conditional branch can
    therefore leave one parameter cold while another has already stepped; an
    early return based on ``optim.state`` being merely non-empty produces an
    incomplete checkpoint that a fresh optimizer cannot restore.
    """
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    # Initialize only the weight. The bias deliberately receives no gradient.
    model.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.weight in optimizer.state
    assert model.bias not in optimizer.state

    weight_state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[model.weight].items()
    }
    init_optim_state(optimizer)

    assert model.bias in optimizer.state
    for key, expected in weight_state.items():
        actual = optimizer.state[model.weight][key]
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
    # The newly materialized state must still make the next real update step 1.
    assert int(optimizer.state[model.bias]["step"]) == 0


# -- TorchCheckpointingManager ------------------------------------------------
#
# The backend (``torch_checkpointing``) is not a dependency and is not installed
# in the development environment, so nothing here can round-trip a checkpoint.
# What *is* testable is the import guard and the state registration, both of
# which are the parts a missing backend would otherwise let rot silently.


def test_a_disabled_torch_checkpointing_manager_never_touches_the_backend() -> None:
    """``enable=False`` must return before the import.

    This is the property that keeps a disabled manager constructible on a
    machine without the backend; if the import moved above the guard, every
    caller would need the package installed just to not use it.
    """
    from llmtuner.components.checkpointer.torch_checkpointing_backend import (
        TorchCheckpointingManager,
    )

    manager = TorchCheckpointingManager(
        Config(enable=False),
        model_parts=[],
        optimizer=None,
        lr_scheduler=None,
        states={},
        folder="/tmp/unused",
    )
    assert manager.enable is False


def test_an_enabled_torch_checkpointing_manager_without_the_backend_raises() -> None:
    """The failure must be an actionable ImportError, not a bare ModuleNotFoundError.

    ``torch_checkpointing`` is optional, so a missing install is an expected
    state rather than a bug: the message has to name the alternative (the DCP
    manager) instead of surfacing whichever submodule happened to import first.
    """
    import importlib.util

    import pytest

    from llmtuner.components.checkpointer.torch_checkpointing_backend import (
        TorchCheckpointingManager,
    )

    if importlib.util.find_spec("torch_checkpointing") is not None:
        pytest.skip("backend is installed; the guard cannot fire")

    with pytest.raises(ImportError, match="not installed"):
        TorchCheckpointingManager(
            Config(enable=True),
            model_parts=[],
            optimizer=None,
            lr_scheduler=None,
            states={},
            folder="/tmp/unused",
        )


def test_an_enabled_manager_registers_the_lr_scheduler(monkeypatch) -> None:
    """The scheduler must ride along under ``LR_SCHEDULER``, like the DCP manager.

    Without it a resumed run's fresh scheduler restarts ``last_epoch`` at 0, so
    the curve restarts on the step after a resume -- invisible while the lr is
    constant, wrong for the rest of the run once warmup or decay is set.

    The backend is absent, so the import is stubbed to reach the registration.
    Everything after it needs real backend objects and is cut short, but the
    manager-owned states dict is populated first, which is what this pins.
    """
    import pytest

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc
    from llmtuner.components.checkpointer.base import BaseCheckpointManager
    from llmtuner.components.checkpointer.checkpoint_keys import LR_SCHEDULER

    captured_states = {}
    initialize = BaseCheckpointManager.__init__

    def capture_states(manager, *args, **kwargs):
        initialize(manager, *args, **kwargs)
        captured_states.update(manager.states)

    monkeypatch.setattr(BaseCheckpointManager, "__init__", capture_states)

    class _StubBackend:
        def __getattr__(self, name):
            return None

    class _ReachedBackendUse(Exception):
        pass

    def _stop(_backend):
        raise _ReachedBackendUse

    monkeypatch.setattr(tc, "require_torch_checkpointing", _StubBackend)
    monkeypatch.setattr(tc, "async_save_config", _stop)

    sentinel = object()
    states: dict[str, object] = {}
    with pytest.raises(_ReachedBackendUse):
        tc.TorchCheckpointingManager(
            Config(enable=True),
            model_parts=[],
            optimizer=None,
            lr_scheduler=sentinel,
            states=states,
            folder="/tmp/unused",
        )
    assert states == {}
    assert captured_states[LR_SCHEDULER] is sentinel


# -- TorchCheckpointingManager HF initial load --------------------------------
# The backend package is absent here, so these run the manager's logic against
# fakes: ``__new__`` plus attribute stuffing, the same way the Trainer tests
# build theirs. What they pin is llmtuner's half of the port -- the guards, the
# probe, and the adapter round-trip wiring -- not the backend's.


class _FakeStorage:
    def __init__(self, files: set[str]) -> None:
        self._files = files

    def isfile(self, path: str) -> bool:
        return path in self._files


class _FakeAdapter:
    def __init__(self) -> None:
        self.to_hf_calls = 0

    def to_hf(self, state):
        self.to_hf_calls += 1
        return {"hf": state}

    def from_hf(self, state):
        return {"back": state["hf"]}


class _RecordingModelState:
    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state) -> None:
        self.loaded = state


_HF_METADATA = object()


def _hf_manager_stub(tmp_path, monkeypatch, *, files=(), hf_metadata=_HF_METADATA):
    import types

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc
    from llmtuner.components.checkpointer.checkpoint_keys import MODEL

    calls: dict[str, object] = {}
    fake_hf_manager = types.SimpleNamespace(
        load=lambda *a, **kw: calls.update(load_args=a, load_kwargs=kw),
        close=lambda: calls.update(closed=True),
    )
    backend = types.SimpleNamespace(
        HuggingFaceSafetensorsDistributedMetadataFormat=hf_metadata,
        HF_SAFETENSORS_INDEX_FILE_TEMPLATE="{item_key}.safetensors.index.json",
        ItemSpec=lambda **kw: types.SimpleNamespace(**kw),
        METADATA_FILE_NAME="metadata.pkl",
    )
    manager = tc.TorchCheckpointingManager.__new__(tc.TorchCheckpointingManager)
    manager.states = {}
    manager.sd_adapter = _FakeAdapter()
    manager._backend = backend
    manager._storage = _FakeStorage(set(files))
    manager._manager_config = types.SimpleNamespace(
        items={
            MODEL: types.SimpleNamespace(
                requires_copy=True, layout=None, resharder=None, required=False
            )
        },
        storage_config=None,
    )
    manager._manager = types.SimpleNamespace(
        load=lambda *a, **kw: calls.update(native_load=(a, kw))
    )
    monkeypatch.setattr(
        tc.TorchCheckpointingManager,
        "_stateful_to_state_dict",
        staticmethod(lambda states: states),
    )
    monkeypatch.setattr(tc, "sync_save_config", lambda backend, **kw: object())
    monkeypatch.setattr(
        tc,
        "default_backend_config",
        lambda *a, **kw: types.SimpleNamespace(build=lambda: fake_hf_manager),
    )
    return manager, calls, MODEL


def test_hf_load_rejects_quantized_exports(tmp_path, monkeypatch) -> None:
    import pytest

    manager, _, MODEL = _hf_manager_stub(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="quantized"):
        manager._load_checkpoint(
            {MODEL: object()}, str(tmp_path), from_hf=True, from_quantized=True
        )


def test_hf_load_requires_a_state_dict_adapter(tmp_path, monkeypatch) -> None:
    import pytest

    manager, _, MODEL = _hf_manager_stub(tmp_path, monkeypatch)
    manager.sd_adapter = None
    with pytest.raises(ValueError, match="sd_adapter"):
        manager._load_checkpoint(
            {MODEL: object()}, str(tmp_path), from_hf=True, from_quantized=False
        )


def test_hf_load_reports_a_backend_without_hf_metadata(tmp_path, monkeypatch) -> None:
    import pytest

    manager, _, MODEL = _hf_manager_stub(tmp_path, monkeypatch, hf_metadata=None)
    with pytest.raises(ValueError, match="hf.metadata"):
        manager._load_checkpoint(
            {MODEL: object()}, str(tmp_path), from_hf=True, from_quantized=False
        )


def test_hf_load_rejects_a_directory_with_no_export(tmp_path, monkeypatch) -> None:
    import pytest

    manager, _, MODEL = _hf_manager_stub(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not a supported"):
        manager._load_checkpoint(
            {MODEL: object()}, str(tmp_path), from_hf=True, from_quantized=False
        )


def test_hf_load_round_trips_through_the_adapter(tmp_path, monkeypatch) -> None:
    manager, calls, MODEL = _hf_manager_stub(
        tmp_path,
        monkeypatch,
        files={f"{tmp_path}/model.safetensors.index.json"},
    )
    model_state = _RecordingModelState()
    manager._load_checkpoint(
        {MODEL: model_state}, str(tmp_path), from_hf=True, from_quantized=False
    )

    assert manager.sd_adapter.to_hf_calls == 1
    # strict, and the load names the export's safetensors metadata format.
    assert calls["load_kwargs"]["strict"] is True
    assert calls["load_kwargs"]["metadata_format"] is (
        manager._backend.HuggingFaceSafetensorsDistributedMetadataFormat
    )
    # The temporary manager is always closed, and the native manager unused.
    assert calls["closed"] is True
    assert "native_load" not in calls
    # from_hf's output lands on the live model state.
    assert model_state.loaded == {"back": model_state}


def test_native_load_restores_stateful_and_rejects_foreign_objects(
    tmp_path, monkeypatch
) -> None:
    import pytest
    from torch.distributed.checkpoint.stateful import Stateful

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc

    manager, calls, MODEL = _hf_manager_stub(
        tmp_path, monkeypatch, files={f"{tmp_path}/metadata.pkl"}
    )
    # The real helper maps Stateful -> state_dict(); mirror that so the restore
    # below reads a plain dict, like production.
    monkeypatch.setattr(
        tc.TorchCheckpointingManager,
        "_stateful_to_state_dict",
        staticmethod(
            lambda states: {
                k: v.state_dict() if isinstance(v, Stateful) else v
                for k, v in states.items()
            }
        ),
    )

    class _StatefulThing(Stateful):
        def state_dict(self):
            return {"w": 1}

        def load_state_dict(self, state) -> None:
            self.loaded = state

    target = _StatefulThing()
    manager._load_checkpoint(
        {MODEL: target}, str(tmp_path), from_hf=False, from_quantized=False
    )
    assert target.loaded == {"w": 1}  # restored via load_state_dict
    assert calls["native_load"][1]["strict"] is True

    # A plain object the backend did not restore in place must fail loudly.
    manager._manager = type(manager._manager)(load=lambda *a, **kw: {MODEL: object()})
    with pytest.raises(TypeError, match="non-Stateful"):
        manager._load_checkpoint(
            {MODEL: object()}, str(tmp_path), from_hf=False, from_quantized=False
        )


def test_optimizer_initialization_restores_grads_and_lr_on_error(monkeypatch):
    import pytest

    param = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param], lr=0.125)
    original_grad = torch.ones_like(param)
    param.grad = original_grad

    def fail(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        init_optim_state(optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.125
    assert param.grad is original_grad


def test_torch_checkpointing_hf_final_save_builds_item_specs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc
    from llmtuner.components.checkpointer.checkpoint_keys import MODEL

    saved = []
    manager = tc.TorchCheckpointingManager.__new__(tc.TorchCheckpointingManager)
    manager._manager = SimpleNamespace(close=lambda: None)
    manager._manager_config = SimpleNamespace(storage_config=None)
    manager.last_save_model_only = True
    manager.last_save_in_hf = True
    manager.export_dtype = torch.float32
    manager._storage = FilesystemCheckpointStorage()
    manager.folder = str(tmp_path)
    manager.states = {MODEL: torch.nn.Linear(2, 2)}
    manager.sd_adapter = SimpleNamespace(
        to_hf=lambda state: state, fqn_to_index_mapping=None
    )
    manager._backend = SimpleNamespace(
        METADATA_FILE_NAME="metadata.pkl",
        LayoutInfo=lambda *args: object(),
        SafetensorsSerialization=lambda: object(),
        ItemSpec=lambda **kwargs: SimpleNamespace(**kwargs),
        LocalFileSystemStorageConfig=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tc,
        "item_specs",
        lambda backend: {
            MODEL: SimpleNamespace(requires_copy=True, resharder=None, required=False)
        },
    )
    monkeypatch.setattr(tc, "sync_save_config", lambda backend: object())
    monkeypatch.setattr(
        tc,
        "default_backend_config",
        lambda *args, **kwargs: SimpleNamespace(
            build=lambda: SimpleNamespace(
                save=lambda *args: saved.append(args), close=lambda: None
            )
        ),
    )
    monkeypatch.setattr(manager, "_stateful_to_state_dict", lambda states: states)
    manager._save_last_step(2)
    assert len(saved) == 1
    assert saved[0][0].endswith("step-2/sharded")
    assert MODEL in saved[0][1]


def test_checkpoint_close_drains_all_resources_and_preserves_first_error():
    import pytest

    manager = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    manager._initialized = True
    manager.enable = True
    calls = []
    original = ValueError("staging failed")

    def staging():
        calls.append("staging")
        raise original

    def saving():
        calls.append("saving")
        raise RuntimeError("saving failed")

    def cleanup():
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    manager.maybe_wait_for_staging = staging
    manager.maybe_wait_for_saving = saving
    manager._close = cleanup
    with pytest.raises(ValueError, match="staging failed") as caught:
        manager.close()
    assert caught.value is original
    assert calls == ["staging", "saving", "cleanup"]
    manager.close()
    assert calls == ["staging", "saving", "cleanup"]


def test_failed_staging_future_is_consumed_once():
    from concurrent.futures import Future

    import pytest

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointAsyncMode,
    )

    manager = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    manager.async_mode = DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM
    manager.staging_future = Future()
    manager.staging_future.set_exception(ValueError("staging failed"))
    with pytest.raises(ValueError, match="staging failed"):
        manager._maybe_wait_for_staging()
    assert manager.staging_future is None
    manager._maybe_wait_for_staging()


def test_async_checkpoint_round_trip_without_a_process_group(tmp_path):
    import torch.distributed as dist

    assert not dist.is_initialized()
    model = nn.Linear(3, 2)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = Config(enable=True, interval=1, keep_latest_k=0, async_mode="async")
    manager = DistributedCheckpointManager(
        config,
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(tmp_path),
    )
    assert manager.pg is None
    try:
        assert manager.save(1)
    finally:
        manager.close()
    with torch.no_grad():
        model.weight.fill_(0)
        model.bias.fill_(0)
    resumed = DistributedCheckpointManager(
        config,
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(tmp_path),
    )
    try:
        assert resumed.load(1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
    finally:
        resumed.close()


def test_legacy_loader_layout_planner_keeps_model_keys_strict(tmp_path):
    import pytest
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "legacy"
    dcp.save(
        {"dataloader": {"version": 1, "cursor": 7}, "weight": torch.ones(2)},
        checkpoint_id=checkpoint,
    )
    target = {
        "dataloader": {"version": 1, "cursor": 0, "batch_layout": {"tokens": 32}},
        "weight": torch.zeros(2),
    }
    dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())
    assert target["dataloader"]["cursor"] == 7
    torch.testing.assert_close(target["weight"], torch.ones(2))
    target["missing_weight"] = torch.zeros(2)
    with pytest.raises(CheckpointException, match="missing_weight"):
        dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())


def test_partial_loader_layout_metadata_is_rejected(tmp_path):
    import pytest
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "partial"
    dcp.save(
        {"dataloader": {"version": 1, "batch_layout": {"tokens": 32}}},
        checkpoint_id=checkpoint,
    )
    target = {
        "dataloader": {
            "version": 1,
            "batch_layout": {"tokens": 32, "rows": 2},
        }
    }
    with pytest.raises(CheckpointException, match="batch_layout.rows"):
        dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())


def test_process_checkpoint_tracks_upload_when_staging_completes_synchronously(
    tmp_path, monkeypatch
):
    from concurrent.futures import Future

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = DistributedCheckpointManager(
        Config(
            enable=True, interval=1, keep_latest_k=0, async_mode="async_with_pinned_mem"
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
    )
    upload = Future()
    manager.stager = object()
    monkeypatch.setattr(manager, "_save_checkpoint", lambda *args, **kwargs: upload)
    try:
        assert manager.save(1)
        assert manager.save_future is upload
        assert manager.staging_future is None
        assert not upload.done()
        upload.set_result(None)
        manager.maybe_wait_for_saving()
        assert manager.save_future is None
    finally:
        if not upload.done():
            upload.set_result(None)
        manager.stager = None
        manager.close()


def test_async_checkpoint_completion_reports_actual_outcome(
    tmp_path, monkeypatch, caplog
):
    import logging
    from concurrent.futures import CancelledError, Future

    import pytest

    for outcome in ("success", "failure", "cancelled"):
        caplog.clear()
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = DistributedCheckpointManager(
            Config(enable=True, interval=1, keep_latest_k=0, async_mode="async"),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
        future = Future()
        monkeypatch.setattr(
            manager, "_save_checkpoint", lambda *args, future=future, **kwargs: future
        )
        original = OSError("disk full")
        with caplog.at_level(logging.INFO):
            assert manager.save(2)
            if outcome == "success":
                future.set_result(None)
            elif outcome == "failure":
                future.set_exception(original)
            else:
                assert future.cancel()
        success = "Finished writing the checkpoint for step 2"
        assert (success in caplog.text) == (outcome == "success")
        if outcome == "failure":
            assert "Checkpoint write for step 2 failed: disk full" in caplog.text
            with pytest.raises(OSError) as caught:
                manager.maybe_wait_for_saving()
            assert caught.value is original
        elif outcome == "cancelled":
            assert "Checkpoint write for step 2 was cancelled" in caplog.text
            with pytest.raises(CancelledError):
                manager.maybe_wait_for_saving()
        manager.close()


def test_async_checkpoint_close_releases_owned_group_after_stager_failure(tmp_path):
    from types import SimpleNamespace

    import pytest
    import torch.distributed as dist

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    manager = None
    try:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        manager = DistributedCheckpointManager(
            Config(enable=True, keep_latest_k=0, async_mode="async"),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=_schedule(optimizer),
            states={},
            folder=str(tmp_path),
        )
        group = manager.pg
        assert group is not None
        assert dist.get_process_group_ranks(group) == [0]
        original = OSError("stager close failed")

        def close():
            raise original

        manager.stager = SimpleNamespace(close=close)
        with pytest.raises(OSError) as caught:
            manager.close()
        assert caught.value is original
        assert manager.pg is None and manager.stager is None
        with pytest.raises(KeyError):
            dist.get_process_group_ranks(group)
        assert dist.is_initialized() and dist.get_world_size() == 1
        dist.barrier()
        manager.close()
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


def test_failed_retention_thread_start_releases_async_group(tmp_path, monkeypatch):
    import threading

    import pytest
    import torch.distributed as dist

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    created = []
    new_group = dist.new_group
    original = RuntimeError("cannot start retention thread")

    def track_group(*args, **kwargs):
        group = new_group(*args, **kwargs)
        created.append(group)
        return group

    def fail_start(self):
        raise original

    monkeypatch.setattr(dist, "new_group", track_group)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        with pytest.raises(RuntimeError) as caught:
            DistributedCheckpointManager(
                Config(enable=True, keep_latest_k=2, async_mode="async"),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=_schedule(optimizer),
                states={},
                folder=str(tmp_path),
            )
        assert caught.value is original
        assert len(created) == 1
        with pytest.raises(KeyError):
            dist.get_process_group_ranks(created[0])
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_auto_resume_skips_native_model_only_final_export(tmp_path):
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    scheduler = _schedule(optimizer.optimizers[0])
    config = Config(enable=True, interval=1, keep_latest_k=0)
    manager = DistributedCheckpointManager(
        config,
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(tmp_path),
    )
    try:
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        assert manager.save(2, last_step=True)
        export = str(tmp_path / "checkpoint" / "step-2")
        assert manager._is_valid_checkpoint(export)
        assert not manager._is_resumable_checkpoint(export)
        assert manager._find_load_step() == 1
        assert manager.load(-1)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
        # Native model-only exports remain available as initial model weights.
        imported = DistributedCheckpointManager(
            Config(enable=True, initial_load_path=export, keep_latest_k=0),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=str(tmp_path / "imported"),
        )
        try:
            assert imported.load(-1)
            assert not model.weight.any() and not model.bias.any()
        finally:
            imported.close()
    finally:
        manager.close()


@pytest.mark.parametrize("outcome", ["pending", "success", "failure"])
def test_torch_checkpointing_wait_retains_only_active_writes(outcome):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from llmtuner.components.checkpointer.torch_checkpointing_backend import (
        TorchCheckpointingManager,
    )

    future = Future()
    manager = SimpleNamespace(
        save_future=future,
        _manager_config=SimpleNamespace(save=SimpleNamespace(wait_timeout_secs=0)),
    )
    if outcome == "success":
        future.set_result(None)
        TorchCheckpointingManager._wait_for_saving(manager)
    else:
        error = TimeoutError if outcome == "pending" else OSError
        if outcome == "failure":
            future.set_exception(OSError("disk full"))
        with pytest.raises(error):
            TorchCheckpointingManager._wait_for_saving(manager)
    assert manager.save_future is (future if outcome == "pending" else None)
    if outcome == "pending":
        future.set_result(None)
        TorchCheckpointingManager._wait_for_saving(manager)
        assert manager.save_future is None


@pytest.mark.parametrize("phase", ["staging", "saving"])
def test_dcp_interrupted_wait_retains_active_future(phase):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointAsyncMode,
    )

    class InterruptedFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                raise KeyboardInterrupt("interrupted wait")
            return super().result(timeout)

    future = InterruptedFuture()
    field = "staging_future" if phase == "staging" else "save_future"
    manager = SimpleNamespace(
        async_mode=DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM,
        **{field: future},
    )
    wait = (
        DistributedCheckpointManager._maybe_wait_for_staging
        if phase == "staging"
        else DistributedCheckpointManager._wait_for_saving
    )
    with pytest.raises(KeyboardInterrupt, match="interrupted wait"):
        wait(manager)
    assert getattr(manager, field) is future
    future.set_result(None)
    wait(manager)
    assert getattr(manager, field) is None


@pytest.mark.parametrize("phase", ["staging", "saving"])
def test_interrupted_close_preserves_resources_until_future_finishes(phase):
    from concurrent.futures import Future

    from llmtuner.components.checkpointer.distributed_checkpoint import (
        DistributedCheckpointAsyncMode,
    )

    class InterruptedFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                raise KeyboardInterrupt("interrupted close")
            return super().result(timeout)

    future = InterruptedFuture()
    manager = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    manager._initialized = True
    manager.enable = True
    manager.async_mode = DistributedCheckpointAsyncMode.ASYNC_WITH_PINNED_MEM
    manager.staging_future = future if phase == "staging" else None
    manager.save_future = future if phase == "saving" else None
    released = []
    manager._close = lambda: released.append(True)
    with pytest.raises(KeyboardInterrupt, match="interrupted close"):
        manager.close()
    assert released == []
    assert manager._initialized is True
    future.set_result(None)
    manager.close()
    manager.close()
    assert released == [True]
    assert manager._initialized is False


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("succeeds", [False, True])
def test_retention_preserves_last_checkpoint_until_new_save_succeeds(
    tmp_path, asynchronous, succeeds
):
    import queue
    from concurrent.futures import Future
    from pathlib import Path
    from types import SimpleNamespace

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = DistributedCheckpointManager(
        Config(
            enable=True,
            interval=1,
            keep_latest_k=0,
            async_mode="async" if asynchronous else "disabled",
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
    )
    old = Path(manager.folder) / "step-1"
    old.mkdir(parents=True)
    (old / ".metadata").touch()
    manager.keep_latest_k = 1
    manager.purge_queue = queue.Queue()
    manager.purge_thread = SimpleNamespace(is_alive=lambda: False)
    future = Future()

    def save(states, *, checkpoint_id, **kwargs):
        if asynchronous:
            return future
        if not succeeds:
            raise OSError("write failed")
        Path(checkpoint_id).mkdir()
        (Path(checkpoint_id) / ".metadata").touch()

    # These fixtures model publication with marker files, not real DCP metadata.
    manager._is_resumable_checkpoint = manager._is_valid_checkpoint
    manager._save_checkpoint = save
    try:
        if not asynchronous and not succeeds:
            with pytest.raises(OSError, match="write failed"):
                manager.save(2)
        else:
            manager.save(2)
        if asynchronous:
            assert manager.purge_queue.empty()
            if succeeds:
                new = Path(manager.folder) / "step-2"
                new.mkdir()
                (new / ".metadata").touch()
                future.set_result(None)
                manager.maybe_wait_for_saving()
            else:
                future.set_exception(OSError("write failed"))
                with pytest.raises(OSError, match="write failed"):
                    manager.maybe_wait_for_saving()
        if succeeds:
            assert manager.purge_queue.get_nowait() == str(old)
        assert manager.purge_queue.empty()
    finally:
        if asynchronous and not future.done():
            future.set_result(None)
        manager.close()


def test_final_model_export_does_not_evict_last_resumable_checkpoint(tmp_path):
    import queue
    from types import SimpleNamespace

    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    manager = DistributedCheckpointManager(
        Config(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer.optimizers[0]),
        states={},
        folder=str(tmp_path),
    )
    try:
        assert manager.save(1)
        manager.keep_latest_k = 1
        manager.purge_queue = queue.Queue()
        manager.purge_thread = SimpleNamespace(is_alive=lambda: False)
        assert manager.save(2, last_step=True)
        assert manager.purge_queue.empty()
        assert manager._find_load_step() == 1
        assert manager.save(3)
        deleted = set()
        while not manager.purge_queue.empty():
            deleted.add(manager.purge_queue.get_nowait())
        assert deleted == {str(tmp_path / "checkpoint" / "step-1")}
        assert manager._find_load_step() == 3
    finally:
        manager.close()


def test_flattened_model_keys_cannot_overwrite_training_state():
    from types import SimpleNamespace

    from llmtuner.components.checkpointer.checkpoint_keys import MODEL, OPTIMIZER

    manager = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    optimizer_state = object()
    manager.states = {
        OPTIMIZER: optimizer_state,
        MODEL: SimpleNamespace(state_dict=lambda: {OPTIMIZER: torch.ones(2)}),
    }
    with pytest.raises(ValueError, match="Model checkpoint keys.*optimizer"):
        manager._flatten_model_state()
    assert manager.states[OPTIMIZER] is optimizer_state


def test_same_step_final_export_preserves_complete_checkpoint(tmp_path):
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    manager = DistributedCheckpointManager(
        Config(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer.optimizers[0]),
        states={},
        folder=str(tmp_path),
    )
    try:
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1)
        with torch.no_grad():
            model.weight.add_(1)
        assert manager.save(1, last_step=True)
        checkpoint = str(tmp_path / "checkpoint" / "step-1")
        assert manager._is_resumable_checkpoint(checkpoint)
        export = str(tmp_path / "checkpoint" / "step-1" / "model-export")
        assert manager._is_valid_checkpoint(export)
        assert not manager._is_resumable_checkpoint(export)
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
    finally:
        manager.close()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("removed by retention"),
        ValueError("corrupt metadata"),
        PermissionError("metadata access denied"),
    ],
)
def test_resumability_scan_only_tolerates_disappearing_metadata(monkeypatch, error):
    from types import SimpleNamespace

    from torch.distributed.checkpoint import _storage_utils

    def read_metadata():
        raise error

    monkeypatch.setattr(
        _storage_utils,
        "_storage_setup",
        lambda *args, **kwargs: SimpleNamespace(read_metadata=read_metadata),
    )
    manager = SimpleNamespace(
        _storage=SimpleNamespace(isfile=lambda path: True),
        _parse_step=lambda name: 1,
    )
    if isinstance(error, FileNotFoundError):
        assert not DistributedCheckpointManager._is_resumable_checkpoint(
            manager, "checkpoint/step-1"
        )
    else:
        with pytest.raises(type(error), match=str(error)):
            DistributedCheckpointManager._is_resumable_checkpoint(
                manager, "checkpoint/step-1"
            )


@pytest.mark.parametrize("async_mode", ["disabled", "async"])
@pytest.mark.parametrize("failure_phase", ["shards", "publication"])
def test_failed_same_step_save_preserves_published_model(
    tmp_path, monkeypatch, async_mode, failure_phase
):
    from copy import deepcopy

    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer import checkpoint_writer
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    scheduler = _schedule(optimizer.optimizers[0])
    manager = DistributedCheckpointManager(
        Config(enable=True, interval=1, keep_latest_k=0, async_mode=async_mode),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(tmp_path),
    )
    try:
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        expected_optimizer = deepcopy(optimizer.state_dict())
        expected_scheduler = deepcopy(scheduler.state_dict())
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        checkpoint = tmp_path / "checkpoint" / "step-1"
        metadata = (checkpoint / ".metadata").read_bytes()
        with torch.no_grad():
            model.weight.add_(5)
        optimizer.step()
        scheduler.step()
        write_data = FileSystemWriter._write_data

        def fail_after_writing(self, *args, **kwargs):
            write_data(self, *args, **kwargs)
            raise OSError("injected failure after writing shards")

        with monkeypatch.context() as patch:
            if failure_phase == "shards":
                patch.setattr(FileSystemWriter, "_write_data", fail_after_writing)
            else:

                def fail_publication(*args, **kwargs):
                    raise OSError("injected failure publishing metadata")

                patch.setattr(checkpoint_writer.os, "replace", fail_publication)
            with pytest.raises(CheckpointException, match="injected failure"):
                manager.save(1)
                manager.maybe_wait_for_saving()
        assert (checkpoint / ".metadata").read_bytes() == metadata
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), expected_optimizer, rtol=0, atol=0
        )
        assert scheduler.state_dict() == expected_scheduler
        # A retry must publish a fresh generation and remain readable by the
        # ordinary DCP reader, without deleting the earlier generation's files.
        with torch.no_grad():
            model.weight.add_(7)
        expected_retry = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        assert (checkpoint / ".metadata").read_bytes() != metadata
        with torch.no_grad():
            model.weight.zero_()
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), expected_retry, rtol=0, atol=0)
    finally:
        manager.close()


def _read_hf_export_into_fresh_model(tmp_path, model, adapter):
    from copy import deepcopy

    restored = deepcopy(model)
    optimizer = torch.optim.AdamW(restored.parameters())
    manager = DistributedCheckpointManager(
        Config(
            enable=True,
            initial_load_path=str(tmp_path / "checkpoint" / "step-1"),
            initial_load_in_hf=True,
            keep_latest_k=0,
        ),
        model_parts=[restored],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path / "imported"),
        sd_adapter=adapter,
    )
    try:
        assert manager.load(-1)
        return restored.state_dict()
    finally:
        manager.close()


@pytest.mark.parametrize("indexed", [False, True])
def test_hf_export_roundtrip_without_process_group(tmp_path, indexed):
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter

    model = nn.ModuleDict({"model": nn.Linear(2, 2)})
    optimizer = torch.optim.AdamW(model.parameters())
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), None)
    if indexed:
        adapter.fqn_to_index_mapping = {"weight": 1, "bias": 2}
    manager = DistributedCheckpointManager(
        Config(enable=True, last_save_in_hf=True, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
        sd_adapter=adapter,
    )
    try:
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1, last_step=True)
        export = tmp_path / "checkpoint" / "step-1"
        assert manager._is_valid_checkpoint(str(export))
        save_file(
            {"weight": torch.full((2, 2), 999.0), "bias": torch.full((2,), 999.0)},
            str(export / "abandoned.safetensors"),
        )
        with torch.no_grad():
            model["model"].weight.zero_()
        torch.testing.assert_close(
            _read_hf_export_into_fresh_model(tmp_path, model, adapter),
            expected,
            rtol=0,
            atol=0,
        )
    finally:
        manager.close()


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("failure_phase", ["consolidation", "publication"])
def test_failed_hf_export_rewrite_preserves_published_weights(
    tmp_path, monkeypatch, indexed, failure_phase
):
    from types import SimpleNamespace

    from torch.distributed.checkpoint import _consolidate_hf_safetensors
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer import hf_storage
    from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter

    model = nn.ModuleDict({"model": nn.Linear(2, 2)})
    optimizer = torch.optim.AdamW(model.parameters())
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), None)
    if indexed:
        adapter.fqn_to_index_mapping = {"weight": 1, "bias": 2}
    manager = DistributedCheckpointManager(
        Config(enable=True, last_save_in_hf=True, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer),
        states={},
        folder=str(tmp_path),
        sd_adapter=adapter,
    )
    try:
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1, last_step=True)
        with torch.no_grad():
            model["model"].weight.add_(5)
        consolidate = _consolidate_hf_safetensors._consolidate_safetensors_files

        def fail_after_consolidation(*args, **kwargs):
            consolidate(*args, **kwargs)
            raise OSError("injected failure after HF consolidation")

        with monkeypatch.context() as patch:
            if failure_phase == "consolidation":
                patch.setattr(
                    _consolidate_hf_safetensors,
                    "_consolidate_safetensors_files",
                    fail_after_consolidation,
                )
            else:

                def fail_publication(*args, **kwargs):
                    raise OSError("injected failure publishing HF index")

                patch.setattr(hf_storage.os, "replace", fail_publication)
            with pytest.raises(CheckpointException, match="injected failure"):
                manager.save(1, last_step=True)
        torch.testing.assert_close(
            _read_hf_export_into_fresh_model(tmp_path, model, manager.sd_adapter),
            expected,
            rtol=0,
            atol=0,
        )
    finally:
        manager.close()


@pytest.mark.parametrize("protocol", ["memory", "file"])
@pytest.mark.parametrize("async_mode", ["disabled", "async"])
@pytest.mark.parametrize("failure_phase", ["shards", "publication"])
def test_fsspec_same_step_failure_preserves_checkpoint(
    tmp_path, monkeypatch, protocol, async_mode, failure_phase
):
    from uuid import uuid4

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint._fsspec_filesystem import (
        FileSystem as FsspecFileSystem,
    )
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer import checkpoint_writer, filesystem
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    folder = (
        f"memory://llmtuner-tests/{uuid4().hex}"
        if protocol == "memory"
        else (tmp_path / "remote-checkpoint").as_uri()
    )
    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    manager = DistributedCheckpointManager(
        Config(
            enable=True,
            folder=folder,
            interval=1,
            keep_latest_k=0,
            async_mode=async_mode,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer.optimizers[0]),
        states={},
        folder=str(tmp_path),
    )
    fs, checkpoint = filesystem.resolve(filesystem.join(folder, "step-1"))
    metadata_path = f"{checkpoint}/.metadata"
    try:
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        metadata = fs.cat_file(metadata_path)
        with torch.no_grad():
            model.weight.add_(5)
        write_data = FileSystemWriter._write_data

        def fail_after_writing(self, *args, **kwargs):
            write_data(self, *args, **kwargs)
            raise OSError("injected remote shard failure")

        def fail_publication(*args, **kwargs):
            raise OSError("injected remote publication failure")

        with monkeypatch.context() as patch:
            if failure_phase == "shards":
                patch.setattr(FileSystemWriter, "_write_data", fail_after_writing)
            else:
                patch.setattr(FsspecFileSystem, "rename", fail_publication)
                # fsspec caches filesystem instances per thread; the async
                # writer resets its filesystem in the worker thread.
                patch.setattr(type(fs), "pipe_file", fail_publication)
                patch.setattr(checkpoint_writer.os, "replace", fail_publication)
            with pytest.raises(CheckpointException, match="injected remote"):
                manager.save(1)
                manager.maybe_wait_for_saving()
        assert fs.cat_file(metadata_path) == metadata
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
        # An unmodified DCP reader must understand the generated remote files.
        target = {key: torch.zeros_like(value) for key, value in expected.items()}
        dcp.load(target, checkpoint_id=filesystem.join(folder, "step-1"))
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        with torch.no_grad():
            model.weight.add_(7)
        retry = {key: value.clone() for key, value in model.state_dict().items()}
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        dcp.load(target, checkpoint_id=filesystem.join(folder, "step-1"))
        torch.testing.assert_close(target, retry, rtol=0, atol=0)
    finally:
        manager.close()
        manager._storage.remove(folder)


def test_generation_writer_rejects_non_atomic_fsspec_protocol(monkeypatch):
    from types import SimpleNamespace

    from torch.distributed.checkpoint._fsspec_filesystem import (
        FileSystem as FsspecFileSystem,
    )

    from llmtuner.components.checkpointer.checkpoint_writer import (
        GenerationFileSystemWriter,
    )
    from llmtuner.errors import UnsupportedCombinationError

    def init_path(self, path):
        self.fs = SimpleNamespace(protocol="ftp")
        return path

    monkeypatch.setattr(FsspecFileSystem, "init_path", init_path)
    with pytest.raises(UnsupportedCombinationError, match="atomic.*replacement"):
        GenerationFileSystemWriter("ftp://server/checkpoint")


@pytest.mark.parametrize("mode", ["distributed", "process"])
def test_memory_checkpoint_rejects_separate_processes_before_group_setup(
    tmp_path, monkeypatch, mode
):
    import torch.distributed as dist

    from llmtuner.errors import UnsupportedCombinationError

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = _schedule(optimizer)
    if mode == "distributed":
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
        # This unit simulates storage policy only; real coordination is covered
        # by trainer_initialization_coordination.py with an actual Gloo group.
        monkeypatch.setattr(
            "llmtuner.accelerator.collectives.get_world_size", lambda group=None: 1
        )

    def unexpected_group(*args, **kwargs):
        pytest.fail("Invalid memory checkpoint mode must fail before group setup")

    monkeypatch.setattr(dist, "new_group", unexpected_group)
    with pytest.raises(UnsupportedCombinationError, match="memory.*process-local"):
        DistributedCheckpointManager(
            Config(
                enable=True,
                folder="memory://llmtuner-tests/invalid",
                async_mode="async"
                if mode == "distributed"
                else "async_with_pinned_mem",
            ),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=str(tmp_path),
        )


@pytest.mark.parametrize("async_mode", ["disabled", "async"])
def test_remote_lost_publication_acknowledgement_leaves_consistent_checkpoint(
    tmp_path, monkeypatch, async_mode
):
    from uuid import uuid4

    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer import filesystem
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    folder = f"memory://llmtuner-tests/{uuid4().hex}"
    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    manager = DistributedCheckpointManager(
        Config(
            enable=True,
            folder=folder,
            interval=1,
            keep_latest_k=0,
            async_mode=async_mode,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer.optimizers[0]),
        states={},
        folder=str(tmp_path),
    )
    fs, _ = filesystem.resolve(folder)
    publish = type(fs).pipe_file
    try:
        assert manager.save(1)
        manager.maybe_wait_for_saving()
        with torch.no_grad():
            model.weight.add_(5)
        expected = {key: value.clone() for key, value in model.state_dict().items()}

        def publish_then_fail(self, *args, **kwargs):
            publish(self, *args, **kwargs)
            raise OSError("lost remote publication acknowledgement")

        with monkeypatch.context() as patch:
            patch.setattr(type(fs), "pipe_file", publish_then_fail)
            with pytest.raises(CheckpointException, match="lost remote publication"):
                manager.save(1)
                manager.maybe_wait_for_saving()
        with torch.no_grad():
            model.weight.zero_()
        assert manager.load(1)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
    finally:
        manager.close()
        manager._storage.remove(folder)


def test_load_waits_for_pending_upload_before_discovering_latest_checkpoint(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    model = nn.Linear(3, 2)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = DistributedCheckpointManager(
        Config(enable=True, interval=1, keep_latest_k=0, async_mode="async"),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        states={},
        folder=str(tmp_path),
    )
    release = Event()
    executor = ThreadPoolExecutor(max_workers=1)
    save = manager._save_checkpoint
    wait = manager._wait_for_saving
    resolve = manager._resolve_load_source

    def delayed_save(*args, **kwargs):
        actual_upload = save(*args, **kwargs)

        def upload():
            assert release.wait(10), "Load did not wait for the pending upload"
            actual_upload.result()

        return executor.submit(upload)

    def wait_for_upload():
        release.set()
        wait()

    monkeypatch.setattr(manager, "_save_checkpoint", delayed_save)
    monkeypatch.setattr(manager, "_wait_for_saving", wait_for_upload)
    try:
        assert manager.save(1)
        pending = manager.save_future
        assert pending is not None and not pending.done()

        def resolve_after_upload(step):
            assert pending.done(), "Load discovered storage before upload completed"
            return resolve(step)

        monkeypatch.setattr(manager, "_resolve_load_source", resolve_after_upload)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        assert manager.load(-1)
        assert manager.save_future is None
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    finally:
        release.set()
        manager.close()
        executor.shutdown(wait=True)


@pytest.mark.parametrize("failed_phase", [None, "staging", "saving"])
def test_load_synchronizes_snapshot_before_state_or_storage_access(failed_phase):
    from types import SimpleNamespace

    calls = []

    def synchronize(phase):
        calls.append(phase)
        if phase == failed_phase:
            raise RuntimeError(f"{phase} failed")

    def resolve(step):
        calls.append("resolve")
        return None

    manager = SimpleNamespace(
        _initialized=True,
        enable=True,
        maybe_wait_for_staging=lambda: synchronize("staging"),
        maybe_wait_for_saving=lambda: synchronize("saving"),
        _resolve_load_source=resolve,
    )
    if failed_phase:
        with pytest.raises(RuntimeError, match=f"{failed_phase} failed"):
            BaseCheckpointManager.load(manager)
        assert calls == (
            ["staging"] if failed_phase == "staging" else ["staging", "saving"]
        )
    else:
        assert BaseCheckpointManager.load(manager) is False
        assert calls == ["staging", "saving", "resolve"]


def test_compiled_checkpoint_planner_restores_tensors_and_group_options(tmp_path):
    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "compiled"
    dcp.save(
        {
            "_orig_mod.model.weight": torch.tensor([2.0, 3.0]),
            "optimizer": {
                "state._orig_mod.model.weight.exp_avg": torch.tensor([0.5, 0.25]),
                "param_groups._orig_mod.model.weight.lr": 0.125,
            },
            "ema": {"state": {"_orig_mod.model.weight.ema_params": torch.ones(2)}},
        },
        checkpoint_id=checkpoint,
    )
    target = {
        "model.weight": torch.zeros(2),
        "optimizer": {
            "state.model.weight.exp_avg": torch.zeros(2),
            "param_groups.model.weight.lr": 0.0,
        },
        "ema": {"state": {"model.weight.ema_params": torch.zeros(2)}},
    }
    dcp.load(target, checkpoint_id=checkpoint, planner=CheckpointLoadPlanner())
    assert torch.equal(target["model.weight"], torch.tensor([2.0, 3.0]))
    assert torch.equal(
        target["optimizer"]["state.model.weight.exp_avg"], torch.tensor([0.5, 0.25])
    )
    assert target["optimizer"]["param_groups.model.weight.lr"] == 0.125
    assert torch.equal(target["ema"]["state"]["model.weight.ema_params"], torch.ones(2))


def test_compiled_checkpoint_planner_rejects_ambiguous_names(tmp_path):
    import pytest
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "ambiguous"
    dcp.save(
        {"model.weight": torch.ones(2), "_orig_mod.model.weight": torch.zeros(2)},
        checkpoint_id=checkpoint,
    )
    with pytest.raises(CheckpointException, match="Ambiguous compiled checkpoint keys"):
        dcp.load(
            {"model.weight": torch.zeros(2)},
            checkpoint_id=checkpoint,
            planner=CheckpointLoadPlanner(),
        )


@pytest.mark.parametrize("flatten", [True, False])
def test_compiled_checkpoint_flat_scalar_keeps_canonical_target(tmp_path, flatten):
    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "nested-compiled"
    dcp.save(
        {
            "optimizer.param_groups.layers._orig_mod.0._orig_mod.weight.lr": 0.25,
            "model.layers._orig_mod.0.weight": torch.ones(2),
        },
        checkpoint_id=checkpoint,
    )
    target = {
        "optimizer.param_groups.layers.0.weight.lr": 0.0,
        "model.layers.0.weight": torch.zeros(2),
    }
    original_keys = set(target)
    dcp.load(
        target,
        checkpoint_id=checkpoint,
        planner=CheckpointLoadPlanner(flatten_state_dict=flatten),
    )
    assert set(target) == original_keys
    assert target["optimizer.param_groups.layers.0.weight.lr"] == 0.25
    assert torch.equal(target["model.layers.0.weight"], torch.ones(2))


def test_compiled_checkpoint_migration_keeps_missing_keys_strict(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import (
        CheckpointLoadPlanner,
    )

    checkpoint = tmp_path / "incomplete-compiled"
    dcp.save({"_orig_mod.model.weight": torch.ones(2)}, checkpoint_id=checkpoint)
    with pytest.raises(CheckpointException, match="model.bias"):
        dcp.load(
            {"model.weight": torch.zeros(2), "model.bias": torch.zeros(2)},
            checkpoint_id=checkpoint,
            planner=CheckpointLoadPlanner(),
        )


def test_staging_group_construction_failure_releases_upload_group(
    tmp_path, monkeypatch
):
    import torch.distributed as dist

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "staging-group-rendezvous").as_uri(),
        rank=0,
        world_size=1,
    )
    groups = []
    new_group = dist.new_group
    failure = OSError("staging group initialization failed")

    def create_group(*args, **kwargs):
        if groups:
            raise failure
        group = new_group(*args, **kwargs)
        groups.append(group)
        return group

    monkeypatch.setattr(dist, "new_group", create_group)
    try:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        with pytest.raises(OSError) as caught:
            DistributedCheckpointManager(
                Config(enable=True, async_mode="async_with_pinned_mem"),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=_schedule(optimizer),
                states={},
                folder=str(tmp_path),
            )
        assert caught.value is failure
        assert len(groups) == 1
        with pytest.raises(KeyError):
            dist.get_process_group_ranks(groups[0])
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_torch_checkpointing_failed_thread_start_closes_backend(
    tmp_path, monkeypatch, error_type, cleanup_fails
):
    from types import SimpleNamespace

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc

    calls = []
    original = error_type("retention thread start failed")

    def close():
        calls.append("close")
        if cleanup_fails:
            raise OSError("backend close failed")

    backend_manager = SimpleNamespace(close=close)
    storage = SimpleNamespace(create_storage=lambda: object())
    manager_config = SimpleNamespace(
        storage_config=storage, build=lambda: backend_manager
    )
    monkeypatch.setattr(tc, "require_torch_checkpointing", lambda: object())
    monkeypatch.setattr(tc, "async_save_config", lambda backend: object())
    monkeypatch.setattr(tc, "default_backend_config", lambda *a, **kw: manager_config)
    monkeypatch.setattr(
        tc,
        "BackendCheckpointStorage",
        lambda storage: SimpleNamespace(remove=lambda path: None),
    )

    def fail_start(thread):
        raise original

    monkeypatch.setattr(tc.threading.Thread, "start", fail_start)
    # Retain the partial instance to check cleanup without relying on GC timing.
    manager = tc.TorchCheckpointingManager.__new__(tc.TorchCheckpointingManager)
    with pytest.raises(error_type) as caught:
        manager.__init__(
            Config(enable=True, keep_latest_k=2),
            model_parts=[],
            optimizer=None,
            lr_scheduler=None,
            states={},
            folder=str(tmp_path),
        )
    assert caught.value is original
    assert calls == ["close"]
    assert not manager._initialized
    manager.close()
    assert calls == ["close"]


@pytest.mark.parametrize("operation", ["load", "save"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, None])
def test_temporary_backend_cleanup_preserves_operation_error(
    tmp_path, monkeypatch, operation, error_type
):
    from types import SimpleNamespace

    from llmtuner.components.checkpointer import torch_checkpointing_backend as tc

    manager, _, model_key = _hf_manager_stub(
        tmp_path,
        monkeypatch,
        files={f"{tmp_path}/model.safetensors.index.json"},
    )
    original = error_type("checkpoint operation failed") if error_type else None
    cleanup_error = OSError("temporary backend close failed")
    calls = []

    def fail(*args, **kwargs):
        calls.append(operation)
        if original is not None:
            raise original

    def close():
        calls.append("close")
        raise cleanup_error

    temporary = SimpleNamespace(load=fail, save=fail, close=close)
    monkeypatch.setattr(
        tc,
        "default_backend_config",
        lambda *a, **kw: SimpleNamespace(build=lambda: temporary),
    )
    with pytest.raises(error_type or OSError) as caught:
        if operation == "load":
            manager._load_checkpoint(
                {model_key: nn.Linear(2, 2)},
                str(tmp_path),
                from_hf=True,
                from_quantized=False,
            )
        else:
            manager.last_save_model_only = False
            manager.last_save_in_hf = False
            manager.folder = str(tmp_path)
            manager._manager = SimpleNamespace(close=lambda: None)
            manager._save_last_step(2)
    assert caught.value is (original if original is not None else cleanup_error)
    assert calls == [operation, "close"]


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("current_amsgrad", [False, True])
def test_adam_dcp_resume_restores_complete_state_before_next_update(
    tmp_path, amsgrad, current_amsgrad
):
    from copy import deepcopy

    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig

    def runtime(seed, use_amsgrad):
        torch.manual_seed(seed)
        model = nn.Linear(3, 2)
        optimizer = OptimizersContainer(
            OptimizerConfig(
                implementation="for-loop",
                param_groups=[
                    ParamGroupConfig(
                        pattern=".*",
                        optimizer_name="Adam",
                        optimizer_kwargs={
                            "lr": 0.01,
                            "amsgrad": use_amsgrad,
                            "weight_decay": 0.1,
                        },
                    )
                ],
            ),
            model_parts=[model],
        )
        scheduler = build_lr_scheduler(
            LRSchedulerConfig(), optimizers=optimizer.optimizers, training_steps=4
        )
        return model, optimizer, scheduler

    def update(runtime, value):
        model, optimizer, scheduler = runtime
        optimizer.zero_grad(set_to_none=True)
        loss = model(value).square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    def manager(runtime):
        model, optimizer, scheduler = runtime
        return DistributedCheckpointManager(
            Config(enable=True, interval=1, keep_latest_k=0),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=str(tmp_path),
        )

    original = runtime(41, amsgrad)
    update(original, torch.ones(2, 3))
    expected_optimizer = deepcopy(original[1].state_dict())
    writer = manager(original)
    try:
        assert writer.save(1)
    finally:
        writer.close()
    restored = runtime(987, current_amsgrad)
    reader = manager(restored)
    try:
        assert reader.load(1)
    finally:
        reader.close()
    torch.testing.assert_close(
        restored[0].state_dict(), original[0].state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[1].state_dict(), expected_optimizer, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[2].state_dict(), original[2].state_dict(), rtol=0, atol=0
    )
    value = torch.full((2, 3), -0.3)
    torch.testing.assert_close(
        update(restored, value), update(original, value), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[0].state_dict(), original[0].state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored[1].state_dict(), original[1].state_dict(), rtol=0, atol=0
    )


def test_legacy_compiled_amsgrad_destination_keeps_canonical_name(tmp_path):
    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer.load_planner import CheckpointLoadPlanner

    checkpoint = tmp_path / "legacy-amsgrad"
    saved = {
        "optimizer.state._orig_mod.weight.exp_avg_sq": torch.tensor([2.0, 3.0]),
        "optimizer.state._orig_mod.weight.max_exp_avg_sq": torch.tensor([4.0, 5.0]),
    }
    saved = {key.removeprefix("optimizer."): value for key, value in saved.items()}
    destination = {"state.weight.exp_avg_sq": torch.zeros(2)}
    dcp.save({"optimizer": saved}, checkpoint_id=checkpoint)

    class State:
        def state_dict(self):
            return destination

        def load_state_dict(self, state):
            loaded = dict(state)
            destination.clear()
            destination.update(loaded)

    # Stateful callback receives newly planned tensor keys, like production's
    # optimizer container. A plain input dict only copies back existing keys.
    dcp.load(
        {"optimizer": State()},
        checkpoint_id=checkpoint,
        planner=CheckpointLoadPlanner(),
    )
    assert set(destination) == {
        "state.weight.exp_avg_sq",
        "state.weight.max_exp_avg_sq",
    }
    torch.testing.assert_close(
        destination["state.weight.max_exp_avg_sq"], torch.tensor([4.0, 5.0])
    )


def test_amsgrad_new_destination_rejects_ambiguous_compiled_keys(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import CheckpointLoadPlanner

    saved = {
        "optimizer": {
            "state.weight.exp_avg_sq": torch.ones(2),
            "state._orig_mod.weight.exp_avg_sq": torch.ones(2),
            "state.weight.max_exp_avg_sq": torch.ones(2),
            "state._orig_mod.weight.max_exp_avg_sq": torch.zeros(2),
        }
    }
    # Only the newly created maximum state is ambiguous; the regular moment
    # has one key in the checkpoint and is already allocated by the destination.
    del saved["optimizer"]["state._orig_mod.weight.exp_avg_sq"]
    dcp.save(saved, checkpoint_id=tmp_path)
    with pytest.raises(CheckpointException, match="Ambiguous compiled checkpoint keys"):
        dcp.load(
            {"optimizer": {"state.weight.exp_avg_sq": torch.zeros(2)}},
            checkpoint_id=tmp_path,
            planner=CheckpointLoadPlanner(),
        )


@pytest.mark.parametrize("saved_size,target_size", [(2, 3), (3, 2)])
def test_checkpoint_planner_rejects_tensor_shape_mismatch_before_copy(
    tmp_path, saved_size, target_size
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from llmtuner.components.checkpointer.load_planner import CheckpointLoadPlanner

    dcp.save({"weight": torch.ones(saved_size)}, checkpoint_id=tmp_path)
    target = torch.full((target_size,), -7.0)
    with pytest.raises(CheckpointException, match="Size mismatch"):
        dcp.load(
            {"weight": target}, checkpoint_id=tmp_path, planner=CheckpointLoadPlanner()
        )
    torch.testing.assert_close(target, torch.full_like(target, -7.0), rtol=0, atol=0)


@pytest.mark.parametrize("protocol", ["memory", "file"])
@pytest.mark.parametrize("async_mode", ["disabled", "async"])
def test_fsspec_retention_drains_deletions_and_keeps_latest_loadable(
    tmp_path, protocol, async_mode
):
    from uuid import uuid4

    import torch.distributed.checkpoint as dcp

    from llmtuner.components.checkpointer import filesystem
    from llmtuner.components.optimizer.optimizer import OptimizersContainer
    from llmtuner.config import OptimizerConfig

    folder = (
        f"memory://llmtuner-retention/{uuid4().hex}"
        if protocol == "memory"
        else (tmp_path / "uri-retention").as_uri()
    )
    model = nn.Linear(2, 2)
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    manager = DistributedCheckpointManager(
        Config(
            enable=True,
            folder=folder,
            interval=1,
            keep_latest_k=2,
            async_mode=async_mode,
        ),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=_schedule(optimizer.optimizers[0]),
        states={},
        folder=str(tmp_path),
    )
    try:
        for step in range(1, 4):
            with torch.no_grad():
                model.weight.fill_(step)
                model.bias.fill_(-step)
            assert manager.save(step)
            manager.maybe_wait_for_saving()
        expected = {key: value.clone() for key, value in model.state_dict().items()}
        # close must drain the actual retention worker; no sleeps or mocked
        # deletion queue can establish that obsolete remote shards disappeared.
        manager.close()
        assert sorted(filesystem.listdir(folder)) == ["step-2", "step-3"]
        assert not manager.purge_thread.is_alive()
        target = {key: torch.zeros_like(value) for key, value in expected.items()}
        dcp.load(target, checkpoint_id=filesystem.join(folder, "step-3"))
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
    finally:
        manager.close()
        filesystem.rmtree(folder)
