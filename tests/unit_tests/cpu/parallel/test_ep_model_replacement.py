"""The EP swap: a swapped-in llmtuner MoE must reproduce the HF block it replaced.

Single-process, CPU. ``tests/ep_wiring_equivalence.py`` covers the multi-rank
all-to-all; what this suite pins is the swap itself: weight movement, router
parity (softmax scoring, optional top-k renormalization), the refusal paths,
and the load-balance aux loss the swapped router carries.

Comparisons run in float64 where the model allows it. They are not exact: both
HF and llmtuner compute routing scores in fp32 (HF via ``softmax(dtype=float)``,
llmtuner's ``RouterGateLinear`` by construction), but HF computes the gate GEMM in
the model dtype while llmtuner computes it in fp32 -- so the scores agree only to
fp32 rounding. 1e-6 separates that noise floor (~5e-8) from a wiring error
(O(1)).

Every MoE config here requests ``experts_implementation="eager"``. transformers
5.x defaults to ``"grouped_mm"``, which dispatches the experts to
``torch._grouped_mm`` -- bf16-only, and unavailable on CPU entirely. The
auto-fallback to ``"eager"`` that normally saves a CPU run does not fire, because
``_grouped_mm_can_dispatch`` checks the device and the pointer alignment but
never the dtype, so a float32/float64 model passes the gate at ``from_config``
and then dies inside the first forward. Asking for ``"eager"`` explicitly gets
HF's per-expert ``F.linear`` loop, which is the same arithmetic form llmtuner's
``GroupedExperts`` falls back to -- so the two sides are comparable op for op,
which is what makes a bitwise assertion meaningful rather than merely close.
"""

from __future__ import annotations

from llmtuner.errors import UnsupportedCombinationError
from tests.caps import require_env

require_env("spmd_types")


import pytest
import torch
from transformers import AutoConfig

from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.common.moe.experts import RoutedExperts
from llmtuner.models.common.moe.layer import MoE
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.expert_parallel import swap_hf_moe_blocks
from llmtuner.parallel.expert_parallel.block_conversion import (
    restore_fp32_state_buffers,
)
from llmtuner.parallel.expert_parallel.hf_inspection import is_hf_moe_block

TOL = 1e-6


@pytest.fixture(autouse=True)
def _aux_loss_state():
    """Snapshot and restore AuxLoss's class-level state around each test."""
    counts = dict(AuxLoss._group_counts)
    acc = dict(AuxLoss.group_acc)
    denominator = AuxLoss._step_denominator
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss._group_counts.update(counts)
    AuxLoss.group_acc.clear()
    AuxLoss.group_acc.update(acc)
    AuxLoss._step_denominator = denominator


def _config(*, norm_topk_prob: bool, aux_coeff: float = 1e-3) -> AutoConfig:
    """A tiny offline Qwen3Moe: 2 layers, 8 experts, top-2 routing."""
    return AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=norm_topk_prob,
        router_aux_loss_coef=aux_coeff,
        max_position_embeddings=256,
        experts_implementation="eager",
    )


def _model(config, *, seed: int = 0) -> HFTransformerModel:
    """Deterministically initialized tiny Qwen3Moe in float64."""
    torch.manual_seed(seed)
    return HFTransformerModel(config).to(torch.float64).eval()


def _data(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(128, (40,), generator=g)
    return ids, torch.arange(40)


# The decoder-layer attributes a MoE block can live under. transformers 5.x
# keeps every family on ``mlp``; 4.x spells Mixtral's and GraniteMoe's
# ``block_sparse_moe`` and DeepSeek's/GLM4's ``moe`` -- and a DeepSeek layer
# carries a *dense* ``mlp`` as well, so the MoE-shaped one has to win.
BLOCK_ATTRS = ("mlp", "block_sparse_moe", "moe")


def _hf_block(layer: torch.nn.Module) -> torch.nn.Module:
    """The layer's MoE block, under whichever attribute this HF build uses."""
    fallback = None
    for name in BLOCK_ATTRS:
        block = getattr(layer, name, None)
        if block is None:
            continue
        if is_hf_moe_block(block):
            return block
        fallback = fallback or block
    if fallback is None:
        raise AssertionError("layer holds no MoE block attribute")
    return fallback


def _hf_block_attr(layer: torch.nn.Module) -> str:
    """The attribute name holding the layer's block, without probing its layout.

    Used where the layout has been deliberately broken and a probe would raise
    instead of answering.
    """
    for name in BLOCK_ATTRS:
        if getattr(layer, name, None) is not None:
            return name
    raise AssertionError("layer holds no MoE block attribute")


def _hf_experts(block: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's expert weights as ``(gate_up_EF2FD, down_EDF)``, in HF's own terms.

    Read straight off the HF block rather than through llmtuner's probe, so a
    comparison against it stays anchored on what HF computes with: 5.x keeps two
    stacked parameters, 4.x keeps one module per expert.
    """
    experts = block.experts
    gate_up = getattr(experts, "gate_up_proj", None)
    if gate_up is not None:
        return gate_up, experts.down_proj
    for gate_name, up_name, down_name in (
        ("gate_proj", "up_proj", "down_proj"),
        ("w1", "w3", "w2"),
    ):
        if hasattr(experts[0], gate_name):
            return (
                torch.stack(
                    [
                        torch.cat(
                            [
                                getattr(expert, gate_name).weight,
                                getattr(expert, up_name).weight,
                            ],
                            dim=0,
                        )
                        for expert in experts
                    ]
                ),
                torch.stack([getattr(expert, down_name).weight for expert in experts]),
            )
    raise AssertionError(f"unrecognized HF expert layout on {type(experts).__name__}")


def _hf_expert_activation(block: torch.nn.Module):
    """The activation HF's experts apply, wherever this build keeps it."""
    experts = block.experts
    act_fn = getattr(experts, "act_fn", None)
    return act_fn if act_fn is not None else experts[0].act_fn


def _hf_gate_up_parameters(block: torch.nn.Module) -> list[torch.Tensor]:
    """The gate/up projections HF holds, as parameters, in either layout."""
    experts = block.experts
    gate_up = getattr(experts, "gate_up_proj", None)
    if gate_up is not None:
        return [gate_up]
    names = (
        ("gate_proj", "up_proj") if hasattr(experts[0], "gate_proj") else ("w1", "w3")
    )
    return [getattr(expert, name).weight for expert in experts for name in names]


def _hf_down_parameters(block: torch.nn.Module) -> list[torch.Tensor]:
    """The down projections HF holds, as parameters, in either layout."""
    experts = block.experts
    down = getattr(experts, "down_proj", None)
    if down is not None:
        return [down]
    name = "down_proj" if hasattr(experts[0], "down_proj") else "w2"
    return [getattr(expert, name).weight for expert in experts]


def _tie_expert_projection(
    dst_block: torch.nn.Module, src_block: torch.nn.Module, projection: str
) -> None:
    """Tie one expert projection across two blocks, in either HF layout.

    A fused (5.x) block shares the stacked parameter directly; a per-expert
    (4.x) block ties each expert's weight, which the swap then binds to the one
    stacked parameter they were converted into.
    """
    dst, src = dst_block.experts, src_block.experts
    if hasattr(src, "gate_up_proj"):
        setattr(dst, projection, getattr(src, projection))
        return
    if projection == "gate_up_proj":
        names = (
            ("gate_proj", "up_proj") if hasattr(src[0], "gate_proj") else ("w1", "w3")
        )
    else:
        names = ("down_proj",) if hasattr(src[0], "down_proj") else ("w2",)
    for dst_expert, src_expert in zip(dst, src, strict=True):
        for name in names:
            getattr(dst_expert, name).weight = getattr(src_expert, name).weight


def _hf_projection_parameters(
    block: torch.nn.Module, projection: str
) -> list[torch.Tensor]:
    """HF's original parameters for one projection, as a flat unique list."""
    experts = block.experts
    if getattr(experts, "gate_up_proj", None) is not None:
        return [getattr(experts, projection)]
    gate_name, up_name, down_name = (
        ("gate_proj", "up_proj", "down_proj")
        if hasattr(experts[0], "gate_proj")
        else ("w1", "w3", "w2")
    )
    names = (gate_name, up_name) if projection == "gate_up_proj" else (down_name,)
    return [getattr(e, name).weight for name in names for e in experts]


def _hf_projection_tensor(
    block: torch.nn.Module, projection: str, *, grad: bool = False
) -> torch.Tensor:
    """HF's projection as one ``(E, 2F, D)``/``(E, D, F)`` tensor, or its grad.

    Reads the layout HF actually uses: the fused parameter directly, or the
    per-expert weights stacked in ``GroupedExperts``' order (gate half, up
    half) -- the same convention ``_hf_experts`` documents.
    """
    experts = block.experts
    fused = getattr(experts, "gate_up_proj", None)
    if fused is not None:
        tensor = fused if projection == "gate_up_proj" else experts.down_proj
        return tensor.grad if grad else tensor
    gate_name, up_name, down_name = (
        ("gate_proj", "up_proj", "down_proj")
        if hasattr(experts[0], "gate_proj")
        else ("w1", "w3", "w2")
    )
    if projection == "gate_up_proj":
        pieces = [
            torch.cat(
                [g.grad if grad else g, u.grad if grad else u],
                dim=0,
            )
            for e in experts
            for g, u in [(getattr(e, gate_name).weight, getattr(e, up_name).weight)]
        ]
    else:
        pieces = [
            p.grad if grad else p
            for e in experts
            for p in [getattr(e, down_name).weight]
        ]
    return torch.stack(pieces)


def _misshape_hf_experts(block: torch.nn.Module, hidden: int) -> None:
    """Break the gate/up width pairing the reader checks, in either layout.

    The shape check is what stands between a bad layout and a quiet half-copy,
    so it has to be reachable on both the fused (5.x) and per-expert (4.x)
    spellings.
    """
    experts = block.experts
    with torch.no_grad():
        if hasattr(experts, "gate_up_proj"):
            # An odd output dim: not 2F.
            experts.gate_up_proj = torch.nn.Parameter(torch.zeros(8, 7, hidden))
        else:
            first = experts[0]
            name = "gate_proj" if hasattr(first, "gate_proj") else "w1"
            getattr(first, name).weight = torch.nn.Parameter(torch.zeros(7, hidden))


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_swapped_model_matches_hf_output(norm_topk_prob: bool) -> None:
    """Same weights, same tokens: the swap must not change the forward.

    A tolerance rather than equality, for the reason spelled out in the module
    docstring: the reference runs one fused 2F-wide GEMM per expert where the
    swap runs two F-wide ones, so the two sides differ in fp32 accumulation.
    ``test_swap_moves_the_weights_verbatim`` pins the weights themselves, which
    is where an actual wiring error would show.
    """
    config = _config(norm_topk_prob=norm_topk_prob)
    ref = _model(config)
    swapped = _model(config)

    assert swap_hf_moe_blocks(swapped) == 2
    assert all(isinstance(_hf_block(layer), MoE) for layer in swapped.layers)

    ids, positions = _data()
    with torch.no_grad():
        expected = ref(ids, positions=positions)
        got = swapped(ids, positions=positions)
    torch.testing.assert_close(got, expected, rtol=TOL, atol=TOL)


def test_swap_moves_the_weights_verbatim() -> None:
    """w1/w3 are the two halves of ``gate_up_proj``; w2 is ``down_proj``.

    Elementwise, with no transpose: transformers 5.x splits each expert's fused
    projection with ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)``, so the gate
    is the first half of the output dim and the up projection the second. That
    the two halves are not swapped is a property nothing else here would catch,
    since both project into the same ``(F, D)`` shape.
    """
    config = _config(norm_topk_prob=True)
    ref = _model(config)
    swapped = _model(config)
    swap_hf_moe_blocks(swapped)

    hf_block = _hf_block(ref.layers[0])
    moe = _hf_block(swapped.layers[0])
    grouped = moe.routed_experts.inner_experts
    assert torch.equal(moe.router.gate.weight, hf_block.gate.weight)
    gate_up, down = _hf_experts(hf_block)
    hidden = gate_up.shape[1] // 2
    assert torch.equal(grouped.w1_EFD, gate_up[:, :hidden])
    assert torch.equal(grouped.w3_EFD, gate_up[:, hidden:])
    assert torch.equal(grouped.w2_EDF, down)


def test_the_gate_and_up_halves_are_not_interchangeable() -> None:
    """Pins the split against HF itself rather than against our reader.

    The copy above is an identity check -- it re-derives the same halves the
    swap derived, so a reader that had gate and up swapped would satisfy it.
    This one rebuilds an expert's output from the *swapped* tensors using HF's
    own two lines and requires it to equal HF's own output, which holds only
    for the true split.
    """
    config = _config(norm_topk_prob=True)
    ref = _model(config)
    swapped = _model(config)
    swap_hf_moe_blocks(swapped)

    hf_block = _hf_block(ref.layers[0])
    grouped = _hf_block(swapped.layers[0]).routed_experts.inner_experts
    act = _hf_expert_activation(hf_block)
    gate_up, down = _hf_experts(hf_block)

    x = torch.randn(6, config.hidden_size, dtype=torch.float64)
    with torch.no_grad():
        for e in range(config.num_experts):
            gate, up = torch.nn.functional.linear(x, gate_up[e]).chunk(2, dim=-1)
            expected = torch.nn.functional.linear(act(gate) * up, down[e])
            got = torch.nn.functional.linear(
                act(x @ grouped.w1_EFD[e].T) * (x @ grouped.w3_EFD[e].T),
                grouped.w2_EDF[e],
            )
            assert torch.equal(got, expected), f"expert {e} is mis-split"


def test_swap_sets_the_fsdp_moe_flags() -> None:
    """FSDP's MoE branch keys off ``layer.moe_enabled`` / ``layer.moe``.

    Without them the expert weights are sharded as dense parameters over the
    dense DP mesh, mixing ranks of different EP coordinates into one FSDP
    group. ``moe`` must be a plain attribute, not a registered submodule: it
    aliases ``layer.mlp``, and registering it would double every expert weight
    in the state_dict.
    """
    swapped = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(swapped)

    for layer in swapped.layers:
        assert layer.moe_enabled is True
        assert layer.moe is layer.mlp
        assert "moe" not in layer._modules
        assert "moe" not in dict(layer.named_modules())


def test_swap_preserves_eval_mode() -> None:
    """A fresh module defaults to training=True; the swap must not flip an
    eval-built model's blocks (the aux loss would fire without a denominator)."""
    swapped = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(swapped)
    assert not any(_hf_block(layer).training for layer in swapped.layers)

    ids, positions = _data()
    with torch.no_grad():
        swapped(ids, positions=positions)  # must not raise for a denominator


def test_a_router_with_bias_is_refused() -> None:
    """``RouterGateLinear`` has no bias slot; dropping one would change scores.

    No supported family has a biased router, so the guard fires only on a
    family the probe does not know -- pinned here so it cannot rot into a
    silent drop.
    """
    model = _model(_config(norm_topk_prob=True))
    gate = model.layers[0].mlp.gate
    gate.bias = torch.nn.Parameter(
        torch.zeros(gate.weight.shape[0], dtype=gate.weight.dtype)
    )

    with pytest.raises(NotImplementedError, match="router bias"):
        swap_hf_moe_blocks(model)


def test_swap_rejects_a_dense_model() -> None:
    """EP on a model with no MoE block is a config mistake; refuse loudly."""
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = _model(config)
    with pytest.raises(UnsupportedCombinationError, match="no HF MoE block"):
        swap_hf_moe_blocks(model)


def test_swap_rejects_a_model_without_layers() -> None:
    with pytest.raises(UnsupportedCombinationError, match=r"\.layers"):
        swap_hf_moe_blocks(torch.nn.Linear(4, 4))


def test_aux_loss_is_injected_on_the_router_scores() -> None:
    """Training forward: the load-balance loss accumulates a metric and moves
    the router's gradient, while the forward output is bitwise unchanged."""
    ids, positions = _data()

    with_aux = _model(_config(norm_topk_prob=True, aux_coeff=1e-3))
    swap_hf_moe_blocks(with_aux)
    without_aux = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    swap_hf_moe_blocks(without_aux)
    assert without_aux.layers[0].mlp.router.aux_loss is None

    AuxLoss.set_step_denominator(torch.tensor(39.0))
    with_aux.train()
    without_aux.train()
    out_with = with_aux(ids, positions=positions)
    out_without = without_aux(ids, positions=positions)
    # Identity forward: the aux loss injects a gradient, not a value.
    assert torch.equal(out_with, out_without)
    out_with.sum().backward()
    out_without.sum().backward()

    router_with = with_aux.layers[0].mlp.router
    router_without = without_aux.layers[0].mlp.router
    assert router_with.aux_loss.instance_acc.item() > 0
    grad_diff = (
        (router_with.gate.weight.grad - router_without.gate.weight.grad)
        .abs()
        .max()
        .item()
    )
    assert grad_diff > 0


def test_aux_loss_requires_the_step_denominator() -> None:
    """Training forward without ``set_step_denominator`` must refuse."""
    model = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(model)
    model.train()
    ids, positions = _data()
    with pytest.raises(ValueError, match="set_step_denominator"):
        model(ids, positions=positions)


# -- DeepSeek-V3: a different block shape --------------------------------------
#
# DeepSeek-V3 differs from Qwen3Moe in two ways the swap has to absorb: the
# router is a bespoke module rather than an ``nn.Linear``, and it carries an
# ``e_score_correction_bias`` buffer. It also has *no* ``router_aux_loss_coef``
# in its HF config, unlike Qwen3Moe.
#
# transformers 5.x moved the routing attributes (``top_k``, ``n_group``,
# ``topk_group``, ``norm_topk_prob``, ``routed_scaling_factor``) onto the *block*
# for every family; in 4.x DeepSeek kept them on the router. The probe reads
# both locations, and ``test_deepseek_v3_routing_attributes_live_on_the_block``
# pins the fact so a future move shows up as a failure rather than as the probe
# quietly falling back.
#
# These run in float32 rather than float64: HF's own ``DeepseekV3MoE.moe``
# allocates its accumulator as ``torch.zeros_like(hidden_states,
# dtype=topk_weights.dtype)``, and ``topk_weights`` is fp32 on every path, so a
# float64 DeepSeek fails in *its own* forward (index_add_ dtype error) before
# any swapped code runs.


def _deepseek_config(**overrides) -> AutoConfig:
    """A tiny offline DeepSeek-V3: 3 layers, first dense, 8 experts, 2 groups."""
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    settings.update(overrides)
    return AutoConfig.for_model("deepseek_v3", **settings)


def _deepseek_model(**overrides) -> HFTransformerModel:
    torch.manual_seed(0)
    return HFTransformerModel(_deepseek_config(**overrides)).float().eval()


def test_deepseek_v3_routing_attributes_are_probeable() -> None:
    """Dependency versions may place routing metadata on block or router."""
    model = _deepseek_model()
    block, router = model.layers[1].mlp, model.layers[1].mlp.gate
    owners = (block, router, model.model.config)
    for aliases in (
        ("top_k", "num_experts_per_tok"),
        ("n_group", "num_group"),
        ("topk_group",),
        ("norm_topk_prob",),
    ):
        assert any(
            getattr(owner, attr, None) is not None
            for owner in owners
            for attr in aliases
        ), f"{aliases} are absent from all supported owners"


@pytest.mark.parametrize(
    "overrides",
    [
        {"norm_topk_prob": True, "routed_scaling_factor": 2.5},
        {"norm_topk_prob": False, "routed_scaling_factor": 2.5},
        {"norm_topk_prob": True, "routed_scaling_factor": 1.0},
        {"scoring_func": "softmax", "norm_topk_prob": True},
    ],
)
def test_deepseek_v3_swap_matches_hf_output(overrides: dict) -> None:
    """Same weights, same tokens: the swap must not change DeepSeek's forward.

    Not bitwise, and the reason is a *shape* difference rather than a wrong
    weight. transformers 5.x fuses an expert's gate and up projections into one
    ``(E, 2F, D)`` tensor and applies them in a single ``F.linear`` of output
    width 2F, splitting afterwards; llmtuner keeps them apart and runs two GEMMs
    of width F. The operands are the same numbers -- ``GateUpProj[:, :F]`` is
    literally ``w1_EFD`` -- but fp32 accumulates over D in a different
    association, so the two agree only to rounding. In float64 the difference is
    exactly 0, which is what identifies it as accumulation order rather than an
    arithmetic error.

    The bar is therefore ``TOL`` at both ends: a tolerance that admits the
    rounding but not a wiring error, since 2e-7 against outputs of magnitude
    0.45 still separates a wrong half or a wrong expert (O(1)) from this noise.
    """
    config = _deepseek_config(**overrides)
    torch.manual_seed(0)
    ref = HFTransformerModel(config).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(config).float().eval()

    # first_k_dense_replace=1, so only the two sparse layers swap.
    assert swap_hf_moe_blocks(swapped) == 2
    assert not isinstance(swapped.layers[0].mlp, MoE)
    assert all(isinstance(layer.mlp, MoE) for layer in swapped.layers[1:])

    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_deepseek_v3_router_settings_reach_the_llmtuner_router() -> None:
    """The probed routing config, as the swapped router ends up holding it.

    Every one of these is read from a *different* place in the HF model (the
    block for the numbers, a buffer on the router for the scoring function), so
    this is the end-to-end check that the probe assembles them correctly.
    """
    model = _deepseek_model()

    swap_hf_moe_blocks(model)
    router = model.layers[1].mlp.router
    assert router.score_func == "sigmoid"
    assert router.route_norm is True
    assert router.route_scale == 2.5
    assert router.num_expert_groups == 2
    assert router.num_limited_groups == 1


def test_deepseek_v3_shared_experts_survive_the_swap() -> None:
    """DeepSeek has an (ungated) shared expert; the swap must carry it over.

    The spelling is ``shared_experts``, plural. Qwen3Moe's blocks have neither,
    so a probe that looked only for the singular would drop this one in silence
    and change the forward by a whole FFN.
    """
    model = _deepseek_model()
    assert model.layers[1].mlp.shared_experts is not None

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.shared_experts is not None


def test_swap_preserves_shared_expert_buffer_dtype() -> None:
    model = _deepseek_model()
    shared = model.layers[1].mlp.shared_experts
    shared.register_buffer("calibration", torch.tensor([1.25], dtype=torch.float64))

    swap_hf_moe_blocks(model)

    assert model.layers[1].mlp.shared_experts is shared
    assert shared.calibration.dtype == torch.float64
    assert shared.calibration.item() == 1.25
    assert model.layers[1].mlp.tokens_per_expert_E.dtype == torch.float32


def test_swap_preserves_falsey_shared_expert() -> None:
    model = _deepseek_model()
    shared = torch.nn.Sequential()
    model.layers[1].mlp.shared_expert = shared

    swap_hf_moe_blocks(model)

    assert model.layers[1].mlp.shared_experts is shared


def test_deepseek_v3_gets_the_load_balancing_bias() -> None:
    """The probed ``e_score_correction_bias`` turns the bias path on -- and only there.

    Qwen3Moe has no such buffer, so the coefficient is decided per block from
    the probed router rather than globally. Otherwise every model would get a
    bias, and a frozen zero one would land in every checkpoint.
    """
    model = _deepseek_model()
    swap_hf_moe_blocks(model)

    sparse = [layer.mlp for layer in model.layers if isinstance(layer.mlp, MoE)]
    assert all(moe.load_balance_coeff is not None for moe in sparse)
    assert all(moe.expert_bias_E is not None for moe in sparse)

    qwen = _model(_config(norm_topk_prob=True))
    swap_hf_moe_blocks(qwen)
    assert all(layer.mlp.load_balance_coeff is None for layer in qwen.layers)


def test_the_bias_is_copied_out_of_the_hf_router() -> None:
    """The bias is optimization state, so a swap must not silently reset it."""
    model = _deepseek_model()
    hf_router = model.layers[1].mlp.gate
    with torch.no_grad():
        hf_router.e_score_correction_bias.copy_(
            torch.arange(8, dtype=torch.float32) / 100
        )

    swap_hf_moe_blocks(model)
    assert torch.equal(
        model.layers[1].mlp.expert_bias_E, hf_router.e_score_correction_bias
    )


def test_deepseek_v3_gets_no_aux_loss_from_its_own_config() -> None:
    """Its HF config has no ``router_aux_loss_coef``, so the loss is absent.

    A real gap rather than an oversight in the swap: DeepSeek-V3's design calls
    for a sequence-wise balance loss, and its config simply has no field to
    carry the coefficient. ``router_aux_loss_coef`` on the swap is the knob that
    supplies one; without it the router carries no aux loss at all.
    """
    model = _deepseek_model()
    assert getattr(model.model.config, "router_aux_loss_coef", None) is None

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.router.aux_loss is None

    explicit = _deepseek_model()
    swap_hf_moe_blocks(explicit, router_aux_loss_coef=1e-3)
    assert explicit.layers[1].mlp.router.aux_loss is not None


def test_the_swap_refuses_to_guess_a_shared_expert_gate() -> None:
    """A gated shared expert is the one shared-expert shape MoE cannot express.

    ``MoE.shared_experts`` is additive; a ``shared_expert_gate`` multiplies.
    Dropping the gate would change the forward by a whole FFN's worth of
    scaling, so refusing is the point. No architecture in the supported set hits
    it (Qwen3.5 does), but the guard is pinned so it cannot rot unnoticed.
    """
    model = _deepseek_model()
    model.layers[1].mlp.shared_expert_gate = torch.nn.Linear(64, 1)
    counts_before = dict(AuxLoss._group_counts)

    with pytest.raises(NotImplementedError, match="shared_expert_gate"):
        swap_hf_moe_blocks(model, router_aux_loss_coef=1e-3)
    assert dict(AuxLoss._group_counts) == counts_before


# -- the other families: where the block lives, and what the experts are called -


def _plain_settings(**overrides) -> dict:
    """DeepSeek-shaped settings, minus ``model_type`` (see ``_plain_model``)."""
    settings = _deepseek_config(**overrides).to_dict()
    settings.pop("model_type", None)
    return settings


def _plain_model(family: str, **overrides) -> HFTransformerModel:
    """A tiny model built straight from ``family``'s own config.

    ``to_dict()`` carries ``model_type``, which collides with ``for_model``'s
    first positional parameter -- so it is dropped before forwarding.
    """
    cfg = AutoConfig.for_model(family, **_plain_settings(**overrides))
    torch.manual_seed(0)
    return HFTransformerModel(cfg).float().eval()


def _mixtral_model(**overrides) -> HFTransformerModel:
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    settings.update(overrides)
    torch.manual_seed(0)
    return (
        HFTransformerModel(AutoConfig.for_model("mixtral", **settings)).float().eval()
    )


@pytest.mark.parametrize("family", ["glm4_moe", "deepseek_v2"])
def test_the_deepseek_router_shape_swaps_exactly(family: str) -> None:
    """GLM4 and DeepSeek-V2 share V3's block shape but not all of its rules.

    GLM4's routing is V3's with the bias rename; DeepSeek-V2's *default*
    ``topk_method`` ("greedy") routes freely over all experts even though its
    block carries ``num_group``/``topk_group``. Reading those grouping
    attributes without checking the method confines every token to a group HF
    never imposes -- a routing change, so it moves the output far more than the
    fused-versus-split GEMM rounding this asserts within (module docstring).
    """
    cfg = AutoConfig.for_model(family, **_plain_settings())
    torch.manual_seed(0)
    ref = HFTransformerModel(cfg).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(cfg).float().eval()

    assert swap_hf_moe_blocks(swapped) == 2

    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_deepseek_v2_greedy_leaves_the_grouping_off() -> None:
    """The attributes are present but unused; the swap must not switch them on.

    This is the shape of the bug the test above catches: the probe reads
    ``num_group`` off the block and gets a number that looks meaningful, while
    HF's ``greedy`` path never consults it.
    """
    model = _plain_model("deepseek_v2")
    block, gate = model.layers[1].mlp, model.layers[1].mlp.gate
    assert (
        getattr(block, "topk_method", None) or getattr(gate, "topk_method", None)
    ) == "greedy"
    assert (
        getattr(block, "num_group", None) or getattr(gate, "num_group", None)
    ) == 2  # present, unused

    swap_hf_moe_blocks(model)
    router = model.layers[1].mlp.router
    assert router.num_expert_groups is None
    assert router.num_limited_groups is None


def test_deepseek_v2_group_limited_greedy_is_refused() -> None:
    """Its rule is a group ``max``; this swap implements V3's top-2 sum.

    Both are "group-limited", so nothing in the config distinguishes them and
    the wrong rule would just pick different experts -- a silent numeric drift.
    """
    model = _plain_model("deepseek_v2", topk_method="group_limited_greedy")

    with pytest.raises(NotImplementedError, match="single best expert"):
        swap_hf_moe_blocks(model)


def test_deepseek_v2_does_not_renormalize() -> None:
    """Its routing ignores ``norm_topk_prob`` entirely.

    transformers 5.x moved ``topk_method`` off the gate and onto the block, and
    dropped ``norm_topk_prob`` from DeepSeek-V2 altogether -- the routing never
    read it. A guard that looked for the declared field therefore stops working
    on the upgrade *while still looking correct*, and the swap silently starts
    renormalizing weights HF leaves raw. The marker has to be ``topk_method``,
    wherever the version puts it.
    """
    model = _plain_model("deepseek_v2")
    block, gate = model.layers[1].mlp, model.layers[1].mlp.gate
    assert (
        getattr(block, "topk_method", None) or getattr(gate, "topk_method", None)
    ) == "greedy"

    swap_hf_moe_blocks(model)
    assert model.layers[1].mlp.router.route_norm is False


def test_mixtral_swaps_in_place() -> None:
    """The replacement must land in the slot HF's own decoder layer calls.

    Which slot that is depends on the transformers line: 5.x moved every family
    onto ``mlp``, 4.x keeps Mixtral on ``block_sparse_moe``. Writing to a
    different attribute would leave the original block in place and route around
    the swap entirely, so the assertion is that the layer's MoE slot holds a
    llmtuner block and that a forward still reaches it.
    """
    model = _mixtral_model()
    assert type(_hf_block(model.layers[0])).__name__ == "MixtralSparseMoeBlock"

    assert swap_hf_moe_blocks(model) == 2

    for layer in model.layers:
        assert isinstance(_hf_block(layer), MoE)

    ids, positions = _data()
    with torch.no_grad():
        out = model(ids, positions=positions)
    assert out.shape == (40, 128)


def test_mixtral_matches_hf_output() -> None:
    """The one family whose per-expert names match ``GroupedExperts``'.

    Mixtral's ``w1``/``w2``/``w3`` are the least likely to be mis-mapped by a
    reader -- and the most dangerous to mis-map, because ``w2`` (the ``(F, D)``
    down projection) fits ``w3_EFD``'s ``(F, D)`` slot exactly. A forward
    comparison against HF is what catches that, shapes being no help at all.

    A tolerance rather than equality: same fused-versus-split GEMM difference as
    every other family (module docstring). ``2F == D`` here, so an argument from
    the shapes being degenerate does not apply -- the accumulation order still
    differs, because HF splits *after* the 2F-wide GEMM and llmtuner never forms
    it.
    """
    cfg = AutoConfig.for_model(
        "mixtral",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        experts_implementation="eager",
    )
    torch.manual_seed(0)
    ref = HFTransformerModel(cfg).float().eval()
    torch.manual_seed(0)
    swapped = HFTransformerModel(cfg).float().eval()

    swap_hf_moe_blocks(swapped)
    ids, positions = _data()
    with torch.no_grad():
        torch.testing.assert_close(
            swapped(ids, positions=positions),
            ref(ids, positions=positions),
            rtol=TOL,
            atol=TOL,
        )


def test_mixtral_returns_a_bare_tensor() -> None:
    """transformers 5.x dropped the ``(hidden, router_logits)`` tuple.

    Its decoder layer used to unpack two values; the swap used to wrap the
    block to keep supplying them. Both are gone, so the block must be a plain
    ``MoE`` -- a surviving wrapper would fail the other way round.
    """
    model = _mixtral_model()
    swap_hf_moe_blocks(model)

    ids, positions = _data()
    with torch.no_grad():
        out = model(ids, positions=positions)
    assert out.shape == (40, 128)


def test_gpt_oss_is_refused() -> None:
    """Its experts are transposed and carry biases; ``GroupedExperts`` has neither.

    ``gate_up_proj`` is ``(E, D, 2F)`` rather than ``(E, 2F, D)``, so the shared
    split convention would slice the *token* dim and then copy a plausibly-shaped
    wrong tensor. Silently converting it is therefore not an option, and the
    refusal has to reach the caller as a refusal rather than as a "not a MoE
    block" skip that leaves the model running with unswapped experts.

    Built from ``GptOssExperts`` directly rather than as a whole model. A
    ``GptOssForCausalLM`` cannot be instantiated on CPU at all -- HF's
    ``_supports_sdpa`` guard rejects the eager attention this build falls back
    to -- and the layout is the thing under test, not the model around it. The
    block is hand-assembled because ``swap_hf_moe_blocks`` walks ``.layers``,
    which a bare experts module does not have.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    config = AutoConfig.for_model(
        "gpt_oss",
        hidden_size=64,
        intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    block = torch.nn.Module()
    block.experts = GptOssExperts(config)
    assert block.experts.gate_up_proj.shape[1] == 64  # (E, D, 2F), not (E, 2F, D)

    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleList([torch.nn.Module()])
    holder.layers[0].mlp = block

    with pytest.raises(NotImplementedError, match="bias vectors"):
        swap_hf_moe_blocks(holder)


def test_a_block_with_the_wrong_expert_layout_is_rejected() -> None:
    """A misshapen ``gate_up_proj`` must not be sliced on a guess.

    Refusing beats guessing: the shape check is what stands between a bad
    layout and a quiet half-copy, and it has to fail as a shape error rather
    than as an index error somewhere deeper.
    """
    model = _deepseek_model()
    _misshape_hf_experts(_hf_block(model.layers[1]), hidden=64)

    with pytest.raises(ValueError, match="unrecognized expert weight shapes"):
        swap_hf_moe_blocks(model)


def test_late_invalid_expert_layout_does_not_replace_earlier_layers() -> None:
    model = _model(_config(norm_topk_prob=True))
    # Attribute names, not probes: the block about to be misshaped can no
    # longer answer a layout probe, and the point is that it is still *there*.
    attrs_before = [_hf_block_attr(layer) for layer in model.layers]
    blocks_before = [
        getattr(layer, name)
        for layer, name in zip(model.layers, attrs_before, strict=True)
    ]
    counts_before = dict(AuxLoss._group_counts)
    _misshape_hf_experts(_hf_block(model.layers[1]), hidden=64)

    with pytest.raises(ValueError, match="unrecognized expert weight shapes"):
        swap_hf_moe_blocks(model)

    assert all(
        getattr(layer, name) is block
        for layer, name, block in zip(
            model.layers, attrs_before, blocks_before, strict=True
        )
    )
    assert all(not getattr(layer, "moe_enabled", False) for layer in model.layers)
    assert dict(AuxLoss._group_counts) == counts_before


def test_the_swap_keeps_the_token_count_buffer_in_fp32() -> None:
    """The swapped MoE's state buffers must survive ``.to(dtype=...)`` unscathed.

    ``Module.to`` converts every floating-point buffer along with the
    parameters, and ``convert_block`` casts the new MoE to match the HF block's
    dtype. In a bf16 run that silently demoted the load-balancing buffers:
    ``tokens_per_expert_E`` is a token *count*, which bf16 cannot hold exactly
    past 256 -- 1001 becomes 1000 -- and ``expert_bias_E`` is an additive
    correction a bf16 round per step would erode. Neither has a gradient, so the
    cast bought nothing. llmtuner is fp32-only today, which is exactly why this
    needs a test: the bug is invisible until a bf16 path exists, and then it is
    silent.

    Qwen3Moe carries no ``e_score_correction_bias``, so only the count buffer is
    present here; ``restore_fp32_state_buffers`` covers both and
    ``test_the_fp32_restore_covers_the_bias_buffer`` pins the other.
    """
    swapped = _model(_config(norm_topk_prob=True))

    assert swap_hf_moe_blocks(swapped) == 2

    moe = swapped.layers[0].mlp
    assert isinstance(moe, MoE)
    assert moe.tokens_per_expert_E.dtype == torch.float32

    # And the conversion really was asked for: the parameters did follow the
    # block, so this is not passing because nothing was cast at all.
    assert moe.router.gate.weight.dtype == torch.float64  # nosec

    # A count bf16 would have mangled round-trips exactly.
    moe.tokens_per_expert_E += 1001
    assert moe.tokens_per_expert_E[0].item() == 1001.0


def test_the_fp32_restore_covers_the_bias_buffer() -> None:
    """``restore_fp32_state_buffers`` is what the check above relies on.

    Built directly rather than through the swap, because the families that carry
    ``expert_bias_E`` are a larger fixture than this property needs. The point is
    narrow and worth isolating: yes to float buffers, no to a plain
    ``Module.to``.
    """
    from llmtuner.models.common.moe.experts import GroupedExperts
    from llmtuner.models.common.moe.routing import TokenChoiceTopKRouter
    from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher

    grouped = GroupedExperts(dim=16, hidden_dim=32, num_experts=256)
    moe = MoE(
        num_experts=256,
        routed_experts=RoutedExperts(grouped, LocalTokenDispatcher(256, 1)),
        router=TokenChoiceTopKRouter(num_experts=256, dim=16),
        load_balance_coeff=1e-3,
    )
    assert moe.expert_bias_E is not None, "no bias registered -- vacuous"

    # Without the restore, this is what the swap used to do.
    moe.to(dtype=torch.bfloat16)
    assert moe.expert_bias_E.dtype == torch.bfloat16  # the bug, demonstrated

    moe.to(dtype=torch.bfloat16)  # re-cast so the fix under test starts from it
    restore_fp32_state_buffers(moe)

    assert moe.expert_bias_E.dtype == torch.float32
    assert moe.tokens_per_expert_E.dtype == torch.float32
    assert moe.router.gate.weight.dtype == torch.bfloat16  # params still follow

    # Exactness, which is the whole reason the dtype matters.
    # 0.012345 as bf16 is 0.012329; as fp32 it is exact to float32 precision.
    moe.expert_bias_E[0] = 0.012345
    moe.tokens_per_expert_E += 1001
    assert (
        moe.expert_bias_E[0].item()
        == torch.tensor(0.012345, dtype=torch.float32).item()
    )
    assert moe.tokens_per_expert_E[0].item() == 1001.0


def test_ep_conversion_preserves_parameter_freezing():
    model = _model(_config(norm_topk_prob=True))
    for layer in model.layers:
        block = _hf_block(layer)
        block.gate.weight.requires_grad_(False)
        for weight in _hf_gate_up_parameters(block):
            weight.requires_grad_(False)
    assert swap_hf_moe_blocks(model) == 2
    for layer in model.layers:
        assert not layer.mlp.router.gate.weight.requires_grad
        experts = layer.mlp.routed_experts.inner_experts
        assert not experts.w1_EFD.requires_grad
        assert not experts.w3_EFD.requires_grad
        assert experts.w2_EDF.requires_grad


def test_interrupted_conversion_restores_auxiliary_counts(monkeypatch):
    from llmtuner.parallel.expert_parallel import model_replacement

    model = _model(_config(norm_topk_prob=True))
    original_blocks = [layer.mlp for layer in model.layers]
    counts_before = dict(AuxLoss._group_counts)
    convert = model_replacement.convert_block
    calls = 0

    def interrupted(block, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("conversion interrupted")
        return convert(block, **kwargs)

    monkeypatch.setattr(model_replacement, "convert_block", interrupted)
    with pytest.raises(KeyboardInterrupt, match="conversion interrupted"):
        swap_hf_moe_blocks(model, router_aux_loss_coef=0.01)
    assert dict(AuxLoss._group_counts) == counts_before
    assert all(
        layer.mlp is original
        for layer, original in zip(model.layers, original_blocks, strict=True)
    )


def test_later_router_conversion_error_preserves_original_block_slots():
    model = _model(_config(norm_topk_prob=True))
    original_blocks = [layer.mlp for layer in model.layers]
    counts_before = dict(AuxLoss._group_counts)
    model.layers[1].mlp.gate.bias = torch.nn.Parameter(torch.zeros(8))

    with pytest.raises(UnsupportedCombinationError, match="bias"):
        swap_hf_moe_blocks(model)

    assert dict(AuxLoss._group_counts) == counts_before
    for layer, original in zip(model.layers, original_blocks, strict=True):
        assert layer.mlp is original
        assert not getattr(layer, "moe_enabled", False)
        assert "moe" not in layer.__dict__


@pytest.mark.parametrize("later_failure", [False, True])
def test_conversion_preserves_shared_expert_dtype_and_child_modes(later_failure):
    model = _model(_config(norm_topk_prob=True))
    shared = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Dropout())
    shared.train()
    shared[0].eval()
    shared.register_buffer("calibration", torch.tensor([1.25]))
    model.layers[0].mlp.shared_expert = shared
    original_weight = shared[0].weight
    original_blocks = [layer.mlp for layer in model.layers]
    if later_failure:
        model.layers[1].mlp.gate.bias = torch.nn.Parameter(torch.zeros(8))
        with pytest.raises(UnsupportedCombinationError, match="bias"):
            swap_hf_moe_blocks(model)
        assert model.layers[0].mlp is original_blocks[0]
    else:
        assert swap_hf_moe_blocks(model) == 2
        assert model.layers[0].mlp.shared_experts is shared

    assert shared[0].weight is original_weight
    assert shared[0].weight.dtype == torch.float32
    assert shared.calibration.dtype == torch.float32
    assert shared.training
    assert not shared[0].training
    assert shared[1].training


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), float("inf"), -0.1])
@pytest.mark.parametrize("inherited", [False, True])
def test_invalid_aux_loss_coefficient_preserves_hf_blocks(value, inherited):
    from llmtuner.errors import ConfigError

    model = _model(_config(norm_topk_prob=True))
    blocks = [layer.mlp for layer in model.layers]
    if inherited:
        object.__setattr__(model.model.config, "router_aux_loss_coef", value)
    kwargs = {} if inherited else {"router_aux_loss_coef": value}
    with pytest.raises(ConfigError, match="router_aux_loss_coef"):
        swap_hf_moe_blocks(model, **kwargs)
    assert all(
        layer.mlp is block for layer, block in zip(model.layers, blocks, strict=True)
    )


def test_swap_preserves_moe_block_sharing_across_layers():
    reference = _model(_config(norm_topk_prob=True))
    swapped = _model(_config(norm_topk_prob=True))
    reference.layers[1].mlp = reference.layers[0].mlp
    swapped.layers[1].mlp = swapped.layers[0].mlp
    ids, positions = _data()
    with torch.no_grad():
        expected = reference(ids, positions=positions)
    assert swap_hf_moe_blocks(swapped) == 2
    assert swapped.layers[0].mlp is swapped.layers[1].mlp
    assert swapped.layers[0].moe is swapped.layers[1].moe
    assert swapped.layers[0].mlp.router.gate.weight is (
        swapped.layers[1].mlp.router.gate.weight
    )
    with torch.no_grad():
        actual = swapped(ids, positions=positions)
    torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("projection", ["router", "gate_up_proj", "down_proj"])
def test_swap_preserves_projection_sharing_and_accumulated_gradients(projection):
    reference = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    swapped = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    for model in (reference, swapped):
        blocks = [layer.mlp for layer in model.layers]
        if projection == "router":
            blocks[1].gate.weight = blocks[0].gate.weight
        else:
            _tie_expert_projection(blocks[1], blocks[0], projection)
    original = reference.layers[0].mlp
    assert swap_hf_moe_blocks(swapped) == 2
    first, second = [layer.mlp for layer in swapped.layers]
    if projection == "router":
        actual = first.router.gate.weight
        assert actual is second.router.gate.weight
        actual_parts = [actual]
        expected_params = [original.gate.weight]
    elif projection == "gate_up_proj":
        a = first.routed_experts.inner_experts
        b = second.routed_experts.inner_experts
        assert a.w1_EFD is b.w1_EFD and a.w3_EFD is b.w3_EFD
        actual_parts = [a.w1_EFD, a.w3_EFD]
        expected_params = _hf_projection_parameters(original, projection)
    else:
        actual = first.routed_experts.inner_experts.w2_EDF
        assert actual is second.routed_experts.inner_experts.w2_EDF
        actual_parts = [actual]
        expected_params = _hf_projection_parameters(original, projection)
    ids, positions = _data()
    reference_output = reference(ids, positions=positions)
    swapped_output = swapped(ids, positions=positions)
    torch.testing.assert_close(swapped_output, reference_output, atol=TOL, rtol=TOL)
    reference_output.square().mean().backward()
    swapped_output.square().mean().backward()
    expected_grad = (
        expected_params[0].grad
        if projection == "router"
        else _hf_projection_tensor(original, projection, grad=True)
    )
    actual_grad = (
        torch.cat([p.grad for p in actual_parts], dim=1)
        if len(actual_parts) == 2
        else actual_parts[0].grad
    )
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
    torch.optim.SGD(actual_parts, lr=0.1).step()
    torch.optim.SGD(expected_params, lr=0.1).step()
    actual_value = (
        torch.cat(actual_parts, dim=1) if len(actual_parts) == 2 else actual_parts[0]
    )
    expected_value = (
        expected_params[0]
        if projection == "router"
        else _hf_projection_tensor(original, projection)
    )
    torch.testing.assert_close(actual_value, expected_value, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("projection", ["router", "down_proj"])
def test_swap_preserves_parameter_shared_with_unconverted_module(projection):
    model = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    original = model.layers[0].mlp
    parameter = (
        original.gate.weight
        if projection == "router"
        else _hf_down_parameters(original)[0]
    )
    consumer = torch.nn.Module()
    consumer.register_parameter("weight", parameter)
    model.extra_consumer = consumer
    if projection == "down_proj" and len(_hf_down_parameters(original)) > 1:
        # A per-expert (4.x) down projection is stacked on conversion, so an
        # external alias to one expert's weight cannot be preserved -- the same
        # refusal the fused layout applies to its split gate/up alias.
        with pytest.raises(UnsupportedCombinationError, match="extra_consumer"):
            swap_hf_moe_blocks(model)
        assert model.layers[0].mlp is original
        assert consumer.weight is parameter
        return
    swap_hf_moe_blocks(model)
    converted = model.layers[0].mlp
    actual = (
        converted.router.gate.weight
        if projection == "router"
        else converted.routed_experts.inner_experts.w2_EDF
    )
    assert actual is consumer.weight
    loss = actual.square().sum() + 2 * consumer.weight.square().sum()
    loss.backward()
    torch.testing.assert_close(parameter.grad, 6 * parameter)


def test_swap_preserves_shared_sign_balancing_bias_and_joint_update():
    from llmtuner.models.common.moe.bias_update import update_expert_bias

    model = _deepseek_model()
    model.layers[2].mlp.gate.e_score_correction_bias = model.layers[
        1
    ].mlp.gate.e_score_correction_bias
    swap_hf_moe_blocks(model)
    first, second = model.layers[1].mlp, model.layers[2].mlp
    assert first.expert_bias_E is second.expert_bias_E
    before = first.expert_bias_E.clone()
    first.tokens_per_expert_E[0] = 8
    second.tokens_per_expert_E[1:] = 3
    total = first.tokens_per_expert_E + second.tokens_per_expert_E
    delta = first.load_balance_coeff * torch.sign(total.mean() - total)
    update_expert_bias([first, second])
    torch.testing.assert_close(first.expert_bias_E, before + delta - delta.mean())
    assert torch.count_nonzero(first.tokens_per_expert_E) == 0
    assert torch.count_nonzero(second.tokens_per_expert_E) == 0


def test_shared_quantile_bias_survives_conversion_and_state_restore():
    from copy import deepcopy

    from llmtuner.models.common.moe.bias_update import update_quantile_expert_bias

    def build():
        model = _deepseek_model(n_group=1, topk_group=1)
        model.layers[2].mlp.gate.e_score_correction_bias = model.layers[
            1
        ].mlp.gate.e_score_correction_bias
        swap_hf_moe_blocks(model, quantile_balancing=True)
        return model

    model = build()
    first, second = model.layers[1].mlp, model.layers[2].mlp
    assert first.expert_bias_E is second.expert_bias_E
    h1 = first.router.quantile_balancer.required_bias_histogram_EB
    h2 = second.router.quantile_balancer.required_bias_histogram_EB
    h1[:, 200] = 3
    h2[:, 700] = 5
    expected = first.router.quantile_balancer.estimate_expert_bias(
        h1 + h2, first.expert_bias_E.clone()
    )
    update_quantile_expert_bias([first, second])
    torch.testing.assert_close(first.expert_bias_E, expected)
    saved = deepcopy(model.state_dict())
    restored = build()
    restored.load_state_dict(saved)
    a, b = restored.layers[1].mlp, restored.layers[2].mlp
    assert a.expert_bias_E is b.expert_bias_E
    torch.testing.assert_close(a.expert_bias_E, expected)
    assert not a.router.quantile_balancer.required_bias_histogram_EB.any()
    assert not b.router.quantile_balancer.required_bias_histogram_EB.any()


@pytest.mark.parametrize("whole_block", [False, True])
def test_swap_rejects_external_fused_expert_alias_before_installation(whole_block):
    model = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    original_blocks = [layer.mlp for layer in model.layers]
    original = _hf_gate_up_parameters(original_blocks[0])[0]
    if whole_block:
        model.external_expert = original_blocks[0]
    else:
        model.register_parameter("external_expert", original)
    before_counts = dict(AuxLoss._group_counts)
    with pytest.raises(UnsupportedCombinationError, match="external_expert"):
        swap_hf_moe_blocks(model)
    assert all(
        layer.mlp is block
        for layer, block in zip(model.layers, original_blocks, strict=True)
    )
    assert model.external_expert is (original_blocks[0] if whole_block else original)
    assert AuxLoss._group_counts == before_counts


def test_direct_swap_rejects_fixed_quantile_conflict_before_replacing_blocks():
    model = _deepseek_model(n_group=1, topk_group=1)
    original = [layer.mlp for layer in model.layers]
    with pytest.raises(UnsupportedCombinationError, match="fixed_router.*quantile"):
        swap_hf_moe_blocks(model, fixed_router=True, quantile_balancing=True)
    assert all(
        layer.mlp is block for layer, block in zip(model.layers, original, strict=True)
    )


def test_fixed_router_ep1_public_assembly_runs_constant_routing():
    from llmtuner.config.parallel_config import ExpertParallelConfig, ParallelConfig
    from llmtuner.parallel.parallelize import parallelize_hf_transformers
    from tests.integration_tests.fixed_router_ep_equivalence import explicit_reference

    reference = _model(_config(norm_topk_prob=True))
    target = _model(_config(norm_topk_prob=True))
    original = target.layers[0].mlp
    inputs = torch.randn(5, 64, dtype=torch.float64)
    expected, _, _ = explicit_reference(
        reference.layers[0].mlp, inputs, torch.zeros(5, dtype=torch.bool)
    )
    result = parallelize_hf_transformers(
        target,
        cfg=ParallelConfig(ep_config=ExpertParallelConfig(fixed_router=True)),
        mesh=None,
        parallel_dims=None,
    )
    assert result is target
    block = target.layers[0].mlp
    assert block is not original
    assert block.router.fixed_router
    assert block.routed_experts.inner_experts.w1_EFD.shape == (8, 48, 64)
    actual = block(inputs)
    # 4.x Qwen3Moe unpacks ``(hidden, router_logits)`` from its block; the swap
    # preserves that arity, handing back None for the internal scores.
    if isinstance(actual, tuple):
        actual, router_logits = actual
        assert router_logits is None
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_shared_expert_local_checkpoint_registration_survives_conversion():
    model = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    block = model.layers[0].mlp
    shared = torch.nn.Linear(64, 64, bias=False, dtype=torch.float64)
    block.shared_expert = shared
    group = object()
    shared._tp_checkpoint_layouts = {"weight": (0, False, group)}
    swap_hf_moe_blocks(model)
    assert model.layers[0].mlp.shared_experts is shared
    assert shared._tp_checkpoint_layouts == {"weight": (0, False, group)}


def test_expert_local_checkpoint_registration_moves_to_converted_experts():
    model = _model(_config(norm_topk_prob=True, aux_coeff=0.0))
    original = model.layers[0].mlp.experts
    group = object()
    original._tp_checkpoint_layouts = {
        "gate_up_proj": (1, True, group),
        "down_proj": (2, False, group),
    }
    swap_hf_moe_blocks(model)
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    assert grouped._tp_checkpoint_layouts == {
        "w1_EFD": (1, False, group),
        "w3_EFD": (1, False, group),
        "w2_EDF": (2, False, group),
    }


def test_checkpoint_layout_failure_rolls_back_auxiliary_registration():
    model = _model(_config(norm_topk_prob=True))
    original = [layer.mlp for layer in model.layers]
    layouts = {"gate_up_proj": (1,)}
    original[0].experts._tp_checkpoint_layouts = layouts
    before_counts = dict(AuxLoss._group_counts)
    with pytest.raises(IndexError):
        swap_hf_moe_blocks(model)
    assert AuxLoss._group_counts == before_counts
    assert original[0].experts._tp_checkpoint_layouts is layouts
    for layer, block in zip(model.layers, original, strict=True):
        assert layer.mlp is block
        assert not getattr(layer, "moe_enabled", False)


@pytest.mark.parametrize("native_path", [False, True])
def test_ep_rebinds_fused_fsdp_exclusions_hooks_and_unit_paths(native_path):
    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = _model(_config(norm_topk_prob=True))
    block = model.layers[0].mlp
    path = next(
        name for name, module in model.named_modules() if module is block.experts
    )
    if native_path:
        path = path.removesuffix("experts") + "routed_experts.inner_experts"
    originals = set(_hf_gate_up_parameters(block))
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_implementation="custom",
            fsdp_apply_modules={
                path: {"ignored_params": originals, "hook_module": block.experts}
            },
        )
    )
    references = FSDPPatternReferences(model, cfg)
    swap_hf_moe_blocks(model, on_replacement=references.record_replacements)
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    rebound = references.bind(model)
    target_path = next(
        name for name, module in model.named_modules() if module is grouped
    )
    assert set(rebound.fsdp_config.fsdp_apply_modules) == {target_path}
    options = rebound.fsdp_config.fsdp_apply_modules[target_path]
    assert options["ignored_params"] == {grouped.w1_EFD, grouped.w3_EFD}
    assert options["hook_module"] is grouped
    assert cfg.fsdp_config.fsdp_apply_modules[path]["ignored_params"] == originals
    assert cfg.fsdp_config.fsdp_apply_modules[path]["hook_module"] is block.experts


def test_replacement_observer_error_keeps_original_blocks_and_aux_counts():
    model = _model(_config(norm_topk_prob=True))
    originals = [layer.mlp for layer in model.layers]
    counts = dict(AuxLoss._group_counts)
    failure = RuntimeError("reference observer failed")

    def reject(modules, parameters):
        assert all(block in modules for block in originals)
        assert all(
            weight in parameters
            for block in originals
            for weight in _hf_gate_up_parameters(block)
        )
        assert all(
            layer.mlp is original
            for layer, original in zip(model.layers, originals, strict=True)
        )
        raise failure

    with pytest.raises(RuntimeError) as caught:
        swap_hf_moe_blocks(model, on_replacement=reject)
    assert caught.value is failure
    assert AuxLoss._group_counts == counts
    assert all(
        layer.mlp is original
        for layer, original in zip(model.layers, originals, strict=True)
    )


@pytest.mark.parametrize("native_path", [False, True])
@pytest.mark.parametrize("wildcard", [False, True])
def test_ep_rebinds_ignored_module_subtree_without_object_overrides(
    native_path, wildcard
):
    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import (
        FSDPPatternReferences,
        _resolve_pattern_units,
    )

    model = _model(_config(norm_topk_prob=True))
    source = model.layers[0].mlp.experts
    path = next(name for name, module in model.named_modules() if module is source)
    if native_path:
        path = path.removesuffix("experts") + "routed_experts.inner_experts"
    if wildcard:
        path = path.replace(".layers.0.", ".layers.*.")
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(fsdp_apply_modules={"": {}}, fsdp_ignored_modules=[path])
    )
    references = FSDPPatternReferences(model, cfg)
    swap_hf_moe_blocks(model, on_replacement=references.record_replacements)
    grouped = model.layers[0].mlp.routed_experts.inner_experts
    rebound = references.bind(model)
    _, _, exclusions = _resolve_pattern_units(model, rebound)
    expected = (
        {
            parameter
            for layer in model.layers
            for parameter in layer.mlp.routed_experts.inner_experts.parameters()
        }
        if wildcard
        else set(grouped.parameters())
    )
    assert exclusions == expected
    assert cfg.fsdp_config.fsdp_ignored_modules == [path]
