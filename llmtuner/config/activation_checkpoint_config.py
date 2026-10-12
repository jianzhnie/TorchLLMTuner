"""Activation-checkpointing configs: the mode's accepted values and the
per-policy settings grafted onto ``TrainingConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmtuner.errors import ConfigError

VALID_AC_MODES: tuple[str, ...] = (
    "none",
    "full",
    "selective",
    "memory_budget",
    "region",
)
"""The accepted ``training.activation_checkpoint_mode`` values.

Declared here because this is where the value is constrained (the field is
parsed and validated in this module), and imported by
``parallel/activation_checkpoint.py`` -- which dispatches on the same set -- so
the config's accepted set and the applier's set cannot drift apart. These are
upstream's four policies plus the off switch; ``"region"`` needs the optional
``torch_remat`` package at apply time, which requires torch >= 2.10 (see
``RegionACConfig``).
"""


@dataclass(kw_only=True)
class SelectiveACConfig:
    """Settings for ``activation_checkpoint_mode='selective'``.

    Ported from torchtitan's ``SelectiveAC.Config``: ``preserve_rng_state``,
    ``determinism_check`` and ``debug`` are the same knobs with the same
    defaults. The save set itself is not configurable (it is
    ``activation_checkpoint.get_default_save_ops``); these are the per-run
    knobs around it.

    Two deviations from upstream, both forced by what llmtuner runs:

    * ``force_recompute_mm_shapes_by_fqns`` defaults to empty rather than
      ``["moe.router.gate"]``. That default assumes the torchtitan ``Module``
      protocol, whose MoE routers are ``nn.Linear`` as ``moe.router.gate``. In
      HF models the router living at ``mlp.gate`` is a container (e.g.
      ``Qwen3MoeTopKRouter``), not a ``Linear``, so the pattern matches nothing
      on an HF MoE and the setting silently does nothing; on a dense HF model
      there is no router at all. Set it to a real ``*.q_proj``/``*.o_proj``
      style name to use it -- a pattern that matches a non-``Linear`` raises
      rather than being ignored, so a wrong guess is loud.
    * The shared save set drops ``aten.topk`` (see that function's docstring:
      HF's routers mutate topk's output in place, which torch's selective
      checkpoint rejects outright). The consequence is that a selective run
      over an HF MoE recomputes its topk and so inherits topk's
      non-determinism on kernels that have any.

    ``preserve_rng_state`` is separate from the flat argument on ``apply_ac``
    -- full and selective each carry their own, as upstream's policy classes
    do, because the two policies make different demands on the recompute's RNG.
    """

    force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Fully-qualified-name substrings selecting the nn.Linear "
            "modules whose weight shapes are recomputed unconditionally. Note "
            "this selects *shapes*, not modules: any matmul with a matching "
            "(in, out) weight is recomputed, wherever it appears. Defaults to "
            "empty -- see the class docstring for why upstream's "
            "'moe.router.gate' default does not carry over."
        },
    )
    preserve_rng_state: bool = field(
        default=True,
        metadata={
            "help": "Stash and restore the RNG state around each checkpointed "
            "region so the backward-time recompute redraws the same random "
            "values, at some speed cost. Set false only if the checkpointed "
            "region is known to hold no random state."
        },
    )
    determinism_check: str = field(
        default="default",
        metadata={
            "help": "Determinism function torch's checkpoint uses to compare "
            "the recompute against the original. 'default' checks tensors that "
            "have no data-invariant structure; 'none' disables."
        },
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Capture activation-checkpointing debug information. "
            "Slower; see torch.utils.checkpoint's documentation for details."
        },
    )

    def __post_init__(self) -> None:
        if self.determinism_check not in ("default", "none"):
            raise ConfigError(
                "selective_ac.determinism_check must be 'default' or 'none', "
                f"got {self.determinism_check!r}"
            )


@dataclass(kw_only=True)
class MemoryBudgetACConfig:
    """Settings for ``activation_checkpoint_mode='memory_budget'``.

    Ported from torchtitan's ``MemoryBudgetAC.Config``. That policy carries no
    checkpointing code of its own: it sets one process-global,
    ``torch._functorch.config.activation_memory_budget``, and lets the compile
    partitioner trade compute for memory inside each compiled region -- so the
    mode only means anything when ``training.compile`` is on, and selecting it
    without compile is a config error, exactly as upstream validates.

    Upstream's ``visualize_memory_budget_pareto`` is not ported: it dumps SVGs
    into a trainer dump folder, a concept llmtuner's AC path does not have.
    """

    memory_budget: float = field(
        default=0.5,
        metadata={
            "help": "How much the compile partitioner trades compute for "
            "memory: 0.0 is the activation memory of full checkpointing over "
            "the compiled region, 1.0 the default runtime-optimized strategy. "
            "Must be in [0, 1]."
        },
    )

    def __post_init__(self) -> None:
        if not 0 <= self.memory_budget <= 1:
            raise ConfigError(
                "memory_budget_ac.memory_budget must be between 0 and 1, got "
                f"{self.memory_budget}"
            )


@dataclass(kw_only=True)
class RegionACConfig:
    """Settings for ``activation_checkpoint_mode='region'``.

    Ported from torchtitan's ``RegionAC.Config``: ``save_regions`` and
    ``recompute_regions`` are the same knobs with the same semantics -- shell
    globs relative to a transformer block, so one policy covers every block, and
    a recompute pattern wins over a save pattern -- and ``determinism_check``
    keeps upstream's default. What differs is where the region *names* come
    from. Upstream's own model code names them at ``torch_remat.region`` call
    sites (``attention.qkv``, ``feed_forward.w13``, ...) and
    ``Module.configure_remat_regions`` hands the patterns down. llmtuner runs HF
    models and does not own their decoder code, so the vocabulary is structural:
    every ``nn.Linear`` in a block is a region, named by its FQN relative to the
    block (``self_attn.q_proj``, ``mlp.gate_proj``, an MoE router's ``gate``).
    ``parallel/remat_regions.py`` explains why that is the same policy in HF's
    spelling, and what the vocabulary deliberately leaves out.

    Of the three knobs upstream's ``RegionAC.Config`` inherits from its policy
    base class, ``determinism_check`` carries over as it stands and ``debug`` is
    not exposed at all: ``torch_remat``'s checkpoint has no debug knob to forward
    it to (upstream only carries that field because the base class has it for the
    other policies). ``preserve_rng_state`` is kept, but can only stay ``False``
    -- ``torch_remat.checkpoint`` refuses ``True`` outright (a generator drawn
    inside a skipped save region would desync the recompute, and boundary-only
    stashing would hide that rather than fix it), and upstream's
    ``RegionAC.Config`` refuses it too, pointing at ``RecomputeStateHook``. The
    field is kept so that answer arrives at config-parse time, with that
    guidance, instead of from inside the library.

    Selecting the mode does not load ``torch_remat``: the package is imported
    when the policy is applied, so a config can name the mode on a machine that
    cannot run it.
    """

    save_regions: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Shell-style glob patterns, relative to a decoder block, "
            "naming the regions whose outputs are retained instead of "
            "recomputed -- e.g. 'self_attn.*' or 'mlp.down_proj'. Everything "
            "else in the block is recomputed, so an empty list (the default) "
            "retains nothing and behaves like full checkpointing. Region names "
            "are the block's nn.Linear FQNs; the set is logged at apply time "
            "and listed in parallel/remat_regions.py."
        },
    )
    recompute_regions: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Shell-style glob patterns, relative to a decoder block, "
            "naming regions that are recomputed even when save_regions also "
            "matches them -- recompute wins over save. With "
            "save_regions=['*'] this spells 'retain everything except these', "
            "suited to starting from no checkpointing and recomputing just "
            "enough to fit a memory budget. Empty (the default) leaves the "
            "save_regions-only behaviour unchanged."
        },
    )
    determinism_check: str = field(
        default="default",
        metadata={
            "help": "The check torch_remat runs to compare the recompute "
            "against the original forward. 'default' checks the tensors that "
            "have no data-invariant structure; 'none' disables it, which is "
            "torch_remat's own default."
        },
    )
    preserve_rng_state: bool = field(
        default=False,
        metadata={
            "help": "Must stay false: torch_remat does not preserve torch's "
            "RNG state, and a region whose callable draws random numbers needs "
            "an explicit RecomputeStateHook instead."
        },
    )

    def __post_init__(self) -> None:
        if self.determinism_check not in ("default", "none"):
            raise ConfigError(
                "region_ac.determinism_check must be 'default' or 'none', got "
                f"{self.determinism_check!r}"
            )
        if self.preserve_rng_state:
            raise ConfigError(
                "region_ac.preserve_rng_state must stay False: region "
                "activation checkpointing does not support it -- "
                "torch_remat.checkpoint refuses True because a generator drawn "
                "inside a skipped save region would desync the recompute, and "
                "boundary-only stashing would hide that rather than fix it. "
                "Register a RecomputeStateHook for the random state your "
                "retained regions use, or leave it false."
            )
