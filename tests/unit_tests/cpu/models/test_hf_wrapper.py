"""`HFTransformerModel`: the seam between the training loop and HuggingFace.

The wrapper's job is plumbing -- expose the decoder's parts under stable names,
add the batch dim HF expects, feed RoPE explicit `position_ids`, and route
attention through a mask -- but three separate layers read those names (FSDP
walks `layers`, the parallel layer renames `tp_config`, the trainer scores the
logits), so a silent change here propagates everywhere.

These run on the sdpa fallback, so the mask exercised below is the one sdpa
gets. The flex path's own mask handling is covered at the end of the file by
flipping `_attn_implementation` on the built config.
"""

from __future__ import annotations

from tests.caps import require_env

require_env("flex_attention", "spmd_types")


import pytest
import torch

from llmtuner.components.loss import IGNORE_INDEX, next_token_targets
from llmtuner.config import LLMTunerConfig, TrainingConfig
from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
from llmtuner.models.hf.factory import (
    build_model_config,
    build_model_config_for,
)
from llmtuner.models.hf.materialization import materialize_meta_model
from llmtuner.models.hf.model import HFTransformerModel

_HIDDEN = 32
_VOCAB = 128


@pytest.fixture
def model() -> HFTransformerModel:
    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    return HFTransformerModel(config).eval()


def test_meta_materialization_restores_rope_and_loads_exact_weights() -> None:
    """Meta construction must not leave RoPE's non-persistent buffers empty."""
    config = build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )
    torch.manual_seed(7)
    reference = HFTransformerModel(config).eval()
    with torch.device("meta"):
        restored = HFTransformerModel(config).eval()

    materialize_meta_model(restored, torch.device("cpu"))
    restored.load_state_dict(reference.state_dict(), strict=True)

    assert torch.equal(restored.rotary_emb.inv_freq, reference.rotary_emb.inv_freq)
    ids = torch.randint(0, _VOCAB, (16,))
    with torch.no_grad():
        assert torch.equal(restored(ids), reference(ids))


def test_empty_architecture_overrides_build_offline_defaults(monkeypatch):
    from transformers import AutoConfig

    def unexpected(*args, **kwargs):
        pytest.fail("explicit offline overrides must not load a Hub config")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unexpected)
    overrides = {}
    config = build_model_config("qwen3", seq_len=64, arch_overrides=overrides)
    assert config.model_type == "qwen3"
    assert config.max_position_embeddings == 64
    assert overrides == {}


# -- part names ---------------------------------------------------------------


def test_meta_materialization_leaves_persistent_buffers_for_checkpoint() -> None:
    reference = torch.nn.Linear(3, 2)
    reference.register_buffer("calibration", torch.tensor([1.25]))
    with torch.device("meta"):
        restored = torch.nn.Linear(3, 2)
        restored.register_buffer("calibration", torch.empty(1))

    materialize_meta_model(restored, torch.device("cpu"))
    restored.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(restored.calibration, reference.calibration)
    torch.testing.assert_close(restored.weight, reference.weight)


def test_meta_materialization_preserves_real_nonpersistent_buffers() -> None:
    with torch.device("meta"):
        model = torch.nn.Linear(3, 2)
    expected = torch.tensor([1.25, 2.5])
    model.register_buffer("runtime_scale", expected.clone(), persistent=False)
    model.register_buffer("scale_alias", model.runtime_scale, persistent=False)

    materialize_meta_model(model, torch.device("cpu"))

    torch.testing.assert_close(model.runtime_scale, expected)
    torch.testing.assert_close(model.scale_alias, expected)
    assert not model.weight.is_meta


def test_unknown_meta_buffer_is_rejected_before_materialization() -> None:
    with torch.device("meta"):
        model = torch.nn.Linear(3, 2)
        model.register_buffer("runtime_scale", torch.empty(2), persistent=False)
    weight, buffer = model.weight, model.runtime_scale

    with pytest.raises(RuntimeError, match="unsupported non-checkpoint buffers"):
        materialize_meta_model(model, torch.device("cpu"))

    assert model.weight is weight and model.weight.is_meta
    assert model.runtime_scale is buffer and model.runtime_scale.is_meta


def test_meta_materialization_preserves_tied_parameters_and_gradients() -> None:
    reference = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
    reference[1].weight = reference[0].weight
    with torch.device("meta"):
        restored = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
        restored[1].weight = restored[0].weight

    materialize_meta_model(restored, torch.device("cpu"))
    restored.load_state_dict(reference.state_dict(), strict=True)

    assert restored[0].weight is restored[1].weight
    assert len(list(restored.parameters())) == len(list(reference.parameters()))
    inputs = torch.randn(2, 3)
    actual, expected = restored(inputs), reference(inputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(restored[0].weight.grad, reference[0].weight.grad)


def test_meta_materialization_preserves_persistent_buffer_aliases() -> None:
    with torch.device("meta"):
        model = torch.nn.Module()
        model.register_buffer("scale", torch.empty(2))
        model.register_buffer("scale_alias", model.scale)

    materialize_meta_model(model, torch.device("cpu"))
    assert model.scale is model.scale_alias
    model.load_state_dict({"scale": torch.ones(2), "scale_alias": torch.ones(2)})
    torch.testing.assert_close(model.scale, torch.ones(2))


def test_named_children_flattens_the_decoder(model: HFTransformerModel) -> None:
    """Five parts, in the order the parallel layer walks them.

    The names are a contract: ``apply_tp`` matches TP-plan entries against these
    paths, and FSDP walks the same children to find the transformer blocks.
    """
    names = [name for name, _ in model.named_children()]
    assert names == ["tok_embeddings", "layers", "norm", "lm_head", "rotary_emb"]


def test_state_dict_does_not_duplicate_tensors(model: HFTransformerModel) -> None:
    """The decoder is aliased for convenience; it must not be registered twice."""
    keys = list(model.state_dict())
    assert not any(key.startswith("_decoder.") for key in keys)
    # Every parameter appears exactly once.
    assert len(keys) == len(set(keys))


# -- forward ------------------------------------------------------------------


def test_forward_applies_lm_head_and_gradients_flow(model: HFTransformerModel) -> None:
    input_ids = torch.randint(0, _VOCAB, (16,))
    logits = model(input_ids, positions=torch.arange(16))

    assert logits.shape == (16, _VOCAB)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_positions_drive_rope(model: HFTransformerModel) -> None:
    """Shifting every position must change the output -- otherwise RoPE is idling."""
    input_ids = torch.randint(0, _VOCAB, (16,))

    with torch.no_grad():
        base = model(input_ids, positions=torch.arange(16))
        shifted = model(input_ids, positions=torch.arange(16) + 3)
        repeated = model(input_ids, positions=torch.arange(16))

    with pytest.raises(AssertionError):
        torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-8)
    assert torch.equal(base, repeated)


# -- the attention seam --------------------------------------------------------


def test_sdpa_fallback_gives_the_decoder_no_mask(model: HFTransformerModel) -> None:
    """Off CUDA the wrapper must NOT hand a BlockMask to the decoder.

    sdpa ignores ``is_causal`` whenever a mask is present and derives causality
    from the mask instead, so a BlockMask would land in ``attn_mask=`` and either
    crash (``BlockMask`` has no ``ndim``) or silently disable masking. The
    decoder is handed neither mask nor ``is_causal``: HF then builds what it
    needs itself, which is the path it is tested against.
    """
    assert model.model.config._attn_implementation != ATTENTION_BACKEND

    kwargs = model._apply_attention(torch.arange(16), None)

    assert kwargs == {"attention_mask": None}


def test_attention_masks_is_a_block_mask(model: HFTransformerModel) -> None:
    positions = torch.arange(8)
    mask = model.get_attention_masks(positions=positions)
    assert type(mask).__name__ == "BlockMask"


def test_flex_backend_gets_the_block_mask(model: HFTransformerModel) -> None:
    """The flex path is the one that consumes the mask, and only it gets it."""
    model.model.config._attn_implementation = ATTENTION_BACKEND

    kwargs = model._apply_attention(torch.arange(16), None)

    assert type(kwargs["attention_mask"]).__name__ == "BlockMask"
    assert kwargs["is_causal"] is False


def test_flex_backend_passes_an_explicit_mask_through(
    model: HFTransformerModel,
) -> None:
    model.model.config._attn_implementation = ATTENTION_BACKEND
    sentinel = model.get_attention_masks(positions=torch.arange(16))

    kwargs = model._apply_attention(torch.arange(16), sentinel)

    assert kwargs["attention_mask"] is sentinel


def test_non_flex_backend_rejects_a_packed_sequence(model: HFTransformerModel) -> None:
    """Packing cannot be expressed by the fallback; it must fail, not go unmasked.

    Packedness is corpus-derived (``attn_mask_type``), not inferred from the
    positions: the guard keys off the same flag ``get_attention_masks`` builds
    the document mask from.
    """
    model.model.config.attn_mask_type = "block_causal"
    packed = torch.tensor([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="packed sequence"):
        model._apply_attention(packed, None)


def test_non_flex_backend_rejects_a_length_one_document_boundary(
    model: HFTransformerModel,
) -> None:
    """A boundary after a length-1 document has no descending position edge.

    ``[0, 0, 1, 2]`` is two documents (lengths 1 and 3) with no restart for a
    ``positions[1:] < positions[:-1]`` scan to find. Keying the guard off that
    scan would let sdpa attend across the boundary silently; keying it off
    ``attn_mask_type`` catches it.
    """
    model.model.config.attn_mask_type = "block_causal"
    boundary_after_length_one = torch.tensor([0, 0, 1, 2])
    assert not bool(
        (boundary_after_length_one[1:] < boundary_after_length_one[:-1]).any()
    )

    with pytest.raises(ValueError, match="packed sequence"):
        model._apply_attention(boundary_after_length_one, None)


# -- the config path the trainer uses ------------------------------------------


def test_build_model_config_for_offline_arch() -> None:
    """A bare architecture name builds a local model from cfg's explicit sizes."""
    cfg = LLMTunerConfig(training=TrainingConfig(seed=42))

    config = build_model_config_for(cfg)

    assert config.model_type == cfg.hf_model
    assert config.vocab_size == cfg.vocab_size
    assert config.hidden_size == cfg.hidden_size
    assert config.num_hidden_layers == cfg.num_hidden_layers
    assert config.max_position_embeddings >= cfg.max_seq_len


@pytest.mark.parametrize("model_type", ["llava", "gemma3"])
@pytest.mark.parametrize("saved_length", [16, 128])
def test_offline_composite_extends_text_context_without_mutating_overrides(
    model_type, saved_length
) -> None:
    overrides = {"text_config": {"max_position_embeddings": saved_length}}
    config = build_model_config(model_type, seq_len=64, arch_overrides=overrides)

    assert config.text_config.max_position_embeddings == max(saved_length, 64)
    assert overrides == {"text_config": {"max_position_embeddings": saved_length}}


def test_wrapper_rejects_an_invalid_gqa_head_ratio() -> None:
    """HF accepts this config but fails later when it repeats KV heads.

    The query-head count must be an integer multiple of the KV-head count;
    checking at the wrapper boundary gives fused and unfused HF attention the
    same fail-fast contract as TorchTitan's shared GQA implementation.
    """
    config = build_model_config(
        "llama",
        seq_len=8,
        arch_overrides={
            "vocab_size": 16,
            "hidden_size": 12,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 2,
        },
    )

    with pytest.raises(ValueError, match=r"num_attention_heads.*divisible"):
        HFTransformerModel(config)


def test_build_model_config_for_accepts_a_local_checkpoint_path(tmp_path) -> None:
    """An absolute checkpoint path is a local config, not a malformed hub id.

    "/abs/path" contains more than one "/", so a slash-count heuristic reads it
    as neither hub id nor bare architecture and would take the offline branch,
    feeding the path to ``AutoConfig.for_model`` as an architecture name. The
    on-disk config.json is authoritative instead: the saved sizes win over the
    ones cfg carries.
    """
    from transformers import AutoConfig

    from llmtuner.config import ModelConfig

    saved = AutoConfig.for_model(
        "qwen3",
        vocab_size=99,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    saved.save_pretrained(tmp_path)

    cfg = LLMTunerConfig(
        model=ModelConfig(model_name_or_path=str(tmp_path)),
        training=TrainingConfig(seed=42),
    )
    config = build_model_config_for(cfg)

    # The sizes are the file's, not cfg's defaults (128 / 64 / ...).
    assert config.vocab_size == 99
    assert config.hidden_size == 24


@pytest.mark.parametrize(
    "identifier",
    ["/missing/model/path", "./missing/model", "org/model", "local-checkpoint"],
)
def test_unregistered_model_identifier_uses_pretrained_resolution(
    monkeypatch, identifier
):
    from transformers import AutoConfig

    from llmtuner.config import ModelConfig

    calls = []

    def load_config(name):
        calls.append(name)
        return AutoConfig.for_model("qwen3")

    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    config = build_model_config_for(
        LLMTunerConfig(model=ModelConfig(model_name_or_path=identifier))
    )
    assert calls == [identifier]
    assert config.model_type == "qwen3"


def test_the_mask_type_follows_the_corpus_rather_than_being_configured() -> None:
    """Packed corpora need the document mask; the synthetic one must not pay for it.

    ``build_model_config_for`` derives ``attn_mask_type`` from the dataset
    selector because the two cannot be set independently without one of them
    being wrong: every non-random corpus is packed by the loader builder,
    and nothing in the config names packing separately. Setting it by hand -- which
    is what the equivalence tests used to do -- is the drift this prevents.
    """
    from dataclasses import replace

    from llmtuner.config import DataloaderConfig

    training = TrainingConfig(seed=42)
    synthetic = build_model_config_for(LLMTunerConfig(training=training))
    assert synthetic.attn_mask_type == "causal"

    packed = build_model_config_for(
        LLMTunerConfig(
            training=replace(
                training,
                dataloader_config=DataloaderConfig(
                    dataset="local_jsonl",
                    tokenizer_path="/tmp/tokenizer",
                    dataset_path="/tmp/corpus",
                ),
            )
        )
    )
    assert packed.attn_mask_type == "block_causal"


def test_a_composite_config_carries_the_mask_type_down_to_the_text_stack() -> None:
    """A VL config's ``attn_mask_type`` must land where the model can read it.

    ``build_model_config_for`` derives the flag and sets it on the config it
    returns -- but for a composite (vision-language) config the model class is
    built from ``text_config``, and every reader of the flag (the packed guard,
    the mask builder, ``apply_cp``'s ulysses check) reads ``model.model.config``.
    Left on the top config the flag is invisible, and a packed corpus silently
    trains with a causal-only mask: attention straight across document
    boundaries. So the sub-config is what this pins.

    Both a composite config that resolves to a class here (``llava``) and one
    that does not are exercised: the carry-down is a property of the config
    shape, and must not depend on the class lookup having succeeded.
    """
    from transformers import AutoConfig

    from llmtuner.models.hf.factory import unwrap_text_config

    for name in ("llava", "gemma3"):
        top = AutoConfig.for_model(name)
        top.attn_mask_type = "block_causal"
        assert not hasattr(top.text_config, "attn_mask_type"), (
            f"{name}: the sub-config already carries the flag, so this test "
            "would pass without the carry-down and prove nothing"
        )

        seen = unwrap_text_config(top)

        assert seen is top.text_config
        assert getattr(seen, "attn_mask_type", "causal") == "block_causal", (
            f"{name}: the model would read 'causal' and attend across documents"
        )


def test_a_composite_without_the_flag_stays_unset() -> None:
    """The carry-down must not invent a flag the caller never derived.

    ``unwrap_text_config`` is also reached by callers that build a config by
    hand (the equivalence tests), and ``getattr(..., 'causal')`` is the
    documented default at the mask site. Synthesizing ``"causal"`` here would
    turn every unset flag into an explicit one and quietly retire that default.
    """
    from transformers import AutoConfig

    from llmtuner.models.hf.factory import unwrap_text_config

    top = AutoConfig.for_model("llava")
    assert not hasattr(top, "attn_mask_type")

    seen = unwrap_text_config(top)

    assert not hasattr(seen, "attn_mask_type")


def test_arch_overrides_reach_an_architecture_without_its_own_config_field() -> None:
    """The six explicit sizes cannot describe an MoE or an MLA attention.

    ``ModelConfig`` names hidden_size / intermediate_size / the head counts and
    nothing else, so a DeepSeek-V3 built from those alone gets the architecture's
    *published* defaults for everything else -- 256 experts per layer, not a toy.
    ``arch_overrides`` is the path those settings take, and this pins that it
    lands on the built config rather than being silently dropped.
    """
    from llmtuner.config import ModelConfig

    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="deepseek_v3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                "n_routed_experts": 4,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 64,
                "first_k_dense_replace": 2,
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
            },
        ),
        training=TrainingConfig(seed=42),
    )

    config = build_model_config_for(cfg)

    assert config.n_routed_experts == 4  # not the published 256
    assert config.num_experts_per_tok == 1
    assert config.moe_intermediate_size == 64
    assert config.first_k_dense_replace == 2
    assert config.q_lora_rank == 32
    assert config.kv_lora_rank == 16


def test_arch_overrides_win_over_the_explicit_sizes() -> None:
    """Both name the same field; the override is the more specific one.

    The six sizes come from ``ModelConfig``'s own defaults, so a caller setting
    only ``arch_overrides`` would otherwise lose the merge to a field it never
    set. Merged last, the override decides.
    """
    from llmtuner.config import ModelConfig

    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="llama", arch_overrides={"vocab_size": 99}
        ),
        training=TrainingConfig(seed=42),
    )

    assert build_model_config_for(cfg).vocab_size == 99


def test_arch_overrides_are_ignored_when_the_config_file_wins(tmp_path) -> None:
    """A local checkpoint carries its own architecture; overriding it by hand is
    the one thing that would make the built model disagree with its weights."""
    from transformers import AutoConfig

    from llmtuner.config import ModelConfig

    saved = AutoConfig.for_model(
        "qwen3",
        vocab_size=99,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    saved.save_pretrained(tmp_path)

    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path=str(tmp_path),
            arch_overrides={"vocab_size": 7},
        ),
        training=TrainingConfig(seed=42),
    )

    assert build_model_config_for(cfg).vocab_size == 99


def test_wrapper_forward_returns_logits_the_trainer_can_score() -> None:
    """The contract the training loop relies on: flat ids in, flat logits out.

    Loss is the trainer's business now, so this pins the boundary -- the wrapper
    yields one logit row per input token, and the trainer's next-token
    cross-entropy over those rows is a finite scalar.

    The labels handed to ``loss_sum`` are already next-token aligned, which is
    what ``preprocess_inputs`` produces for both loaders; ``loss_sum`` does no
    shifting of its own. A sequence whose every position is predictable
    therefore contributes one prediction per token.
    """
    from llmtuner.trainer.trainer import Trainer

    cfg = LLMTunerConfig(
        training=TrainingConfig(seed=42, max_seq_len=32, global_batch_size=2)
    )

    model = HFTransformerModel(build_model_config_for(cfg)).eval()
    ids = torch.randint(0, cfg.vocab_size, (cfg.global_batch_size * cfg.max_seq_len,))

    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (ids.shape[0], cfg.vocab_size)
    loss_sum = Trainer.loss_sum(logits, ids)
    assert loss_sum.ndim == 0
    assert float(loss_sum) > 0

    # The model's own path marks its row ends IGNORE_INDEX, and those are then
    # excluded from the denominator rather than silently counted. The count
    # comes from ``count_valid_tokens``, which sees the unsharded labels -- the
    # loss itself only skips the ignored rows.
    row_aware = next_token_targets(ids, seq_len=cfg.max_seq_len)
    counted = int((row_aware != IGNORE_INDEX).sum())
    assert counted == ids.shape[0] - cfg.global_batch_size
    row_loss = Trainer.loss_sum(logits, row_aware)
    assert row_loss.ndim == 0
    # Row-final positions predict nothing, so they contribute nothing.
    assert float(row_loss) < float(loss_sum)


# -- unsupported-model and experts-kernel guards ------------------------------


def _qwen3_config():
    return build_model_config(
        "qwen3",
        seq_len=64,
        arch_overrides={
            "vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    )


def test_a_dsa_model_is_handed_a_dense_mask() -> None:
    """DSA consumes its mask as a tensor, so it must not get a BlockMask.

    ``index_topk`` is the DSA-specific config attr (torchtitan's ``uses_dsa``);
    the modeling code then calls ``.dim()`` on the mask and adds it to the
    scores, which a BlockMask cannot answer. The dense mask itself is pinned in
    ``test_masks.py``, against the same modifiers the flex path uses.
    """
    config = _qwen3_config()
    config.index_topk = 16
    model = HFTransformerModel(config)

    mask = model.get_attention_masks(torch.arange(4))

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 3] == float("-inf")


def test_an_unknown_experts_implementation_is_rejected() -> None:
    """A typo must not silently fall back to some other kernel."""
    config = _qwen3_config()
    config.experts_implementation = "flash_mm"

    with pytest.raises(ValueError, match="experts_implementation"):
        HFTransformerModel(config)


def test_an_unsettable_experts_implementation_is_rejected() -> None:
    """Honor the request or fail -- never substitute (torchtitan's semantics).

    Dense Qwen3 has no ``@use_experts_implementation`` machinery, so asking it
    for grouped_mm cannot be honored.
    """
    config = _qwen3_config()
    config.experts_implementation = "grouped_mm"

    with pytest.raises(ValueError, match="does not support a settable experts"):
        HFTransformerModel(config)


def test_native_experts_implementation_is_the_default_and_builds() -> None:
    """The default leaves the HF model's own experts kernel untouched."""
    from llmtuner.models.hf.factory import resolve_model_class
    from llmtuner.models.hf.model_setup import configure_experts_implementation

    config = _qwen3_config()
    assert getattr(config, "experts_implementation", "native") == "native"
    configure_experts_implementation(config, model_cls=resolve_model_class(config))
    assert getattr(config, "_experts_implementation", None) is None

    model = HFTransformerModel(config).eval()
    assert model.model is not None


def test_tp_config_reads_the_inner_attribute_and_falls_back_to_the_property() -> None:
    """HF builds the plan as an instance ``_tp_config`` (config plan + each child's
    own), and exposes it through the ``tp_config`` property. The wrapper must read
    that attribute -- the property switches to ``_ep_config`` when the config asks
    for expert parallelism, and those are EP specs, not TP ones -- but a model
    that only answers through the property must not read as "no plan", because
    ``apply_tp`` would then silently shard nothing.

    Driven with a stand-in for ``self`` (the getter only touches ``self.model``),
    so the two spellings are pinned without building a model whose HF version
    happens to carry both.
    """
    from types import SimpleNamespace

    class AttributeOnly:
        _tp_config = {"layers.*.self_attn.q_proj": "rowwise"}

        @property
        def tp_config(self) -> dict[str, str]:
            return {"layers.*.self_attn.q_proj": "colwise"}

    assert HFTransformerModel.tp_config.fget(
        SimpleNamespace(model=AttributeOnly())
    ) == {"model.layers.*.self_attn.q_proj": "rowwise"}

    class PropertyOnly:
        @property
        def tp_config(self) -> dict[str, str]:
            return {"layers.*.self_attn.q_proj": "colwise"}

    assert HFTransformerModel.tp_config.fget(SimpleNamespace(model=PropertyOnly())) == {
        "model.layers.*.self_attn.q_proj": "colwise"
    }

    class Neither:
        pass

    assert HFTransformerModel.tp_config.fget(SimpleNamespace(model=Neither())) == {}


@pytest.mark.parametrize("attribute", ["_tp_plan", "tp_plan"])
def test_tp_config_reads_transformers_plan_interfaces(attribute: str) -> None:
    from types import SimpleNamespace

    inner = SimpleNamespace(**{attribute: {"layers.*.q_proj": "colwise"}})
    assert HFTransformerModel.tp_config.fget(SimpleNamespace(model=inner)) == {
        "model.layers.*.q_proj": "colwise"
    }


def test_vocab_size_is_the_inner_configs_number(model: HFTransformerModel) -> None:
    """The loss seam reads the vocabulary off the model, not off the config field.

    ``global_vocab_size`` has to match the head the model was actually built
    with: a hub id's ``AutoConfig`` wins over ``ModelConfig.vocab_size`` (and a
    local checkpoint directory never fills that field at all), so the model's
    own config is the only value that keeps a replicated head on the plain CE
    path. Also pinned: a config that names no vocabulary yields ``None`` rather
    than raising, which is what makes the trainer's fallback possible.
    """
    from types import SimpleNamespace

    assert model.vocab_size == _VOCAB
    assert (
        HFTransformerModel.vocab_size.fget(
            SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
        )
        is None
    )


def test_pipeline_chunk_without_embedding_or_head_is_not_weight_tied(model):
    model.tok_embeddings = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    assert not model.enable_weight_tying


@pytest.mark.parametrize("field", ["num_attention_heads", "num_key_value_heads"])
@pytest.mark.parametrize("value", [0, -1, True, 2.0, 1.5, "4"])
def test_hf_attention_heads_require_positive_integers(field, value):
    from transformers.configuration_utils import PretrainedConfig

    from llmtuner.models.hf.model_setup import validate_head_config

    config = PretrainedConfig(num_attention_heads=4, num_key_value_heads=2)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        validate_head_config(config)


@pytest.mark.parametrize("placeholder", [None, 42, "placeholder"])
def test_submodule_name_resolution_skips_non_module_attributes(placeholder):
    from llmtuner.models.hf.model_setup import resolve_submodule_name

    decoder = torch.nn.Module()
    decoder.embed_tokens = placeholder
    decoder.wte = torch.nn.Embedding(8, 4)
    assert (
        resolve_submodule_name(
            decoder, ("embed_tokens", "wte"), component="token embedding"
        )
        == "wte"
    )


def test_submodule_name_resolution_rejects_missing_module():
    from llmtuner.models.hf.model_setup import resolve_submodule_name

    decoder = torch.nn.Module()
    decoder.norm = None
    with pytest.raises(AttributeError, match="has no final norm module"):
        resolve_submodule_name(decoder, ("norm", "ln_f"), component="final norm")


@pytest.mark.parametrize("architecture", ["AutoConfig", "__version__"])
def test_model_class_resolution_skips_non_model_attributes(architecture):
    from transformers import LlamaConfig, LlamaForCausalLM

    from llmtuner.models.hf.factory import resolve_model_class

    config = LlamaConfig(architectures=[architecture])
    assert resolve_model_class(config) is LlamaForCausalLM
