import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
from llmtuner.components.tokenizer.multimodal import MultimodalTokenizer


@pytest.mark.parametrize("special", [False, True])
def test_multimodal_registered_markers_preserve_atomicity_and_metadata(
    tmp_path, special
):
    from tokenizers import AddedToken
    from tokenizers.pre_tokenizers import Whitespace

    tokens = {name: f"<{name}>" for name in MultimodalTokenizer.TOKEN_FIELDS}
    vocab = {token: index for index, token in enumerate(tokens.values())}
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.add_tokens(
        [
            AddedToken(token, special=special, normalized=False, lstrip=True)
            for token in tokens.values()
        ]
    )
    metadata = {
        index: token.__getstate__()
        for index, token in backend.get_added_tokens_decoder().items()
    }
    backend.save(str(tmp_path / "tokenizer.json"))
    tokenizer = MultimodalTokenizer(
        tokenizer_path=str(tmp_path),
        **{f"{name}_token": token for name, token in tokens.items()},
    )
    assert tokenizer.encode(
        "<vision_start><image><image><vision_end>",
        add_bos=False,
        add_eos=False,
    ) == [
        vocab[token]
        for token in ("<vision_start>", "<image>", "<image>", "<vision_end>")
    ]
    assert tokenizer.tokenizer.get_vocab() == vocab
    assert {
        index: token.__getstate__()
        for index, token in tokenizer.tokenizer.get_added_tokens_decoder().items()
    } == metadata


@pytest.mark.parametrize("missing", [False, True])
def test_multimodal_tokens_resolve_from_base_vocabulary(tmp_path, missing):
    tokens = {name: f"<{name}>" for name in MultimodalTokenizer.TOKEN_FIELDS}
    vocab = {token: index for index, token in enumerate(tokens.values())}
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    from tokenizers.pre_tokenizers import Whitespace

    backend.pre_tokenizer = Whitespace()
    assert backend.get_added_tokens_decoder() == {}
    backend.save(str(tmp_path / "tokenizer.json"))
    kwargs = {f"{name}_token": token for name, token in tokens.items()}
    if missing:
        kwargs["image_token"] = "<missing>"
        with pytest.raises(ValueError, match="not found in tokenizer"):
            MultimodalTokenizer(tokenizer_path=str(tmp_path), **kwargs)
    else:
        tokenizer = MultimodalTokenizer(tokenizer_path=str(tmp_path), **kwargs)
        for name, token in tokens.items():
            assert getattr(tokenizer, f"{name}_id") == vocab[token]
        assert tokenizer.encode(
            "<vision_start><image><image><vision_end>",
            add_bos=False,
            add_eos=False,
        ) == [2, 0, 0, 3]
        assert tokenizer.get_vocab_size() == len(vocab)


@pytest.mark.parametrize("source", ["vocab.json", "vocab.txt", "tokenizer.json"])
def test_added_tokens_preserve_configured_ids_and_decoder_flags(tmp_path, source):
    vocab = {"[UNK]": 0, "hello": 1}
    if source == "tokenizer.json":
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.add_tokens(["<extra>", "<eos>"])
        backend.save(str(tmp_path / source))
    elif source == "vocab.json":
        (tmp_path / source).write_text(json.dumps(vocab), encoding="utf-8")
    else:
        (tmp_path / source).write_text("[UNK]\nhello\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<eos>",
                "add_eos_token": True,
                "added_tokens_decoder": {
                    "3": {"content": "<eos>", "special": True},
                    "2": {"content": "<extra>", "special": False},
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.token_to_id("<extra>") == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.encode("hello") == [1, 3]
    assert tokenizer.decode([2, 3], skip_special_tokens=True) == "<extra>"
    assert tokenizer.get_vocab_size() == 4


@pytest.mark.parametrize(
    "decoder",
    [
        {"3": {"content": "<gap>"}},
        {"1": {"content": "[UNK]"}},
        {"2": {"content": "<same>"}, "3": {"content": "<same>"}},
        {"2": {"content": "<one>"}, "02": {"content": "<two>"}},
        {"-1": {"content": "<negative>"}},
        {"bad": {"content": "<invalid>"}},
        {"": {"content": "<empty-id>"}},
        {"٢": {"content": "<unicode-id>"}},
        {"２": {"content": "<fullwidth-id>"}},
        {"+2": {"content": "<signed-id>"}},
        {"2.0": {"content": "<fractional-id>"}},
    ],
)
def test_incompatible_added_token_ids_fail_before_vocabulary_changes(decoder):
    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
    )
    tokenizer._hf_config = {"bos_token": "<bos>", "added_tokens_decoder": decoder}
    original = tokenizer.tokenizer.get_vocab()
    with pytest.raises(ValueError):
        tokenizer._infer_special_tokens()
    assert tokenizer.tokenizer.get_vocab() == original


@pytest.mark.parametrize("rule", ["a", "a b extra"])
def test_bpe_merge_rejects_malformed_rules_with_line_number(tmp_path, rule):
    (tmp_path / "vocab.json").write_text(
        json.dumps({"a": 0, "b": 1, "ab": 2}), encoding="utf-8"
    )
    (tmp_path / "merges.txt").write_text(f"#version: 0.2\n{rule}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*exactly two tokens"):
        HuggingFaceTokenizer(tokenizer_path=str(tmp_path))


@pytest.mark.parametrize("has_config", [False, True])
def test_unknown_token_is_inferred_from_backend_when_config_omits_it(
    tmp_path, has_config
):
    backend = Tokenizer(WordLevel({"<unknown>": 0, "known": 1}, unk_token="<unknown>"))
    backend.save(str(tmp_path / "tokenizer.json"))
    if has_config:
        (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.unk_token == "<unknown>"
    assert tokenizer.unk_id == 0
    assert tokenizer.encode("unseen") == [tokenizer.unk_id]


def test_vocab_text_preserves_empty_token_and_original_ids(tmp_path):
    (tmp_path / "vocab.txt").write_text("[UNK]\n\ntail\n", encoding="utf-8")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.token_to_id("") == 1
    assert tokenizer.token_to_id("tail") == 2
    assert tokenizer.get_vocab_size() == 3
    assert tokenizer.encode("tail") == [2]


@pytest.mark.parametrize("key", ["add_bos_token", "add_eos_token"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_special_token_defaults_reject_non_boolean_config(key, value):
    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.bos_id = tokenizer.eos_id = None
    tokenizer._hf_config = {key: value}
    with pytest.raises(ValueError, match=key):
        tokenizer._infer_should_add_bos_eos()


@pytest.mark.parametrize("decoder", [[], None, {"1": {}}, {"1": None}, {"1": 3}])
def test_invalid_added_token_config_does_not_modify_vocabulary(decoder):
    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer._hf_config = {"bos_token": "<bos>", "added_tokens_decoder": decoder}
    original = tokenizer.tokenizer.get_vocab()
    with pytest.raises(ValueError):
        tokenizer._infer_special_tokens()
    assert tokenizer.tokenizer.get_vocab() == original


@pytest.mark.parametrize(
    "template",
    [
        [{"name": "other", "template": "wrong"}, {"name": "default", "template": "ok"}],
        {"only": "ok"},
    ],
)
def test_named_chat_template_selects_default_or_unique_template(tmp_path, template):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template})
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.apply_chat_template([]) == "ok"


@pytest.mark.parametrize(
    "template, error",
    [
        ({"first": "one", "second": "two"}, "default entry"),
        ({"default": None}, "must be a string"),
        ([{"name": "default"}], "must be a string"),
    ],
)
def test_invalid_chat_template_collections_are_rejected(tmp_path, template, error):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template})
    )
    with pytest.raises(ValueError, match=error):
        HuggingFaceTokenizer(tokenizer_path=str(tmp_path))


@pytest.mark.parametrize(
    "vocab",
    [
        [],
        {},
        {"[UNK]": 0, "hello": 3},
        {"[UNK]": 0, "hello": 0},
        {"[UNK]": -1},
        {"[UNK]": True},
        {"[UNK]": "0"},
    ],
)
def test_vocab_json_rejects_invalid_id_layout(tmp_path, vocab):
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab.json"):
        HuggingFaceTokenizer(tokenizer_path=str(tmp_path))


def test_vocab_json_accepts_unsorted_contiguous_ids(tmp_path):
    (tmp_path / "vocab.json").write_text(
        json.dumps({"hello": 1, "[UNK]": 0}), encoding="utf-8"
    )
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.encode("hello", add_bos=False, add_eos=False) == [1]
    assert tokenizer.tokenizer.get_vocab_size() == 2


@pytest.mark.parametrize("policy", ["padding", "truncation", "both"])
def test_saved_batch_policies_do_not_modify_raw_training_tokens(tmp_path, policy):
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from llmtuner.datasets.text.renderer import RendererTokenizerAdapter

    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "one": 2, "two": 3, "three": 4}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    if policy in ("padding", "both"):
        backend.enable_padding(length=6, pad_id=1, pad_token="[PAD]")
    if policy in ("truncation", "both"):
        backend.enable_truncation(max_length=2)
    backend.save(str(tmp_path / "tokenizer.json"))
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tmp_path))
    assert tokenizer.encode("one two three", add_bos=False, add_eos=False) == [2, 3, 4]
    adapter = RendererTokenizerAdapter(tokenizer)
    encoded = adapter(
        "one two three", add_special_tokens=False, return_offsets_mapping=True
    )
    assert encoded["input_ids"] == [2, 3, 4]
    assert encoded["offset_mapping"] == [(0, 3), (4, 7), (8, 13)]


def test_multimodal_rejects_nonatomic_explicit_matcher(tmp_path):
    from tokenizers import AddedToken
    from tokenizers.pre_tokenizers import Whitespace

    tokens = {name: f"{name}marker" for name in MultimodalTokenizer.TOKEN_FIELDS}
    vocab = {token: index for index, token in enumerate(tokens.values())}
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.add_tokens([AddedToken(tokens["image"], single_word=True)])
    backend.save(str(tmp_path / "tokenizer.json"))
    with pytest.raises(ValueError, match="image_token.*must encode atomically"):
        MultimodalTokenizer(
            tokenizer_path=str(tmp_path),
            **{f"{name}_token": token for name, token in tokens.items()},
        )


@pytest.mark.parametrize(
    "left,right",
    [
        (left, right)
        for index, left in enumerate(MultimodalTokenizer.TOKEN_FIELDS)
        for right in MultimodalTokenizer.TOKEN_FIELDS[index + 1 :]
    ],
)
def test_multimodal_rejects_ambiguous_marker_roles(tmp_path, left, right):
    tokens = {name: f"<{name}>" for name in MultimodalTokenizer.TOKEN_FIELDS}
    vocab = {token: index for index, token in enumerate(tokens.values())}
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.save(str(tmp_path / "tokenizer.json"))
    tokens[right] = tokens[left]
    with pytest.raises(ValueError, match="distinct"):
        MultimodalTokenizer(
            tokenizer_path=str(tmp_path),
            **{f"{name}_token": token for name, token in tokens.items()},
        )


def test_multimodal_rejects_distinct_markers_with_aliased_vocabulary_ids(tmp_path):
    tokens = {name: f"<{name}>" for name in MultimodalTokenizer.TOKEN_FIELDS}
    vocab = {token: index for index, token in enumerate(tokens.values())}
    vocab["[UNK]"] = len(vocab)
    assets = json.loads(Tokenizer(WordLevel(vocab, unk_token="[UNK]")).to_str())
    assets["model"]["vocab"][tokens["video"]] = vocab[tokens["image"]]
    # Edit the asset directly: serializing a loaded backend can discard one of
    # the aliases and accidentally test missing-token validation instead.
    (tmp_path / "tokenizer.json").write_text(json.dumps(assets))
    backend = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert backend.token_to_id(tokens["image"]) == backend.token_to_id(tokens["video"])
    with pytest.raises(ValueError, match="distinct.*IDs"):
        MultimodalTokenizer(
            tokenizer_path=str(tmp_path),
            **{f"{name}_token": token for name, token in tokens.items()},
        )
