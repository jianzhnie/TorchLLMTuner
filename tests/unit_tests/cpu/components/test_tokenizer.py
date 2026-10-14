import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer


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
