"""Regressions for whole-song examples, boundaries, and sequence limits."""

import copy
import json

import pytest

from prosodia_lyricist import ipa
from prosodia_lyricist.data import read_manifest
from prosodia_lyricist.prepare import prepare
from prosodia_lyricist.xai_features import SOURCE_KEYS, encode_example


def add_line(annotation):
    annot = annotation["annotations"]["annot"]
    annot["lines"].append({"text": "light", "time": [4, 5], "index": 0})
    annot["words"].append({"text": "light", "time": [4, 5], "index": 1})
    annot["notes"].append({"text": "light", "time": [4, 5], "index": 2})
    return annotation


def prepare_songs(tmp_path, annotations, stress_source="unknown"):
    source = tmp_path / "raw"
    source.mkdir()
    for i, annotation in enumerate(annotations):
        (source / f"{i}.json").write_text(json.dumps(annotation))
    directory = tmp_path / "prepared"
    manifest = prepare(
        {
            "data": {
                "dali_dir": str(source),
                "prepared_dir": str(directory),
                "max_syllables": 8,
                "language": "english",
                "min_ncc": 0,
                "seed": 1234,
                "valid_fraction": 0.2,
                "test_fraction": 0.2,
                "stress_source": stress_source,
            }
        }
    )
    return directory, manifest


def test_two_lines_one_example_with_original_boundaries(tmp_path, annotation, tokenizer):
    directory, manifest = prepare_songs(tmp_path, [add_line(annotation)])
    split = manifest["songs"]["song-a"]["split"]
    rows = (directory / f"{split}.jsonl").read_text().splitlines()
    assert len(rows) == manifest["counts"][f"{split}_songs"] == 1
    assert manifest["counts"][f"{split}_lines"] == 2
    song = json.loads(rows[0])
    assert [line["line_id"] for line in song["lines"]] == [0, 1]
    assert [line["text"] for line in song["lines"]] == ["hello world", "light"]
    example = encode_example(song, tokenizer, 8)
    assert tokenizer.convert_ids_to_tokens(example["input_ids"]) == [
        "<s>",
        "<title>",
        "a",
        "song",
        "<syllable_3>",
        "<template>",
        "<unknown>",
        "<unknown>",
        "<unknown>",
        ".",
        "<syllable_1>",
        "<template>",
        "<unknown>",
        ".",
        "</s>",
    ]
    assert tokenizer.convert_ids_to_tokens(example["labels"]) == [
        "hello",
        "world",
        ".",
        "light",
        ".",
        "</s>",
    ]
    assert [r for r in example["remainder_ids"] if r] == [3, 2, 1, 1]
    assert example["labels"] == encode_example(song, tokenizer, 8)["labels"]
    assert len({len(example[key]) for key in SOURCE_KEYS}) == 1
    for i, token in enumerate(example["input_ids"]):
        if token == tokenizer.convert_tokens_to_ids("."):
            assert example["length_ids"][i] == example["remainder_ids"][i] == 0


def test_invalid_line_rejects_whole_song(tmp_path, annotation):
    invalid = add_line(copy.deepcopy(annotation))
    invalid["info"]["id"] = "invalid"
    invalid["annotations"]["annot"]["notes"].pop()
    directory, manifest = prepare_songs(tmp_path, [annotation, invalid])
    assert set(manifest["songs"]) == {"song-a"}
    assert manifest["counts"]["songs_rejected_lines"] == 1
    assert manifest["counts"]["lines_rejected"] == 1
    assert json.loads((directory / "rejected.jsonl").read_text())["line_id"] == 1


def test_old_line_manifest_requires_repreparation(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="prepare again"):
        read_manifest(tmp_path)


def test_modern_syllable_uses_ipa_not_display_text():
    from types import SimpleNamespace

    assert ipa.syllable_features(SimpleNamespace(ipa="ˈɑː", txt="ah")) == {
        "ipa": "ˈɑː",
        "stress": "strong",
        "length": "long",
    }
    assert ipa.syllable_features(SimpleNamespace(ipa="ˌaɪ"))["stress"] == "strong"


def test_modern_parser_selects_one_wordform_and_preserves_repetition():
    pytest.importorskip("prosodic")
    words = ipa.parse_words("I am a fire, fire")
    assert [w["text"] for w in words] == ["I", "am", "a", "fire", ",", "fire"]
    assert [len(w["syllables"]) for w in words] == [1, 1, 1, 1, 0, 1]
    assert words[-1] == words[-3]


def test_real_ipa_song_preparation(tmp_path, annotation):
    pytest.importorskip("prosodic")
    directory, manifest = prepare_songs(tmp_path, [add_line(annotation)], stress_source="ipa")
    split = manifest["songs"]["song-a"]["split"]
    row = json.loads((directory / f"{split}.jsonl").read_text())
    assert [len(line["syllables"]) for line in row["lines"]] == [3, 1]
    assert manifest["pronunciation"]["version"]
    assert manifest["pronunciation"]["wordform"] == "first"


def test_question_and_empty_melisma_keep_complete_ipa_song(tmp_path, annotation, tokenizer):
    pytest.importorskip("prosodic")
    annot = add_line(annotation)["annotations"]["annot"]
    annot["words"][1]["text"] = "world?"
    annot["words"][2].update(text="", time=[4, 4.5], index=0)
    annot["words"].append({"text": "light", "time": [4.5, 5], "index": 1})
    annot["notes"][4].update(text="~", time=[4, 4.5])
    annot["notes"].append({"text": "light", "time": [4.5, 5], "index": 3})
    annot["lines"][0]["time"] = [0, 4.5]
    annot["lines"][1]["time"] = [4.5, 5]
    directory, manifest = prepare_songs(tmp_path, [annotation], stress_source="ipa")
    split = manifest["songs"]["song-a"]["split"]
    row = json.loads((directory / f"{split}.jsonl").read_text())
    assert len(row["lines"]) == 2
    line = row["lines"][0]
    assert line["text"] == "hello world ?"
    assert line["words"][-1] == {"text": "?", "syllable_count": 0}
    assert len(line["syllables"]) == len(line["sung_syllables"]) == 3
    assert line["sung_syllables"][-1]["duration"] == 2.5
    assert line["sung_syllables"][-1]["note_count"] == 2
    assert not (directory / "rejected.jsonl").read_text()
    # Punctuation survives in XAI's plain lyric target stream.
    tokenizer.add_tokens(["?"])
    example = encode_example(row, tokenizer, 8)
    question = example["labels"].index(tokenizer.convert_tokens_to_ids("?"))
    assert example["labels"][question + 1] == tokenizer.convert_tokens_to_ids(".")
    from prosodia_lyricist.prepare import SCHEMA_VERSION

    assert manifest["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("text", ["?", "?!?"])
def test_punctuation_alone_has_no_ipa_syllables(text):
    pytest.importorskip("prosodic")
    with pytest.raises(ValueError, match="No IPA syllables in line"):
        ipa.parse_words(text)


def test_word_without_pronunciation_still_rejected(monkeypatch):
    from types import SimpleNamespace

    word = SimpleNamespace(txt="missing", wordtype=SimpleNamespace(form=None))
    monkeypatch.setattr(
        ipa,
        "backend",
        lambda: SimpleNamespace(Text=lambda *a, **kw: SimpleNamespace(wordtokens=[word])),
    )
    with pytest.raises(ValueError, match="No IPA pronunciation for 'missing'"):
        ipa.parse_words("missing")


def test_midi_song_generated_once_and_limits_checked(
    tmp_path,
    tokenizer,
    tiny_model,
    record,
    monkeypatch,
):
    from types import SimpleNamespace

    import torch

    from prosodia_lyricist import infer as inference
    from prosodia_lyricist.prepare import SCHEMA_VERSION
    from prosodia_lyricist.xai_model import XAILyricist

    tiny_model.save(tmp_path, tokenizer)
    run = {
        "data_schema_version": SCHEMA_VERSION,
        "config": {
            "data": {"stress_source": "unknown"},
            "model": {"max_source_length": 64, "max_target_length": 32},
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setattr(inference, "midi_records", lambda *a, **kw: [record, record])
    calls = []

    def generate(self, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            sequences=torch.tensor([tokenizer.encode("hello world. light.")]),
            explanations=[{"text": "hello world. light.", "hit_token_limit": False}],
        )

    monkeypatch.setattr(XAILyricist, "generate", generate)
    lyrics = inference.infer(tmp_path, "unused.mid", title="a song", device="cpu", top_k=1)
    assert lyrics == ["hello world", "light"]
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 31
    template = tokenizer.convert_tokens_to_ids("<template>")
    assert calls[0]["input_ids"].eq(template).sum().item() == 2
    with pytest.raises(ValueError, match="target token limit"):
        inference.infer(tmp_path, "unused.mid", device="cpu", max_new_tokens=32)
    run["config"]["model"]["max_source_length"] = 14  # Each phrase fits, whole song does not.
    (tmp_path / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="song template exceeds"):
        inference.infer(tmp_path, "unused.mid", device="cpu")
    assert len(calls) == 1
    run.pop("data_schema_version")
    (tmp_path / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="older input scheme"):
        inference.infer(tmp_path, "unused.mid", device="cpu")


def test_ipa_cache_failure_is_not_a_data_rejection(monkeypatch):
    from types import SimpleNamespace

    def failed_text(*args, **kwargs):
        raise PermissionError("cache is not writable")

    monkeypatch.setattr(ipa, "backend", lambda: SimpleNamespace(Text=failed_text))
    with pytest.raises(RuntimeError, match="resource/cache failure"):
        ipa.parse_words("unseen word")
