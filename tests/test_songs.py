"""Regressions for whole-song examples, boundaries, and sequence limits."""

import copy
import json

import pytest

from prosodia_lyricist import ipa
from prosodia_lyricist.data import LyricDataset, read_manifest
from prosodia_lyricist.features import SOURCE_KEYS, encode_example
from prosodia_lyricist.prepare import prepare


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
    dataset = LyricDataset(
        directory, split, tokenizer, max_source_length=1024, max_target_length=1024, scaffold=True
    )
    assert len(dataset) == 1
    example = dataset[0]
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
        "<keywords>",
        ".",
        "<syllable_1>",
        "<template>",
        "<unknown>",
        "<keywords>",
        ".",
        "</s>",
    ]
    assert tokenizer.convert_ids_to_tokens(example["labels"]) == [
        "<s>",
        "hello",
        "world",
        ".",
        "light",
        ".",
        "</s>",
    ]
    assert [r for r in example["remainder_ids"] if r] == [3, 2, 1, 1]
    assert example["syllable_labels"] == [-100, 2, 1, -100, 1, -100, -100]
    assert example["remainder_labels"] == [-100, 1, 0, -100, 0, -100, -100]
    assert example["sentence_labels"] == [0, 1, 1, 0, 1, 0, 0]
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


@pytest.mark.parametrize("side", ["source", "target"])
def test_complete_song_limits_inclusive_and_never_truncated(tmp_path, annotation, tokenizer, side):
    # Ordinary words tokenize to one token with this test tokenizer. Make one
    # song exactly 1024 tokens on the selected side, and a second 1025 tokens.
    songs = []
    for i in range(3):
        song = copy.deepcopy(annotation)
        song["info"].update(id=f"limit-{i}", artist="same", title="same")
        if i:
            count = 1013 + i if side == "source" else 1019 + i
            if side == "source":
                song["info"]["title"] = " ".join(["song"] * count)
            else:
                song["annotations"]["annot"]["words"][0]["text"] = " ".join(["hello"] * count)
        songs.append(song)
    # Share an audio id so all three songs enter the same split even with long titles.
    for song in songs:
        song["info"]["audio"] = {"url": "same-audio"}
    directory, manifest = prepare_songs(tmp_path, songs)
    split = manifest["songs"]["limit-0"]["split"]
    rows = [json.loads(row) for row in (directory / f"{split}.jsonl").read_text().splitlines()]
    key = "input_ids" if side == "source" else "labels"
    assert [len(encode_example(row, tokenizer, 8)[key]) for row in rows[1:]] == [1024, 1025]
    dataset = LyricDataset(
        directory, split, tokenizer, max_source_length=1024, max_target_length=1024
    )
    assert len(dataset) == 2
    assert dataset.skipped == 1
    assert len(dataset[1][key]) == 1024


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
    assert ipa.syllable_features(SimpleNamespace(ipa="ˌaɪ"))["stress"] == "substrong"


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


def test_midi_song_generated_once_and_limits_checked(
    tmp_path,
    tokenizer,
    tiny_model,
    record,
    monkeypatch,
):
    import torch

    from prosodia_lyricist import infer as inference
    from prosodia_lyricist.model import ProsodyBart
    from prosodia_lyricist.prepare import SCHEMA_VERSION

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
        return torch.tensor([tokenizer.encode("hello world. light.")])

    monkeypatch.setattr(ProsodyBart, "generate", generate)
    lyrics = inference.infer(tmp_path, "unused.mid", title="a song", device="cpu", top_k=1)
    assert lyrics == ["hello world", "light"]
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 31
    template = tokenizer.convert_tokens_to_ids("<template>")
    assert calls[0]["input_ids"].eq(template).sum().item() == 2
    with pytest.raises(ValueError, match="target token limit"):
        inference.infer(tmp_path, "unused.mid", device="cpu", max_new_tokens=32)
    run["config"]["model"]["max_source_length"] = 16  # Each phrase fits, whole song does not.
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


@pytest.mark.parametrize("source_limit,target_limit", [(16, 1024), (1024, 6)])
def test_limits_apply_to_combined_song(tmp_path, annotation, tokenizer, source_limit, target_limit):
    directory, manifest = prepare_songs(tmp_path, [add_line(annotation)])
    split = manifest["songs"]["song-a"]["split"]
    row = json.loads((directory / f"{split}.jsonl").read_text())
    for line in row["lines"]:
        example = encode_example(line, tokenizer, 8)
        assert len(example["input_ids"]) <= source_limit
        assert len(example["labels"]) <= target_limit
    with pytest.raises(ValueError, match="No usable examples"):
        LyricDataset(
            directory,
            split,
            tokenizer,
            max_source_length=source_limit,
            max_target_length=target_limit,
        )
