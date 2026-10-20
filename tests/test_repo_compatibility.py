"""Regression contracts from the released repository and the baseline fixes."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prosodia_lyricist import evaluation, ipa
from prosodia_lyricist.midi import midi_melody_record, midi_records
from prosodia_lyricist.xai_features import encode_source


def mock_tokens(monkeypatch, tokens, modern):
    monkeypatch.setattr(
        ipa,
        "backend",
        lambda: SimpleNamespace(
            Text=lambda *args, **kwargs: SimpleNamespace(
                wordtokens=[
                    SimpleNamespace(
                        txt=t, wordtype=SimpleNamespace(form=SimpleNamespace(syllables=s))
                    )
                    for t, s in zip(tokens, modern, strict=True)
                ]
            )
        ),
    )


@pytest.fixture
def historical_data():
    if not ipa._resources().joinpath("english.tsv").is_file():
        pytest.skip("Install historical IPA data with python -m prosodia_lyricist.historical_ipa")
    return ipa.historical_provenance()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("'laɪk", ("strong", "short")),
        ("ˌaɪ", ("substrong", "short")),
        ("`ə", ("substrong", "short")),
        ("iː", ("weak", "long")),
        ("i:", ("weak", "short")),
    ],
)
def test_repo_ipa_uses_secondary_stress_and_only_the_long_mark(value, expected):
    result = evaluation.syllable_prosody({"ipa": value}, prosody_mode="actual")
    assert (result["stress"], result["length"]) == expected


def test_historical_pronunciation_order_and_unstress(monkeypatch, historical_data):
    tokens = ["I", "we", "be", "would", "you", "I'm", "I'd", "for", "were", "just", "I"]
    mock_tokens(monkeypatch, tokens, [["ə"]] * len(tokens))
    parsed = ipa.parse_words(" ".join(tokens), prosody_mode="actual")
    assert [w["syllables"][0]["stress"] for w in parsed] == [
        "strong",
        "strong",
        "strong",
        "strong",
        "weak",
        "weak",
        "weak",
        "weak",
        "weak",
        "strong",
        "strong",
    ]
    assert all(w["pronunciation_source"] == ipa.HISTORICAL_SOURCE for w in parsed)
    assert parsed[4]["syllables"][0] == {"ipa": "juː", "stress": "weak", "length": "long"}
    assert historical_data["version"] == "1.5.0"


def test_historical_count_and_modern_fallback_are_explicit(monkeypatch, historical_data):
    mock_tokens(monkeypatch, ["heaven", ",", "qxunlisted"], [["hɛvn"], [], ["`aɪ", "ə"]])
    words = ipa.parse_words("heaven, qxunlisted", prosody_mode="actual")
    assert len(words[0]["syllables"]) == 2
    assert words[1] == {"text": ",", "syllables": []}
    assert words[2]["pronunciation_source"] == ipa.FALLBACK_SOURCE
    assert words[2]["syllables"][0]["stress"] == "substrong"
    assert words[2]["syllables"][0]["length"] == "short"
    metrics = evaluation.evaluate_prosody(
        [{"syllables": words[0]["syllables"]}], ["heaven, qxunlisted"], prosody_mode="actual"
    )
    assert metrics["phrases"][0]["pronunciation_fallback_words"] == ["qxunlisted"]


def test_xai_source_never_receives_keyword_marker_or_text(record, tokenizer):
    record = {"title": "a song", "lines": [{**record, "keyword": "light"}]}
    first = encode_source(record, tokenizer, 8)
    record["lines"][0]["keyword"] = "hello world"
    assert encode_source(record, tokenizer, 8) == first
    assert tokenizer.convert_tokens_to_ids("<keywords>") not in first["input_ids"]
    assert [n for n in first["remainder_ids"] if n] == [3, 2, 1]


@pytest.mark.parametrize("resolution", [480, 960, 241])
def test_midi_matches_executed_reference_helper_and_marker_grouping(tmp_path, resolution):
    midi = pytest.importorskip("miditoolkit")
    reference = Path(__file__).resolve().parents[2] / "XAI-Lyricist/utils/prosody_utils.py"
    if not reference.exists():
        pytest.skip("Adjacent reference checkout is unavailable")
    tree = ast.parse(reference.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("stress", "getProsody")
    ]
    namespace = {"miditoolkit": midi, "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(reference), "exec"), namespace)
    song = midi.MidiFile(ticks_per_beat=resolution)
    track = midi.Instrument(0)
    onsets = [
        0,
        3 * resolution // 2,
        5 * resolution,
        8 * resolution,
        9 * resolution,
        11 * resolution,
    ]
    durations = [
        resolution // 2,
        3 * resolution // 4,
        resolution,
        resolution,
        2 * resolution,
        resolution // 2,
    ]
    track.notes = [
        midi.Note(80, 60 + i, start, start + duration)
        for i, (start, duration) in enumerate(zip(onsets, durations, strict=True))
    ]
    song.instruments = [track]
    song.markers = [midi.Marker("phrase", onsets[2]), midi.Marker("phrase", onsets[-2])]
    path = tmp_path / "reference.mid"
    song.dump(str(path))
    expected = namespace["getProsody"](str(path))
    actual = midi_records(path, stress_source="actual")
    syllables = [s for line in actual for s in line["syllables"]]
    assert [(s["stress"], s["length"]) for s in syllables] == [
        (stress.strip("<>"), length.lower()) for stress, length in expected[:5]
    ]
    assert len(actual) == 2 and len(syllables) == 5
    assert actual[-1]["unmarked_notes_omitted"] == 1
    assert actual[0]["stress_grid_ticks_per_beat"] == resolution
    notes = midi_melody_record(path, phrase_policy="repo")
    assert [len(line["melody"]["midi_pitches"]) for line in notes["lines"]] == [3, 2]
    assert notes["unmarked_notes_omitted"] == 1


def test_actual_report_displays_mixed_bucket_thresholds(tmp_path, historical_data):
    midi = pytest.importorskip("miditoolkit")
    from prosodia_lyricist.infer import template_report
    from prosodia_lyricist.report import markdown_report

    song = midi.MidiFile(ticks_per_beat=480)
    track = midi.Instrument(0)
    track.notes = [
        midi.Note(80, 60, 0, 240),
        midi.Note(80, 62, 3840, 4320),
        midi.Note(80, 64, 4320, 5280),
    ]
    song.instruments = [track]
    song.markers = [midi.Marker("phrase", 5280)]
    path = tmp_path / "mixed.mid"
    song.dump(str(path))
    records = midi_records(path, stress_source="actual")
    assert records[0]["length_threshold_ticks"] is None
    report = template_report(
        path, records, title="", track=0, stress_source="actual", prosody_mode="actual"
    )
    text = markdown_report(report)
    assert "Length threshold (ticks)" in text
    assert "240.000" in text and "720.000" in text
    assert "supplement's ambiguous" not in text


def test_repo_dali_preparation_records_historical_rules(tmp_path, annotation, historical_data):
    pytest.importorskip("prosodic")
    from prosodia_lyricist.prepare import SCHEMA_VERSION, prepare

    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(8):
        row = copy.deepcopy(annotation)
        row["info"].update(id=f"repo-{index}", title=f"Song {index}")
        row["annotations"]["annot"]["words"][0]["text"] = "heaven"
        row["annotations"]["annot"]["words"][1]["text"] = "you"
        (raw / f"{index}.json").write_text(json.dumps(row))
    config = dict(
        data=dict(
            dali_dir=str(raw),
            prepared_dir=str(tmp_path / "prepared"),
            max_syllables=8,
            language="english",
            min_ncc=0,
            seed=1234,
            valid_fraction=0.2,
            test_fraction=0.2,
            stress_source="ipa",
            prosody_mode="actual",
        )
    )
    manifest = prepare(config)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["pronunciation"]["rules"] == ipa.REPO_RULES
    assert manifest["pronunciation"]["pronunciation_policy"] == ipa.HISTORICAL_POLICY
    assert (
        manifest["pronunciation"]["historical_dictionary"]["files_sha256"]
        == historical_data["files_sha256"]
    )
    rows = [
        json.loads(line)
        for path in (tmp_path / "prepared").glob("*.jsonl")
        if path.name in ("train.jsonl", "valid.jsonl", "test.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 8
    for row in rows:
        line = row["lines"][0]
        assert line["words"][0]["syllable_count"] == 2
        assert all(w["pronunciation_source"] == ipa.HISTORICAL_SOURCE for w in line["words"])
        assert line["syllables"][-1]["stress"] == "weak"
        assert line["syllables"][-1]["length"] == "long"
