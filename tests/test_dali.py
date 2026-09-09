import gzip
import json
import pickle
import sys
import types

import numpy as np
import pytest

from prosodia_lyricist.dali import (
    AnnotationError,
    extract_lines,
    lexical_stresses,
    read_annotation,
    sung_syllables,
)


def test_parent_alignment_and_melisma(annotation):
    records, rejected = extract_lines(annotation["info"], annotation["annotations"]["annot"])
    assert not rejected
    assert records[0]["text"] == "hello world"
    syllables = records[0]["syllables"]
    assert len(syllables) == 3
    assert syllables[1]["note_count"] == 2
    assert syllables[1]["duration"] == 1.0  # gap is not sung duration
    assert [s["length"] for s in syllables] == ["short", "short", "long"]
    assert [s["stress"] for s in syllables] == ["weak", "strong", "strong"]


def test_matching_pronunciation_variant_and_unknown():
    pronunciations = {"test": [["T", "EH1", "S", "T"], ["T", "EH2", "S", "T", "AH0"]]}
    assert lexical_stresses("test!", 2, pronunciations) == ["substrong", "weak"]
    assert lexical_stresses("test", 3, pronunciations) == ["unknown"] * 3
    assert lexical_stresses("invented", 1, pronunciations) == ["unknown"]


def test_duration_only_mode(annotation):
    records, _ = extract_lines(
        annotation["info"], annotation["annotations"]["annot"], stress_source="unknown"
    )
    assert all(s["stress"] == "unknown" for s in records[0]["syllables"])


@pytest.mark.parametrize(
    "notes",
    [
        [{"text": "~", "time": [0, 1]}],
        [{"text": "a", "time": [1, 1]}],
        [{"text": "a", "time": [float("nan"), 2]}],
        [{"text": "a", "time": [0, 2]}, {"text": "b", "time": [1, 3]}],
    ],
)
def test_invalid_notes_are_rejected(notes):
    with pytest.raises(AnnotationError):
        sung_syllables(notes)


def test_continuation_fragment():
    result = sung_syllables([{"text": "lo", "time": [0, 1]}, {"text": "~ve", "time": [1, 2]}])
    assert result[0]["text"] == "love"
    assert len(result) == 1


def test_invalid_line_does_not_drop_other_lines(annotation):
    annot = annotation["annotations"]["annot"]
    annot["lines"].append({"text": "a", "time": [4, 5], "index": 0})
    annot["words"].append({"text": "a", "time": [4, 5], "index": 1})
    records, rejected = extract_lines(annotation["info"], annot)
    assert len(records) == len(rejected) == 1
    assert rejected[0]["line_id"] == 1


def test_syllable_limit_and_invalid_parent(annotation):
    annot = annotation["annotations"]["annot"]
    records, rejected = extract_lines(annotation["info"], annot, max_syllables=2)
    assert not records and rejected
    annot["notes"][0]["index"] = 100
    with pytest.raises(AnnotationError, match="out of range"):
        extract_lines(annotation["info"], annot)


def test_read_json(tmp_path, annotation):
    path = tmp_path / "song.json"
    path.write_text(json.dumps(annotation))
    info, annot = read_annotation(path)
    assert info["id"] == "song-a"
    assert len(annot["notes"]) == 4


def test_read_official_pickle_layout(tmp_path, annotation, monkeypatch):
    # Match the qualified class name and protocol used by the DALI release.
    package = types.ModuleType("DALI")
    module = types.ModuleType("DALI.Annotations")
    cls = type("Annotations", (), {"__module__": "DALI.Annotations"})
    module.Annotations = cls
    package.Annotations = module
    monkeypatch.setitem(sys.modules, "DALI", package)
    monkeypatch.setitem(sys.modules, "DALI.Annotations", module)
    entry = cls()
    entry.info = annotation["info"]
    entry.info["scores"]["NCC"] = np.float64(0.9)
    entry.annotations = annotation["annotations"]
    path = tmp_path / "song.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(entry, handle, protocol=2)
    info, annot = read_annotation(path)
    assert info["scores"]["NCC"] == 0.9
    assert annot == annotation["annotations"]["annot"]


def test_restrict_pickle_globals(tmp_path):
    path = tmp_path / "invalid.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(ValueError("not a DALI annotation"), handle)
    with pytest.raises(pickle.UnpicklingError, match="Unsupported DALI pickle type"):
        read_annotation(path)
