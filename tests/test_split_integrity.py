"""Leakage regression: content identities, transitive corpus repair, and ancestry."""

import copy
import json
from pathlib import Path

import pytest

from prosodia_lyricist import bridge_train
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION, prepare, song_groups, split_for
from prosodia_lyricist.split_integrity import (
    CONTENT_POLICY,
    audit_content_splits,
    content_fingerprints,
    main,
    read_content_index,
    repair_splits,
    require_clean_splits,
    training_exposure,
)

FULL_LYRICS = " ".join(f"differentword{i}" for i in range(30))


def song(song_id, *, interval=2, text="short lyric"):
    return {"song_id": song_id, "lines": [{
        "text": text,
        "melody": {"midi_pitches": [60, 60 + interval, 64], "onset_seconds": [0, 1, 2],
                   "note_duration_seconds": [0.5, 0.25, 0.5]},
        "syllables": [{"stress": "strong", "length": "long"},
                      {"stress": "weak", "length": "short"}],
        "note_alignment": {"scheme": "dali_word_note_starts_v1", "note_count": 3,
                           "words": [{"note_starts": [0, 1], "syllable_indices": [0, 1]}]},
    }]}


def prepared(directory, records, *, source="dali"):
    directory.mkdir()
    data = dict(prepared_dir=str(directory), source=source, include_melody=True,
                stress_source="ipa", max_syllables=8)
    manifest = dict(schema_version=SCHEMA_VERSION, config=data, songs={}, sha256={},
                    pronunciation={"rules": "test"})
    for split in ("train", "valid", "test"):
        rows = records.get(split, [])
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["sha256"][path.name] = file_sha256(path)
        for row in rows:
            manifest["songs"][row["song_id"]] = dict(split=split, group=row["song_id"])
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return read_content_index(directory)


def configuration(index, tmp_path):
    data = json.loads((Path(index["prepared_dir"]) / "manifest.json").read_text())["config"]
    return {
        "data": data,
        "model": dict(
            conditioning="bridge", melody_initialization="random",
            melody_encoder=dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16,
                                dropout=0, pooling="cls", max_length=32),
            max_song_notes=32, max_song_lines=4, max_target_length=32,
        ),
        "training": dict(output_dir=str(tmp_path / "runs"), device="cpu", seed=1,
                         epochs=1, batch_size=1, patience=1, learning_rate=0.001,
                         melody_unfreeze_epoch=0, warmup_steps=0,
                         schedule="constant_after_warmup"),
    }


def test_hashes_actual_encoder_inputs_independently_of_metadata_and_targets():
    first = song("first")
    second = song("different-metadata")
    second["lines"][0]["melody"]["midi_pitches"] = [72, 74, 76]
    second["lines"][0]["melody"]["onset_seconds"] = [10, 11, 12]
    second["lines"][0]["syllables"][0]["length"] = "short"
    a, b = (content_fingerprints(s["lines"]) for s in (first, second))
    assert a["input"] == b["input"] and a["target"] != b["target"]
    groups = song_groups({"a": {"content_fingerprints": a},
                          "b": {"content_fingerprints": b}})
    assert groups["a"] == groups["b"]
    # Equal target sequences alone must never identify unrelated melodies.
    c = content_fingerprints(song("third", interval=7)["lines"])
    assert c["target"] == a["target"]
    assert len(set(song_groups({"a": {"content_fingerprints": a},
                                "c": {"content_fingerprints": c}}).values())) == 2


def test_full_lyrics_join_different_melodies_but_short_generic_text_does_not():
    a = content_fingerprints(song("a", text=FULL_LYRICS)["lines"])
    b = content_fingerprints(song("b", interval=5, text=FULL_LYRICS.upper() + "!")["lines"])
    assert a["lyrics"] == b["lyrics"] and a["melody"] != b["melody"]
    assert "lyrics" not in content_fingerprints(song("a")["lines"])


def test_prepare_groups_musical_duplicates_with_different_metadata(tmp_path, annotation):
    raw = tmp_path / "raw"
    raw.mkdir()
    ids = [next(f"id-{i}" for i in range(100) if split_for(f"id-{i}", 1234, .2, .2) == s)
           for s in ("train", "valid")]
    for i, song_id in enumerate(ids):
        record = copy.deepcopy(annotation)
        record["info"].update(id=song_id, artist=f"Artist{i}", title=f"Title{i}",
                              audio={"url": f"different-{i}.flac"})
        for note in record["annotations"]["annot"]["notes"]:
            note["freq"] = [440.0]
        (raw / f"{i}.json").write_text(json.dumps(record))
    manifest = prepare({"data": dict(
        dali_dir=str(raw), prepared_dir=str(tmp_path / "out"), include_melody=True,
        stress_source="unknown", max_syllables=8, min_ncc=0, language="english",
        seed=1234, valid_fraction=.2, test_fraction=.2,
    )})
    assert len({e["split"] for e in manifest["songs"].values()}) == 1
    assert len({e["group"] for e in manifest["songs"].values()}) == 1
    assert require_clean_splits(read_content_index(tmp_path / "out"))["status"] == "passed"


def test_repair_transitive_cross_corpus_groups_preserves_sources(tmp_path):
    dali = prepared(tmp_path / "dali", {
        "train": [song("a")], "valid": [song("b", text=FULL_LYRICS)],
        "test": [song("unrelated", interval=9)],
    })
    lakh = prepared(tmp_path / "lakh", {
        "test": [song("c", interval=5, text=FULL_LYRICS)],
        "valid": [song("unrelated-lakh", interval=8)],
    }, source="lakh")
    combined = {"entries": dali["entries"] + lakh["entries"]}
    report = audit_content_splits(combined)
    assert len(report["conflicts"]) == 1
    assert report["affected_heldout"] == {"valid": 1, "test": 1}
    before = {str(p): p.read_bytes() for d in ("dali", "lakh")
              for p in (tmp_path / d).iterdir()}
    result = repair_splits([dali, lakh], tmp_path / "clean")
    assert result["audit"]["status"] == "passed"
    assert {e["song_id"] for e in result["reassigned_songs"]} == {"b", "c"}
    assert all(e["to"] == "train" for e in result["reassigned_songs"])
    assert before == {p: Path(p).read_bytes() for p in before}
    with pytest.raises(ValueError, match="must be new"):
        repair_splits([dali], tmp_path / "clean")


def test_audit_uses_actual_legacy_rows_and_rejects_changed_bytes(tmp_path):
    index = prepared(tmp_path / "data", {"train": [song("a")], "valid": [song("b")]})
    with pytest.raises(ValueError, match="Supervised split leakage"):
        require_clean_splits(index)
    (tmp_path / "data" / "valid.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="differs from its preparation manifest"):
        read_content_index(tmp_path / "data")


@pytest.mark.parametrize("parent_split, child_split, fails", [
    ("train", "valid", True), ("train", "test", True), ("train", "train", False),
    ("valid", "valid", False), ("valid", "test", True),
])
def test_supervised_ancestors_and_model_selection_exposure(
    tmp_path, parent_split, child_split, fails,
):
    parent = prepared(tmp_path / "dali", {
        parent_split: [song("dali-original", text=FULL_LYRICS)],
    })
    child = prepared(tmp_path / "lakh", {
        child_split: [song("lakh-arrangement", interval=5, text=FULL_LYRICS)],
    }, source="lakh")
    exposure = training_exposure(parent)
    report = audit_content_splits(child, exposure)
    assert report["status"] == ("failed" if fails else "passed")
    grandchild_exposure = training_exposure(child, exposure)
    assert exposure["entries"][0] in grandchild_exposure["entries"]


def test_training_blocks_leakage_before_loading_a_parent(tmp_path, monkeypatch):
    index = prepared(tmp_path / "data", {"train": [song("a")], "valid": [song("b")]})
    cfg = configuration(index, tmp_path)
    cfg["model"].update(melody_initialization="bridge", melody_checkpoint="not-opened")
    monkeypatch.setattr(ProsodyBridge, "load", lambda *a: pytest.fail("must audit first"))
    with pytest.raises(ValueError, match="Supervised split leakage"):
        bridge_train.train_bridge(cfg)


def test_training_records_ancestry_and_blocks_a_legacy_parent(tmp_path):
    index = prepared(tmp_path / "data", {
        "train": [song("a")], "valid": [song("b", interval=5)],
    })
    cfg = configuration(index, tmp_path)
    output = bridge_train.train_bridge(cfg)
    model = ProsodyBridge.load(output / "best")
    assert model.supervised_exposure == training_exposure(index)
    cfg["model"].update(melody_initialization="bridge", melody_checkpoint=str(output / "best"))
    fine_tuned = bridge_train.train_bridge(cfg, output_dir=tmp_path / "fine-tuned")
    assert ProsodyBridge.load(fine_tuned / "best").supervised_exposure == model.supervised_exposure
    metadata_path = output / "best" / "bridge.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["supervised_exposure"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Legacy bridge has no supervised"):
        bridge_train.train_bridge(cfg)


def test_cli_writes_failure_report_and_never_silently_accepts_contamination(tmp_path):
    index = prepared(tmp_path / "data", {"train": [song("a")], "valid": [song("b")]})
    output = tmp_path / "audit.json"
    with pytest.raises(SystemExit) as exc:
        main(["--prepared-dir", index["prepared_dir"], "--output", str(output)])
    assert exc.value.code == 1
    report = json.loads(output.read_text())
    assert report["policy"] == CONTENT_POLICY and report["status"] == "failed"
