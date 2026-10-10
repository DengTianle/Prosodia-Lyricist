"""Split audits never rebuild notes and warning mode retains data integrity checks."""

import csv
import json

import pytest

from prosodia_lyricist import audit_bridge_splits, bridge_train
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    data = dict(prepared_dir=str(directory), include_melody=True, stress_source="ipa",
                max_syllables=8, lines_per_window=2, pretraining_manifest="old.csv")
    manifest = dict(schema_version=SCHEMA_VERSION, config=data, songs={}, sha256={},
                    pronunciation={"rules": "test"})
    for split in ("train", "valid", "test"):
        song = f"song-{split}"
        manifest["songs"][song] = {
            "split": split, "group": song,
            "identity": {"artist": "Artist", "title": song, "audio": {"url": ""}},
        }
        record = {"song_id": song, "lines": [{
            "syllables": [{"stress": "strong", "length": "long"}],
            "melody": {"midi_pitches": [60], "onset_seconds": [0],
                       "note_duration_seconds": [0.5]},
        }]}
        path = directory / f"{split}.jsonl"
        path.write_text(json.dumps(record) + "\n")
        manifest["sha256"][path.name] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


def upstream(tmp_path, *, song="upstream", split="train", title="Unrelated", lines=2):
    path = tmp_path / "upstream.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "dali_id", "split", "artist", "title", "line_count", "audio_path",
        ])
        writer.writeheader()
        writer.writerow(dict(sample_id="sample", dali_id=song, split=split, artist="Artist",
                             title=title, line_count=lines, audio_path=""))
    return path


def train_config(prepared, path):
    directory, manifest = prepared
    return {
        "data": {**manifest["config"], "pretraining_manifest": str(path) if path else None},
        "model": {
            "conditioning": "bridge", "melody_initialization": "pretrained",
            "melody_checkpoint": "checkpoint.pt", "max_window_notes": 16,
            "max_song_notes": 16, "max_song_lines": 4, "max_target_length": 16,
        },
        "training": dict(
            pretraining_audit_mode="warning", output_dir=str(directory.parent / "runs"),
            device="cpu", seed=1, epochs=1, batch_size=1, patience=1,
            learning_rate=0.001, melody_learning_rate=0.001, melody_unfreeze_epoch=None,
            warmup_steps=0, schedule="constant_after_warmup",
        ),
    }


@pytest.mark.parametrize("song,title", [
    ("song-valid", "Different title"), ("different-id", "Song VALID!"),
])
def test_detects_shared_ids_and_duplicate_titles(prepared, tmp_path, song, title):
    report = audit_bridge_splits.audit_prepared_splits(
        prepared[0], upstream(tmp_path, song=song, title=title),
    )
    assert report["status"] == "failed"
    assert report["conflicts"][0]["kind"] == "pretraining_train_in_downstream_heldout"
    assert report["conflicts"][0]["downstream"] == [{"song_id": "song-valid", "split": "valid"}]


def test_audio_identity_and_non_training_mismatch(prepared, tmp_path):
    directory, manifest = prepared
    manifest["songs"]["song-train"]["identity"]["audio"]["url"] = str(tmp_path / "audio.flac")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    path = upstream(tmp_path, split="test")
    path.write_text(path.read_text().replace("Unrelated,2,", "Unrelated,2,audio.flac"))
    report = audit_bridge_splits.audit_prepared_splits(directory, path)
    assert report["status"] == "failed"
    assert report["conflicts"][0]["kind"] == "split_mismatch"


def test_clean_audit_and_cli_leave_prepared_files_unchanged(prepared, tmp_path, monkeypatch):
    directory, manifest = prepared
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    path = upstream(tmp_path, song="song-valid", split="val", title="Song valid")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["audit", "--prepared-dir", str(directory),
                                    "--pretraining-manifest", str(path), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        audit_bridge_splits.main()
    assert exc.value.code == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["shared_songs"] == 1
    assert report["sha256"] == file_sha256(path)
    assert report["data_sha256"] == manifest["sha256"]
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_legacy_identities_recovered_without_preparation(prepared, tmp_path, monkeypatch):
    directory, manifest = prepared
    info = manifest["songs"]["song-test"].pop("identity")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    path = upstream(tmp_path, title="Song test")
    assert audit_bridge_splits.audit_prepared_splits(directory, path)["status"] == "incomplete"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "export.json").write_text(json.dumps({
        "info": {"id": "song-test", **info},
        "annotations": {"type": "horizontal", "annot": {"notes": [], "words": [], "lines": []}},
    }))

    def forbidden(*args, **kwargs):
        pytest.fail("The split audit must not extract notes or lyrics")

    monkeypatch.setattr("prosodia_lyricist.prepare.extract_lines", forbidden)
    monkeypatch.setattr("prosodia_lyricist.melody_data.attach_melody", forbidden)
    report = audit_bridge_splits.audit_prepared_splits(directory, path, dali_dir=raw)
    assert report["status"] == "failed"
    assert report["missing_downstream_identities"] == []


def test_unchanged_preparation_audit_supports_legacy_metadata(prepared, tmp_path):
    directory, manifest = prepared
    path = upstream(tmp_path)
    for info in manifest["songs"].values():
        info.pop("identity")
    manifest["pretraining_split_audit"] = {"sha256": file_sha256(path)}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    report = audit_bridge_splits.audit_prepared_splits(directory, path)
    assert report["status"] == "passed"
    assert report["basis"] == "unchanged_preparation_audit"
    path.write_text(path.read_text() + "\n")
    assert audit_bridge_splits.audit_prepared_splits(directory, path)["status"] == "incomplete"


@pytest.mark.parametrize("state", ["missing", "malformed", "none"])
def test_unavailable_is_never_reported_clean(prepared, tmp_path, state):
    path = tmp_path / "missing.csv"
    if state == "malformed":
        path.write_text("wrong,columns\n1,2\n")
    report = audit_bridge_splits.audit_prepared_splits(
        prepared[0], None if state == "none" else path,
    )
    assert report["status"] == "unavailable"


@pytest.mark.parametrize("state,expected", [
    ("missing", "unavailable"), ("changed", "passed"), ("conflict", "failed"),
])
def test_pretrained_warning_training_records_current_audit(
    prepared, tmp_path, monkeypatch, caplog, state, expected,
):
    path = tmp_path / "not-downloaded.csv" if state == "missing" else upstream(
        tmp_path, song="song-valid" if state == "conflict" else "upstream",
    )
    config = train_config(prepared, path)
    architecture = dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16,
                        dropout=0.0, projection_dim=None, max_length=32)
    monkeypatch.setattr(bridge_train, "load_contrastive_melody", lambda path: (
        MelodyTransformerEncoder(**architecture), architecture,
        {"pretraining_train_split": "train", "pretraining_valid_split": "val"},
    ))
    output = bridge_train.train_bridge(config, smoke_test=True)
    run = json.loads((output / "run.json").read_text())
    assert run["pretraining_split_check"]["status"] == expected
    assert run["pretraining_split_check"]["mode"] == "warning"
    assert json.loads((output / "best" / "run.json").read_text()) == run
    if expected != "passed":
        assert "continuing in warning mode" in caplog.text


def test_warning_keeps_feature_config_and_integrity_checks(prepared, tmp_path):
    config = train_config(prepared, tmp_path / "missing.csv")
    config["data"]["max_syllables"] += 1
    with pytest.raises(ValueError, match="Data configuration differs"):
        bridge_train.train_bridge(config, smoke_test=True)
    config["data"]["max_syllables"] -= 1
    (prepared[0] / "test.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="differs from its preparation manifest"):
        bridge_train.train_bridge(config, smoke_test=True)


def test_warning_keeps_window_compatibility_checks(prepared, tmp_path):
    config = train_config(prepared, upstream(tmp_path, lines=4))
    with pytest.raises(ValueError, match="lines_per_window must match"):
        bridge_train.train_bridge(config, smoke_test=True)


def test_strict_still_rejects_changed_upstream_and_invalid_mode(prepared, tmp_path):
    directory, manifest = prepared
    path = upstream(tmp_path)
    manifest["pretraining_split_audit"] = {"sha256": "previous-checksum"}
    with pytest.raises(ValueError, match="unchanged melody pretraining manifest"):
        audit_bridge_splits.check_training_splits(
            {**manifest["config"], "pretraining_manifest": str(path)}, manifest,
            mode="strict", required=True, encoder_lines=2,
        )
    config = train_config(prepared, path)
    config["training"]["pretraining_audit_mode"] = "typo"
    with pytest.raises(ValueError, match="pretraining_audit_mode"):
        bridge_train.train_bridge(config)
