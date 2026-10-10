"""Preparation plans directional exposure protection before expensive extraction."""

import copy
import csv
import json
from pathlib import Path

import pytest

from prosodia_lyricist import ipa, prepare
from prosodia_lyricist.melody_data import read_melody_manifest


@pytest.fixture
def preflight_config(tmp_path, annotation):
    raw = tmp_path / "raw"
    raw.mkdir()
    song = copy.deepcopy(annotation)
    song["info"].update(id="down", artist="Artist", title="First",
                        audio={"url": str(tmp_path / "second.flac")})
    for note in song["annotations"]["annot"]["notes"]:
        note["freq"] = [440.0]
    (raw / "down.json").write_text(json.dumps(song))
    upstream = tmp_path / "upstream.csv"
    with upstream.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "dali_id", "split", "artist", "title", "audio_path", "line_count",
        ])
        writer.writeheader()
        writer.writerows([
            dict(sample_id="a", dali_id="up-a", split="train", artist="Artist", title="First",
                 audio_path="first.flac", line_count=2),
            dict(sample_id="b", dali_id="up-b", split="val", artist="Artist", title="Second",
                 audio_path="second.flac", line_count=2),
        ])
    return {"data": dict(
        dali_dir=str(raw), prepared_dir=str(tmp_path / "prepared"), max_syllables=8,
        language="english", min_ncc=0, seed=1234, valid_fraction=0.2, test_fraction=0.2,
        stress_source="ipa", include_melody=True, pretraining_manifest=str(upstream),
        lines_per_window=2,
    )}


def forbidden(*args, **kwargs):
    pytest.fail("Split checks must run before pronunciation or note extraction")


def test_transitive_upstream_conflict_is_reported_before_extraction(preflight_config, monkeypatch):
    monkeypatch.setattr(ipa, "backend", forbidden)
    monkeypatch.setattr(prepare, "extract_lines", forbidden)
    monkeypatch.setattr("prosodia_lyricist.melody_data.attach_melody", forbidden)
    report = prepare.prepare(preflight_config, check_splits=True)
    group = report["upstream_split_overlaps"][0]
    assert group["upstream"] == [
        {"song_id": "up-a", "split": "train"}, {"song_id": "up-b", "split": "valid"},
    ]
    assert group["identities"]["down:down"]["title"] == "First"
    assert group["identities"]["down:down"]["audio"].endswith("second.flac")
    assert report["songs"]["down"]["split"] == "train"
    assert report["conflicts"] == []
    directory = Path(preflight_config["data"]["prepared_dir"])
    assert [path.name for path in directory.iterdir()] == ["split_preflight.json"]


@pytest.mark.parametrize("problem", ["missing", "malformed", "windows"])
def test_invalid_upstream_fails_before_backend_and_extraction(
    preflight_config, monkeypatch, problem,
):
    path = Path(preflight_config["data"]["pretraining_manifest"])
    if problem == "missing":
        preflight_config["data"]["pretraining_manifest"] = str(path.parent / "missing.csv")
    elif problem == "malformed":
        path.write_text("invalid,header\n1,2\n")
    else:
        preflight_config["data"]["lines_per_window"] = 4
    monkeypatch.setattr(ipa, "backend", forbidden)
    monkeypatch.setattr(prepare, "read_annotation", forbidden)
    monkeypatch.setattr(prepare, "extract_lines", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare.prepare(preflight_config)


def test_full_preparation_uses_early_plan(preflight_config, monkeypatch):
    preflight_config["data"]["stress_source"] = "unknown"
    original = prepare.extract_lines
    directory = Path(preflight_config["data"]["prepared_dir"])

    def checked_extract(*args, **kwargs):
        assert (directory / "split_preflight.json").exists()
        return original(*args, **kwargs)

    monkeypatch.setattr(prepare, "extract_lines", checked_extract)
    manifest = prepare.prepare(preflight_config)
    report = json.loads((directory / "split_preflight.json").read_text())
    assert manifest["songs"]["down"]["split"] == report["songs"]["down"]["split"] == "train"
    assert manifest["pretraining_split_audit"]["conflicts"] == []
    assert manifest["pretraining_split_audit"]["upstream_split_overlaps"]


def test_rejecting_group_representative_does_not_reshuffle_survivors(preflight_config):
    data = preflight_config["data"]
    data["stress_source"] = "unknown"
    upstream = Path(data["pretraining_manifest"])
    upstream.write_text(upstream.read_text().replace(",train,", ",val,"))
    raw = Path(data["dali_dir"])
    rejected = json.loads((raw / "down.json").read_text())
    rejected["info"]["id"] = "a-rejected"
    rejected["annotations"]["annot"]["notes"][0]["text"] = "~"
    (raw / "rejected.json").write_text(json.dumps(rejected))
    manifest = prepare.prepare(preflight_config)
    report = json.loads((Path(data["prepared_dir"]) / "split_preflight.json").read_text())
    assert set(manifest["songs"]) == {"down"}
    assert manifest["songs"]["down"]["group"] == "a-rejected"
    assert manifest["songs"]["down"]["split"] == report["songs"]["down"]["split"]
    assert report["songs"]["a-rejected"]["split"] == report["songs"]["down"]["split"]


def test_lyric_manifest_reader_still_rejects_internal_split_overlap(preflight_config):
    path = Path(preflight_config["data"]["pretraining_manifest"])
    path.write_text(path.read_text().replace("Second", "First"))
    with pytest.raises(ValueError, match="crosses splits"):
        read_melody_manifest(path, require_lyrics=False)


def test_check_splits_cli_preserves_existing_dataset(preflight_config, monkeypatch):
    config_path = Path(preflight_config["data"]["dali_dir"]).parent / "config.yaml"
    preflight_config["model"] = {}
    config_path.write_text(json.dumps(preflight_config))  # JSON is valid YAML.
    directory = Path(preflight_config["data"]["prepared_dir"])
    directory.mkdir()
    (directory / "train.jsonl").write_text("existing data")
    monkeypatch.setattr(ipa, "backend", forbidden)
    monkeypatch.setattr("sys.argv", ["prepare", "--config", str(config_path), "--check-splits"])
    prepare.main()
    assert (directory / "train.jsonl").read_text() == "existing data"
    assert (directory / "split_preflight.json").exists()
