"""Whole-song source contract, optional conditioning, and downstream fine-tuning."""

import copy

# ruff: noqa: F811 -- pytest fixtures imported from the shared bridge test modules
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_bridge import save_contrastive
from test_bridge_initialization import config  # noqa: F401 -- shared prepared-data fixture
from test_bridge_note import batch_for, lines, model  # noqa: F401 -- shared aligned model fixture

from prosodia_lyricist.bridge_data import BridgeDataset, encode_bridge_song
from prosodia_lyricist.bridge_infer import predict_midi_templates
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import train_bridge
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.melody_encoder.encoding import encode_note_sequence
from prosodia_lyricist.meter import encode_meter
from prosodia_lyricist.midi import midi_melody_record


def test_source_is_one_global_sequence_and_independent_of_lyrics():
    source = lines()
    encoded = encode_bridge_song(source)
    melody = {
        key: np.concatenate([line["melody"][key] for line in source])
        for key in ("midi_pitches", "onset_seconds", "note_duration_seconds")
    }
    expected = encode_note_sequence(
        melody["midi_pitches"],
        melody["onset_seconds"] - melody["onset_seconds"][0],
        melody["note_duration_seconds"],
    )
    np.testing.assert_array_equal(encoded["melody_features"], expected)
    # A fresh phrase does not reset pitch reference or IOI normalization.
    local = np.concatenate([encode_bridge_song([line])["melody_features"] for line in source])
    assert not np.array_equal(local, expected)
    altered = copy.deepcopy(source)
    for line in altered:
        line.pop("syllables")
        line.pop("note_alignment")
        line.pop("words")
        line["text"] = "unrelated lyrics"
    np.testing.assert_array_equal(encode_bridge_song(altered)["melody_features"], expected)
    # Transposition and a time-origin shift preserve the representation.
    for line in altered:
        line["melody"]["midi_pitches"] = [p + 2 for p in line["melody"]["midi_pitches"]]
        line["melody"]["onset_seconds"] = [t + 100 for t in line["melody"]["onset_seconds"]]
    np.testing.assert_array_equal(encode_bridge_song(altered)["melody_features"], expected)


def test_main_encoder_receives_complete_songs_once(model, monkeypatch):
    batch = batch_for(model)
    seen = []
    original = model.melody_encoder.encode

    def capture(features, *args, **kwargs):
        seen.append(features.shape)
        return original(features, *args, **kwargs)

    monkeypatch.setattr(model.melody_encoder, "encode", capture)
    model(**batch)
    assert seen == [(2, 8, 177)]
    assert not hasattr(model, "song_encoder") and not hasattr(model, "encode_windows")


def test_meter_is_optional_and_mixed_observation_is_masked(model, tmp_path):
    batch = batch_for(model)
    model.eval()
    baseline = model(**batch).logits
    assert not batch["meter_mask"].any()  # DALI supplies no invented meter.
    model.enable_meter()
    batch["meter_features"] = torch.randn_like(batch["meter_features"])
    batch["meter_mask"][0, :5] = True
    torch.testing.assert_close(model(**batch).logits, baseline, rtol=0, atol=0)
    model.set_melody_trainable(False)
    model(**batch).loss.backward()
    assert model.meter_adapter[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.melody_encoder.parameters())
    optimizer = torch.optim.SGD(model.meter_adapter.parameters(), lr=0.1)
    optimizer.step()
    after = model(**batch).logits
    assert not torch.equal(after[0], baseline[0])
    torch.testing.assert_close(after[1], baseline[1], rtol=1e-5, atol=1e-6)
    # Missing entries can contain junk/NaNs without entering either attention or pooling.
    batch["meter_features"][~batch["meter_mask"]] = float("nan")
    torch.testing.assert_close(model(**batch).logits, after, rtol=0, atol=0)
    model.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    assert restored.meter_conditioning
    torch.testing.assert_close(restored(**batch).logits, after, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"numerator": 0, "denominator": 4, "beat_position": 0},
        {"numerator": 3, "denominator": 3, "beat_position": 0},
        {"numerator": 3, "denominator": 4, "beat_position": 3},
        {"numerator": 3, "denominator": 4, "beat_position": float("nan")},
    ],
)
def test_invalid_observed_meter_is_rejected(entry):
    source = lines()[:1]
    source[0]["melody"]["meter"] = [entry] * 5
    with pytest.raises(ValueError, match="meter"):
        encode_meter(source)


def test_midi_meter_changes_tempo_and_unknown_prefix(tmp_path):
    import mido

    # Build without toolkit defaults: meter is unknown before the first explicit signature.
    song = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    song.tracks.append(track)
    events = [
        (0, mido.MetaMessage("set_tempo", tempo=500000)),
        (0, mido.Message("note_on", note=60, velocity=80)),
        (240, mido.Message("note_off", note=60)),
        (480, mido.MetaMessage("time_signature", numerator=3, denominator=4)),
        (480, mido.Message("note_on", note=62, velocity=80)),
        (720, mido.Message("note_off", note=62)),
        (1200, mido.MetaMessage("set_tempo", tempo=1000000)),
        (1200, mido.Message("note_on", note=64, velocity=80)),
        (1440, mido.Message("note_off", note=64)),
        (1920, mido.MetaMessage("time_signature", numerator=6, denominator=8)),
        (2160, mido.Message("note_on", note=65, velocity=80)),
        (2280, mido.Message("note_off", note=65)),
        (2280, mido.MetaMessage("marker", text="end")),
    ]
    previous = 0
    for tick, event in events:
        track.append(event.copy(time=tick - previous))
        previous = tick
    path = tmp_path / "meter.mid"
    song.save(path)
    source = midi_melody_record(path)["lines"]
    assert source[0]["melody"]["meter"] == [
        None,
        dict(numerator=3, denominator=4, beat_position=0),
        dict(numerator=3, denominator=4, beat_position=1.5),
        dict(numerator=6, denominator=8, beat_position=1),
    ]
    features, observed = encode_meter(source)
    assert observed.tolist() == [False, True, True, True]
    np.testing.assert_allclose(features[1, :4], [0, 1, 0, 1], atol=1e-6)
    np.testing.assert_allclose(features[2, :4], [0, -1, 0, -1], atol=1e-6)


def test_contrastive_fusion_freezing_padding_and_self_contained_reload(model, tmp_path):
    model.eval()
    batch = batch_for(model)
    baseline = model(**batch).logits
    # Deliberately use another width; the adapter must support it.
    tower_config = dict(
        d_model=12, num_heads=3, num_layers=1, dim_feedforward=24, dropout=0.5, max_length=32
    )
    model.add_contrastive(tower_config)
    torch.testing.assert_close(model(**batch).logits, baseline, rtol=0, atol=0)
    model.train()
    assert not model.contrastive_encoder.training
    first = model(**batch)
    torch.testing.assert_close(model(**batch).logits, first.logits, rtol=0, atol=0)
    first.loss.backward()
    assert model.contrastive_gate.grad.abs() > 0
    assert all(p.grad is None for p in model.contrastive_encoder.parameters())
    with torch.no_grad():
        model.contrastive_gate.fill_(0.5)
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    assert model.contrastive_adapter[1].weight.grad.abs().sum() > 0
    model.set_contrastive_trainable(True)
    assert model.contrastive_encoder.training
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    assert model.contrastive_encoder.input_projection[0].weight.grad.abs().sum() > 0
    model.eval()
    expected = model(**batch).logits
    assert not torch.equal(expected, baseline)
    padded = dict(batch)
    for key in ("melody_features", "meter_features"):
        padded[key] = F.pad(batch[key], (0, 0, 0, 3), value=float("nan"))
    for key in ("melody_attention_mask", "note_line_ids", "meter_mask"):
        padded[key] = F.pad(batch[key], (0, 3), value=0)
    torch.testing.assert_close(model(**padded).logits, expected, atol=1e-6, rtol=1e-5)
    model.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    torch.testing.assert_close(restored(**batch).logits, expected, rtol=1e-5, atol=1e-6)


def test_fine_tune_checkpoint_adds_meter_and_learns_it(config, tmp_path):
    base = train_bridge(config)
    before = ProsodyBridge.load(base / "best")
    directory = tmp_path / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    for split in ("train", "valid"):
        path = directory / f"{split}.jsonl"
        rows = [json.loads(row) for row in path.read_text().splitlines()]
        for song in rows:
            song["lines"][0]["melody"]["meter"] = [
                dict(numerator=3, denominator=4, beat_position=i) for i in range(3)
            ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["sha256"][path.name] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    config["model"].update(
        melody_initialization="bridge",
        melody_checkpoint=str(base / "best"),
        meter_conditioning=True,
    )
    config["training"].update(melody_unfreeze_epoch=None)
    tuned = train_bridge(config)
    after = ProsodyBridge.load(tuned / "best")
    assert after.meter_adapter[-1].weight.abs().sum() > 0
    assert torch.equal(
        before.melody_encoder.input_projection[0].weight,
        after.melody_encoder.input_projection[0].weight,
    )
    assert after.provenance["source_provenance"] == before.provenance


def test_training_fusion_checkpoint_is_optional_and_tower_can_be_unfrozen(config, tmp_path):
    upstream = tmp_path / "up.csv"
    upstream.write_text("sample_id,dali_id,split,line_count\nx,unrelated,train,2\n")
    checkpoint = tmp_path / "contrastive.pt"
    tower = save_contrastive(checkpoint)
    config["data"]["pretraining_manifest"] = str(upstream)
    config["training"]["pretraining_audit_mode"] = "warning"
    config["model"]["contrastive_fusion"] = dict(checkpoint=str(checkpoint), unfreeze_epoch=0)
    output = train_bridge(config)
    checkpoint.unlink()  # Reload must use the bridge's own saved weights.
    model = ProsodyBridge.load(output / "best")
    assert model.contrastive_gate.abs() > 0
    assert not torch.equal(
        tower.input_projection[0].weight, model.contrastive_encoder.input_projection[0].weight
    )
    assert model.provenance["contrastive"]["sha256"]
    run = json.loads((output / "run.json").read_text())
    assert run["contrastive_fusion"] and run["source_scheme"] == "global_note_177d_v1"


def test_whole_song_limits_checksums_and_legacy_metadata(config, tmp_path):
    directory = config["data"]["prepared_dir"]
    ds = BridgeDataset(directory, "train", max_song_notes=3, max_song_lines=1, limit=1)
    assert len(ds) == 1 and ds[0]["line_count"] == 1
    with pytest.raises(ValueError, match="No usable bridge songs"):
        BridgeDataset(directory, "train", max_song_notes=2)
    with pytest.raises(ValueError, match="No usable bridge songs"):
        BridgeDataset(directory, "train", max_target_length=2)
    config["data"].pop("lines_per_window")  # Old prepared note arrays remain reusable.
    config["data"]["dali_dir"] = str(tmp_path / "relocated-raw")
    train_bridge(config, smoke_test=True)


def test_midi_inference_keeps_tail_and_outputs_xai_templates(model, tmp_path, tokenizer):
    import miditoolkit

    from prosodia_lyricist.xai_features import encode_source

    song = miditoolkit.MidiFile(ticks_per_beat=480)
    track = miditoolkit.Instrument(0)
    track.notes = [miditoolkit.Note(80, 60 + i, i * 480, i * 480 + 240) for i in range(5)]
    song.instruments = [track]
    song.markers = [miditoolkit.Marker("phrase", 480), miditoolkit.Marker("phrase", 1440)]
    path = tmp_path / "tail.mid"
    song.dump(str(path))
    records, meta = predict_midi_templates(model.eval(), path)
    assert meta["note_counts"] == [2, 2, 1]
    assert meta["syllable_counts"] == [2, 2, 1]
    assert meta["source_scheme"] == "global_note_177d_v1"
    assert "encoder_windows" not in meta
    assert encode_source({"title": "song", "lines": records}, tokenizer, 8)["input_ids"]


def test_old_checkpoint_fails_with_retraining_guidance(tmp_path):
    (tmp_path / "bridge.json").write_text(json.dumps(dict(format_version=9)))
    with pytest.raises(ValueError, match="retrain older"):
        ProsodyBridge.load(tmp_path)


def test_evaluation_cli_reports_global_limits_and_aligned_coverage(config, tmp_path, capsys):
    from prosodia_lyricist.bridge_eval import main

    output = train_bridge(config, smoke_test=True)
    report_path = tmp_path / "evaluation.json"
    report = main(
        [
            "--checkpoint",
            str(output / "best"),
            "--prepared-dir",
            config["data"]["prepared_dir"],
            "--split",
            "valid",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--no-progress",
            "--output",
            str(report_path),
        ]
    )
    assert json.loads(report_path.read_text()) == report == json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 6
    assert report["data"]["examples"] == report["data"]["songs"] == 2
    assert report["prediction"]["slots"] == report["prediction"]["compared_slots"] == 4
    assert set(report["duration_baselines"]["threshold_scopes"]) == {"global", "song", "line"}
    assert "max_window_notes" not in report["model_limits"]


def test_overlong_song_is_skipped_whole_before_alignment_recovery(config, tmp_path):
    path = tmp_path / "data" / "train.jsonl"
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    tail = copy.deepcopy(rows[0]["lines"][0])
    tail["melody"]["onset_seconds"] = [3, 4, 5]
    rows[0]["lines"].append(tail)
    for line in rows[0]["lines"]:
        del line["note_alignment"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][path.name] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    ds = BridgeDataset(path.parent, "train", max_song_notes=4)
    assert len(ds) == 9 and ds.skipped == ds.skipped_limits["song_notes"] == 1
    assert rows[0]["song_id"] not in ds.songs
