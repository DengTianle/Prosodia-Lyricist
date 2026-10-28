import json
from pathlib import Path

import pytest

from prosodia_lyricist import midi_eval


def phrase(*pairs, text="lyrics"):
    return {"text": text, "syllables": [{"stress": s, "length": n} for s, n in pairs]}


S = ("strong", "short")
L = ("weak", "long")


def test_accuracy_counts_joint_correctness_and_unpaired_positions():
    predictions = [phrase(S, ("strong", "long"), L), phrase(S)]
    references = [phrase(S, L), phrase(L, S), phrase(S)]
    scores = midi_eval.score_templates(predictions, references)
    assert scores["counts"]["slots"] == 6
    assert scores["counts"]["reference_slots"] == 5
    assert scores["counts"]["predicted_slots"] == 4
    assert scores["counts"]["paired_slots"] == 3
    assert scores["accuracy"] == {"strength": 1 / 6, "length": 2 / 6, "combined": 1 / 6}
    assert scores["counts"]["count_mismatch_phrases"] == 3
    assert scores["equal_count_phrases"]["accuracy"]["length"] is None
    assert len(scores["phrases"]) == 3


def test_micro_aggregation_and_equal_count_diagnostic():
    first = midi_eval.score_templates([phrase(S)], [phrase(S)])
    second = midi_eval.score_templates([phrase(L, L, S)], [phrase(S, S)])
    report = midi_eval.aggregate_scores([first, second])
    assert report["accuracy"]["combined"] == 1 / 4  # Not the song mean (1/2).
    assert report["equal_count_phrases"]["accuracy"]["combined"] == 1
    # Extra phrases also contribute their full length to the denominator.
    extra = midi_eval.score_templates([phrase(S), phrase(L, L)], [phrase(S)])
    assert extra["accuracy"]["combined"] == 1 / 3
    assert midi_eval.score_templates([], [phrase(S)])["accuracy"]["combined"] == 0


def test_markers_removed_before_ipa_and_blank_lines_ignored(tmp_path, monkeypatch):
    path = tmp_path / "song.lyrics.txt"
    path.write_text("\ufeffWe'll(1) sing( 2 )!\n\n(3)\n(oh) night\n", encoding="utf-8")
    seen = []

    def parse(text, *, prosody_mode):
        seen.append((text, prosody_mode))
        return phrase(S)["syllables"]

    monkeypatch.setattr(midi_eval, "text_prosody", parse)
    references, removed = midi_eval.read_references(path, prosody_mode="actual")
    assert removed == 3
    assert seen == [("We'll sing!", "actual"), ("(oh) night", "actual")]
    assert len(references) == 2


def test_discovery_uses_existing_pairs_and_rejects_missing_lyrics(tmp_path):
    (tmp_path / "manifest.json").write_text('{"converted": [{"song": "absent"}]}')
    (tmp_path / "present.mid").touch()
    with pytest.raises(ValueError, match="Missing lyrics"):
        midi_eval.discover_songs(tmp_path)
    (tmp_path / "present.lyrics.txt").touch()
    assert [s.song_id for s in midi_eval.discover_songs(tmp_path)] == ["present"]
    with pytest.raises(ValueError, match="Unknown songs"):
        midi_eval.discover_songs(tmp_path, songs=["absent"])
    with pytest.raises(ValueError, match="positive"):
        midi_eval.discover_songs(tmp_path, limit=0)


def test_failures_and_common_subset_are_explicit(monkeypatch):
    monkeypatch.setattr(midi_eval, "read_references", lambda *a, **k: ([phrase(S)], 1))
    songs = [midi_eval.EvalSong(name, Path(name), Path(name)) for name in ("a", "b")]

    def restricted(song):
        if song.song_id == "b":
            raise ValueError("unsupported meter")
        return [phrase(L)]

    report = midi_eval.evaluate_songs(
        songs, {"all": lambda song: [phrase(S)], "restricted": restricted}
    )
    assert report["common_song_ids"] == ["a"]
    assert report["methods"]["all"]["evaluated_songs"] == 2
    assert report["methods"]["restricted"]["failed_songs"] == 1
    assert report["methods"]["all"]["common_subset"]["counts"]["slots"] == 1
    assert report["per_song"][1]["methods"]["restricted"]["error"] == "unsupported meter"

    def broken(*args, **kwargs):
        raise RuntimeError("missing IPA resources")

    monkeypatch.setattr(midi_eval, "read_references", broken)
    with pytest.raises(RuntimeError, match="missing IPA resources"):
        midi_eval.evaluate_songs(songs, {"all": lambda song: [phrase(S)]})


def make_midi(path):
    midi = pytest.importorskip("miditoolkit")
    source = midi.MidiFile(ticks_per_beat=480)
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60, 0, 240), midi.Note(80, 62, 480, 960)]
    source.instruments = [instrument]
    source.markers = [midi.Marker("end", 240)]  # Deliberately leave an unmarked tail.
    source.dump(str(path))


def test_cli_real_midi_algorithms_share_phrasing(tmp_path, monkeypatch, capsys):
    make_midi(tmp_path / "song.mid")
    (tmp_path / "song.lyrics.txt").write_text("one(1)\ntwo\n")
    monkeypatch.setattr(midi_eval, "text_prosody", lambda *a, **k: phrase(S)["syllables"])
    output = tmp_path / "report.json"
    midi_eval.main(["--data-dir", str(tmp_path), "--output", str(output), "--no-progress"])
    report = json.loads(output.read_text())
    assert set(report["methods"]) == set(midi_eval.MIDI_METHODS)
    for method in report["methods"].values():
        assert method["counts"]["predicted_slots"] == 2
        assert method["counts"]["count_mismatch_phrases"] == 0
    assert report["per_song"][0]["ignored_numeric_markers"] == 1
    assert "Strength" in capsys.readouterr().out


def test_bridge_loaded_once_and_infers_without_lyric_counts(tmp_path, monkeypatch):
    import torch

    from prosodia_lyricist.bridge_model import ProsodyBridge

    model = ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16,
             dropout=0, max_length=16),
        bridge_scope="song", query_mode="note", max_syllables=8,
        max_window_notes=8, max_song_notes=16, max_song_lines=4, max_target_length=16,
        d_model=8, num_heads=2, num_layers=1, dim_feedforward=16,
        song_encoder_layers=1, dropout=0,
    )
    model.save(tmp_path / "checkpoint")
    make_midi(tmp_path / "song.mid")
    (tmp_path / "song.lyrics.txt").write_text("one\ntwo\n")
    loads, generate_calls = [], []
    original_load = ProsodyBridge.load
    original_generate = ProsodyBridge.generate

    def load(checkpoint):
        loads.append(checkpoint)
        return original_load(checkpoint)

    def generate(self, **inputs):
        assert "syllable_counts" not in inputs and "labels" not in inputs
        assert not torch.is_grad_enabled()
        generate_calls.append(1)
        return original_generate(self, **inputs)

    monkeypatch.setattr(ProsodyBridge, "load", load)
    monkeypatch.setattr(ProsodyBridge, "generate", generate)
    # Deliberate count mismatch: reference slots must not become model inputs.
    monkeypatch.setattr(midi_eval, "text_prosody", lambda *a, **k: phrase(S, S)["syllables"])
    predictor = midi_eval.BridgePredictor(tmp_path / "checkpoint", device="cpu")
    songs = midi_eval.discover_songs(tmp_path)
    report = midi_eval.evaluate_songs(songs * 2, {"bridge": predictor}, prosody_mode="paper")
    assert len(loads) == 1 and len(generate_calls) == 2
    assert report["methods"]["bridge"]["counts"]["predicted_slots"] == 4
    assert report["methods"]["bridge"]["counts"]["reference_slots"] == 8
    with pytest.raises(ValueError, match="IPA rules differ"):
        midi_eval.evaluate_songs(songs, {"bridge": predictor}, prosody_mode="actual")
