import pytest

from prosodia_lyricist.midi import midi_note_comparison, midi_records, supplement_stress

miditoolkit = pytest.importorskip("miditoolkit")


def make_midi(path, *, markers=True, overlap=False):
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(0)
    instrument.notes = [
        miditoolkit.Note(80, 60, 0, 240),
        miditoolkit.Note(80, 62, 100 if overlap else 480, 960),
        miditoolkit.Note(80, 64, 1200, 1440),
    ]
    midi.instruments.append(instrument)
    if markers:
        midi.markers = [miditoolkit.Marker("Phrase_0", 960)]
    midi.dump(str(path))


def test_midi_phrases_and_tail(tmp_path):
    path = tmp_path / "melody.mid"
    make_midi(path)
    records = midi_records(path, title="a song")
    assert [len(r["syllables"]) for r in records] == [2, 1]
    assert [s["length"] for s in records[0]["syllables"]] == ["short", "long"]
    unknown = midi_records(path, stress_source="unknown")
    assert all(s["stress"] == "unknown" for r in unknown for s in r["syllables"])


def test_missing_markers_and_polyphonic_track(tmp_path):
    path = tmp_path / "melody.mid"
    make_midi(path, markers=False)
    with pytest.raises(ValueError, match="phrase-end markers"):
        midi_records(path)
    make_midi(path, overlap=True)
    with pytest.raises(ValueError, match="monophonic"):
        midi_records(path)


@pytest.mark.parametrize("duration", [480, 240, 120])
def test_supplement_figure_one(duration):
    assert [supplement_stress(i * duration, duration, 480) for i in range(4)] == [
        "strong",
        "weak",
        "strong",
        "weak",
    ]


def test_supplement_rounding_fallback_and_melody_mean(tmp_path):
    assert supplement_stress(119, 240, 480) == "strong"
    assert supplement_stress(120, 240, 480) == "weak"  # Exact tie rounds upward.
    assert supplement_stress(720, 360, 480) == "strong"  # Dotted: quarter-note grid.
    path = tmp_path / "means.mid"
    make_midi(path)
    midi = miditoolkit.MidiFile(str(path))
    midi.instruments[0].notes[-1].end = 2400
    midi.dump(str(path))
    supplement = midi_records(path)
    historical = midi_records(path, stress_source="heuristic")
    assert supplement[0]["length_threshold_ticks"] == 640
    assert supplement[0]["syllables"][1]["length"] == "short"
    assert historical[0]["syllables"][1]["length"] == "long"


def test_supplement_rejects_non_four_four(tmp_path):
    path = tmp_path / "waltz.mid"
    make_midi(path)
    midi = miditoolkit.MidiFile(str(path))
    midi.time_signature_changes = [miditoolkit.TimeSignature(3, 4, 0)]
    midi.dump(str(path))
    with pytest.raises(ValueError, match="only 4/4"):
        midi_records(path)


def test_imagine_all_notes_and_marker_onsets_preserved():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "imagine.mid"
    records = midi_records(path)
    assert len(records) == 17  # 16 markers plus an unmarked trailing phrase.
    assert sum(len(r["syllables"]) for r in records) == 113
    assert len(records[0]["syllables"]) == 7
    assert records[0]["syllables"][-1]["note"]["start_tick"] == 1800
    assert {r["length_threshold_ticks"] for r in records} == {
        sum(s["note"]["duration_ticks"] for r in records for s in r["syllables"]) / 113
    }


def test_note_comparison_retains_phrases_above_model_slot_limit(tmp_path):
    path = tmp_path / "many-notes.mid"
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    instrument = miditoolkit.Instrument(0)
    instrument.notes = [miditoolkit.Note(80, 60, i * 120, (i + 1) * 120) for i in range(70)]
    midi.instruments = [instrument]
    midi.markers = [miditoolkit.Marker("end", 8400)]
    midi.dump(str(path))
    comparison = midi_note_comparison(path)
    assert comparison["method"] == "supplement"
    assert comparison["role"] == "comparison_only"
    assert comparison["lines"] == midi_records(path, max_syllables=70)
    assert len(comparison["lines"][0]["syllables"]) == 70
