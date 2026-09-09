import pytest

from prosodia_lyricist.midi import midi_records

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
