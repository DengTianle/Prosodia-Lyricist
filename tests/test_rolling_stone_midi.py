"""Source alignment and MIDI contracts for the external evaluation pilot."""

import csv
import json
from fractions import Fraction

import pytest

pytest.importorskip("miditoolkit")

from scripts.prepare_rolling_stone_midi import (  # noqa: E402
    align_sources,
    parse_melody,
    prepare,
    read_stress,
)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    for folder in ("melismas", "stress-format-space", "rs_lyrics"):
        (root / folder).mkdir(parents=True)
    # The melisma crosses a barline. Empty bars have no new onset, not rests.
    (root / "melismas/song_dt.mel").write_text(
        "[C] [OCT=4] 1.(2 | 32)4. | R*2 | 5 |", encoding="utf-8"
    )
    (root / "stress-format-space/song.str").write_text(
        "404 0.0000 60 0 1 HELLO[1]\n"
        "404 0.6667 62 2 0 HELLO[2]\n"
        "404 1.5000 65 5 1 WORLD[1]\n"
        "404 4.0000 67 7 3 AH[1]\n",
        encoding="utf-8",
    )
    (root / "rs_lyrics/song.txt").write_text("Hello!\n\nWorld\nAh\n", encoding="utf-8")
    overlap = tmp_path / "overlap.csv"
    with overlap.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subset_song", "status"])
        writer.writeheader()
        writer.writerow(dict(subset_song="song", status="absent"))
        writer.writerow(dict(subset_song="cover", status="cover_only"))
    return root, overlap, tmp_path / "output"


def test_syllabic_notes_melisma_alignment_and_phrase_roundtrip(corpus):
    root, overlap, output = corpus
    manifest = prepare(root, overlap, output, limit=1)
    entry = manifest["converted"][0]
    assert (entry["notes"], entry["syllables"], entry["phrases"]) == (4, 4, 3)
    annotation = json.loads((output / entry["annotations"]).read_text())
    assert annotation["syllables"][1]["note_indices"] == [1]
    assert [p["last_note"] for p in annotation["phrases"]] == [1, 2, 3]
    assert [n["pitch"] for n in annotation["notes"]] == [60, 62, 65, 67]
    assert [n["source_note_index"] for n in annotation["notes"]] == [0, 1, 4, 5]
    assert annotation["notes"][1]["source_melisma_note_count"] == 3
    assert annotation["source_note_count"] == 6
    assert annotation["omitted_melisma_continuations"] == 2
    # The retained melisma onset extends to the next syllable, rather than
    # ending at the first removed continuation (quarter beat 4).
    resolution = annotation["timing"]["ticks_per_beat"]
    assert annotation["notes"][1]["end_tick"] == 6 * resolution
    # Last note of the second phrase ends at the explicit rest, not after it.
    assert annotation["notes"][2]["end_tick"] == 8 * resolution
    assert annotation["meter_changes"] == [dict(numerator=4, denominator=4, tick=0)]
    assert annotation["timing"]["durations_observed"] is False
    assert manifest["evaluation_ready"] is False
    assert (output / entry["lyrics"]).read_text() == "Hello!\nWorld\nAh\n"


def test_mode_accidentals_octaves_and_tritone():
    melody = parse_melody("[C] [OCT=4] [..b..bb] 3n3^3v3 | [D] 17 |")
    assert [n["pitch"] for n in melody.notes] == [63, 64, 75, 63, 62, 61]
    assert [n["pitch"] for n in parse_melody("[C] [OCT=4] 1#4 | 1 |").notes] == [60, 66, 72]


def test_repeating_melisma_is_one_note_per_lyric_syllable(corpus):
    root, overlap, output = corpus
    # Regression for the TURN/YOU/ON passages of A Day in the Life.
    (root / "melismas/song_dt.mel").write_text(
        "[G] [OCT=3] (34 34 34 34) | (34 34 34 34) | (34 34 34 34) |"
    )
    (root / "stress-format-space/song.str").write_text(
        "404 0.0000 59 4 1 TURN[1]\n404 1.0000 59 4 0 YOU[1]\n404 2.0000 59 4 0 ON[1]\n"
    )
    (root / "rs_lyrics/song.txt").write_text("Turn you on\n")
    prepare(root, overlap, output, songs=["song"])
    annotation = json.loads((output / "song.annotations.json").read_text())
    assert [n["pitch"] for n in annotation["notes"]] == [59, 59, 59]
    assert [n["start_tick"] for n in annotation["notes"]] == [0, 1920, 3840]
    assert [n["end_tick"] for n in annotation["notes"]] == [1920, 3840, 5760]
    assert annotation["omitted_melisma_continuations"] == 21
    assert annotation["phrases"][0]["last_note"] == 2


def test_subdivision_and_multidot_symbols():
    melody = parse_melody("[C] [OCT=4] 1-2_3= |")
    # 3 notes + 3 + 4 + 6 no-onset cells: 16 cells total.
    assert [n["onset"] for n in melody.notes] == [Fraction(0), Fraction(1), Fraction(9, 4)]
    assert parse_melody("[C] [OCT=4] 1 | | | 2 |").notes[0]["end"] == 4


def test_meter_in_silent_bars_is_preserved_and_filtered(corpus):
    root, overlap, output = corpus
    text = "[C] [OCT=4] 1 | [2/4] R*2 | [4/4] 2 |"
    melody = parse_melody(text)
    assert [m["meter"] for m in melody.measures] == [(4, 4), (2, 4), (2, 4), (4, 4)]
    assert melody.notes[-1]["onset"] == 8
    (root / "melismas/song_dt.mel").write_text(text)
    with pytest.raises(ValueError, match="Not entirely 4/4"):
        prepare(root, overlap, output, songs=["song"])


@pytest.mark.parametrize(
    "change,message",
    [
        (("HELLO", "OTHER"), "word streams differ"),
        (("0.6667", "0.7000"), "onset mismatch"),
        (("62 2", "63 2"), "pitch mismatch"),
        (("HELLO[2]", "HELLO[3]"), "discontinuous word"),
    ],
)
def test_source_mismatches_are_never_guessed(corpus, change, message):
    root, overlap, output = corpus
    path = root / "stress-format-space/song.str"
    path.write_text(path.read_text().replace(*change))
    with pytest.raises(ValueError, match=message):
        prepare(root, overlap, output, songs=["song"])
    assert not list(output.glob("*.mid"))


def test_line_boundary_must_not_split_a_word(corpus):
    root, _, _ = corpus
    melody = parse_melody((root / "melismas/song_dt.mel").read_text())
    rows = read_stress(root / "stress-format-space/song.str")
    with pytest.raises(ValueError, match="splits a .str word"):
        align_sources(melody, rows, "Hel\nlo World Ah")


def test_selection_excludes_covers_and_protects_outputs(corpus):
    root, overlap, output = corpus
    with pytest.raises(ValueError, match="not marked absent"):
        prepare(root, overlap, output, songs=["cover"])
    prepare(root, overlap, output, songs=["song"])
    with pytest.raises(FileExistsError, match="must be empty"):
        prepare(root, overlap, output, songs=["song"])


@pytest.mark.parametrize(
    "text",
    [
        "[C] [OCT=4] 1",  # Missing final barline.
        "[C] [OCT=4] (1 |",  # Open melisma.
        "[C] [OCT=4] 1((2)) |",
        "[C] [OCT=4] 1x2 |",
        "[C] [OCT=4] R*0 | 1 |",
        "[C] [OCT=4] [0] 1 |",
    ],
)
def test_malformed_melodies_rejected(text):
    with pytest.raises(ValueError):
        parse_melody(text)
