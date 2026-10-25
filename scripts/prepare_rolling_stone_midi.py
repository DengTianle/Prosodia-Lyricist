"""Convert DALI-absent Rolling Stone songs to marked melody MIDI and annotations.

Uses exactly one MIDI note per .str syllable (first note of each melisma only).
Uses .mel to validate pitches/onsets and recover exact timing, rests and meter,
and nonempty .txt lyric lines for phrases. Refuses mismatched sources.
Durations are estimated, capped at one local measure; tempo is synthetic.
See https://rockcorpus.midside.com/melodic_transcriptions.html and lyrics_stress.html.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import miditoolkit

SCHEME = "rolling_stone_syllabic_lyrics_v2"
NOTE_POLICY = "str_syllabic_notes_first_note_of_each_melisma"
MAJOR = (0, 2, 4, 5, 7, 9, 11)
PITCH_CLASSES = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
TOKEN = re.compile(r"\[[^\]]+\]|R(?:\*\d+)?|[\^v]*[#bn]*[1-7]|[.\-_=()|]")


@dataclass
class Melody:
    notes: list
    measures: list
    rests: list
    end: Fraction


def characters(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    return "".join(c for c in text if c.isalnum())


def parse_melody(text):
    """Parse the corpus grid with exact fractions; retain cross-bar melismas."""
    text = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    tokens = TOKEN.findall(text)
    remainder = TOKEN.sub("", text)
    if remainder.strip():
        raise ValueError(f"Unsupported melody syntax: {remainder.strip()[:80]!r}")
    notes, measures, rests = [], [], []
    meter, key, octave, mode = (4, 4), None, None, set()
    previous, syllable, melisma = None, -1, False
    melisma_first = False
    position, cells = Fraction(0), []
    measure_index = 0

    def declare(token):
        nonlocal meter, key, octave, mode
        value = token[1:-1]
        if re.fullmatch(r"\d+/\d+", value):
            if cells:
                raise ValueError("Time signature inside a measure")
            n, d = map(int, value.split("/"))
            if n < 1 or d < 1 or d & (d - 1):
                raise ValueError(f"Invalid MIDI meter: {value}")
            meter = (n, d)
        elif value == "0":
            raise ValueError("Unmetered passages cannot be converted to metrical MIDI")
        elif re.fullmatch(r"[A-G][#b]?", value):
            key = (PITCH_CLASSES[value[0]] + value.count("#") - value.count("b")) % 12
            mode = set()
        elif re.fullmatch(r"OCT=-?\d+", value):
            octave = int(value[4:])
        elif re.fullmatch(r"[.b]{7}", value):
            if any(c == "b" and i not in (1, 2, 4, 5, 6) for i, c in enumerate(value)):
                raise ValueError(f"Invalid mode signature: {value}")
            mode = {i for i, c in enumerate(value) if c == "b"}
        else:
            raise ValueError(f"Unknown melody declaration: {token}")

    def finish_measure():
        nonlocal position, cells, measure_index
        length = Fraction(meter[0] * 4, meter[1])
        measures.append({"index": measure_index, "start": position, "meter": meter})
        if cells and isinstance(cells[0], tuple) and cells[0][0] == "rest":
            if len(cells) != 1 or melisma:
                raise ValueError("Multi-rest must occupy a whole bar outside a melisma")
            count = cells[0][1]
            rests.append(position)
            for i in range(1, count):
                measures.append(
                    {"index": measure_index + i, "start": position + i * length, "meter": meter}
                )
            position += count * length
            measure_index += count
        else:
            for i, cell in enumerate(cells):
                if cell is not None:
                    notes.append(
                        {
                            **cell,
                            "onset": position + i * length / len(cells),
                            "measure": measure_index,
                            "measure_length": length,
                        }
                    )
            position += length
            measure_index += 1
        cells = []

    for token in tokens:
        if token.startswith("["):
            declare(token)
        elif token == "|":
            finish_measure()
        elif token.startswith("R"):
            if cells:
                raise ValueError("Rest must occupy a complete measure")
            count = int(token[2:]) if "*" in token else 1
            if count < 1:
                raise ValueError("Multi-rest count must be positive")
            cells.append(("rest", count))
        elif token == "(":
            if melisma:
                raise ValueError("Nested melisma")
            melisma, melisma_first = True, True
        elif token == ")":
            if not melisma or melisma_first:
                raise ValueError("Unmatched or empty melisma")
            melisma = False
        elif token in (".", "-", "_", "="):
            cells.extend([None] * {".": 1, "-": 3, "_": 4, "=": 6}[token])
        else:
            if key is None or octave is None:
                raise ValueError("Melody needs initial key and OCT declarations")
            degree = int(token[-1]) - 1
            accidental = token.count("#") - token.count("b")
            if "n" not in token and degree in mode:
                accidental -= 1
            pc = (key + MAJOR[degree] + accidental) % 12
            if previous is None:
                pitch = (octave + 1) * 12 + pc
            else:
                # Closest representative, with the tritone resolved upward.
                pitch = previous + (pc - previous + 5) % 12 - 5
            pitch += 12 * (token.count("^") - token.count("v"))
            if not 0 <= pitch <= 127:
                raise ValueError(f"Pitch outside MIDI range: {pitch}")
            if not melisma or melisma_first:
                syllable += 1
            cells.append({"pitch": pitch, "syllable_index": syllable})
            previous, melisma_first = pitch, False
    if cells or melisma:
        raise ValueError("Unterminated measure or melisma")
    if not notes:
        raise ValueError("Melody has no notes")
    melody = Melody(notes, measures, rests, position)
    estimate_note_ends(melody)
    return melody


def estimate_note_ends(melody):
    """Estimate offsets from the notes actually emitted, not removed continuations."""
    for i, note in enumerate(melody.notes):
        next_onset = melody.notes[i + 1]["onset"] if i + 1 < len(melody.notes) else melody.end
        rest_index = bisect_right(melody.rests, note["onset"])
        next_rest = melody.rests[rest_index] if rest_index < len(melody.rests) else melody.end
        note["end"] = min(next_onset, next_rest, note["onset"] + note["measure_length"])


def syllabic_melody(melody, rows, groups):
    """After source validation, keep precisely the notes represented in .str."""
    notes = [
        {
            **melody.notes[indices[0]],
            "pitch": row["pitch"],
            "source_note_index": indices[0],
            "source_melisma_note_count": len(indices),
        }
        for row, indices in zip(rows, groups, strict=True)
    ]
    result = Melody(notes, melody.measures, melody.rests, melody.end)
    estimate_note_ends(result)
    return result


def read_stress(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path.name}:{number}: expected six fields")
        meter, onset, pitch, degree, stress, word = fields
        match = re.fullmatch(r"(.+)\[(\d+)\]", word)
        if match is None:
            raise ValueError(f"{path.name}:{number}: invalid word/syllable")
        row = dict(
            meter=int(meter),
            onset_measure=Fraction(onset),
            pitch=int(pitch),
            scale_degree=int(degree),
            corpus_stress=int(stress),
            word=match[1],
            word_syllable=int(match[2]),
        )
        if not 0 <= row["pitch"] <= 127 or row["corpus_stress"] not in (0, 1, 2, 3):
            raise ValueError(f"{path.name}:{number}: invalid pitch/stress")
        if rows and row["onset_measure"] <= rows[-1]["onset_measure"]:
            raise ValueError("Syllable onsets must be strictly increasing")
        if row["word_syllable"] != 1 and (
            not rows
            or row["word"] != rows[-1]["word"]
            or row["word_syllable"] != rows[-1]["word_syllable"] + 1
        ):
            raise ValueError(f"{path.name}:{number}: discontinuous word syllables")
        rows.append(row)
    if not rows:
        raise ValueError("Empty stress file")
    return rows


def align_sources(melody, rows, lyrics):
    """Validate pitch/time at every sung syllable, then transfer lyric line ends."""
    groups = []
    for i, note in enumerate(melody.notes):
        if note["syllable_index"] == len(groups):
            groups.append([])
        groups[note["syllable_index"]].append(i)
    if len(groups) != len(rows):
        raise ValueError(f"Melody has {len(groups)} syllables; .str has {len(rows)}")
    for i, (row, indices) in enumerate(zip(rows, groups, strict=True)):
        note = melody.notes[indices[0]]
        measure = melody.measures[note["measure"]]
        onset = note["measure"] + (note["onset"] - measure["start"]) / note["measure_length"]
        n, d = measure["meter"]
        if row["meter"] != n * 100 + d:
            raise ValueError(f"Syllable {i}: .mel/.str meter mismatch")
        # .str prints measure positions to four decimal places.
        if abs(onset - row["onset_measure"]) > Fraction(51, 1_000_000):
            raise ValueError(f"Syllable {i}: .mel/.str onset mismatch ({float(onset):.6f})")
        if note["pitch"] != row["pitch"]:
            raise ValueError(
                f"Syllable {i}: .mel/.str pitch mismatch ({note['pitch']} != {row['pitch']})"
            )

    words, offset = [], 0
    for i, row in enumerate(rows):
        if row["word_syllable"] == 1:
            if not characters(row["word"]):
                raise ValueError("Word normalizes to empty text")
            offset += len(characters(row["word"]))
            words.append({"end_character": offset, "first_syllable": i, "text": row["word"]})
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    expected = "".join(characters(w["text"]) for w in words)
    if expected != "".join(characters(line) for line in lines):
        raise ValueError("Lyrics and .str word streams differ after punctuation normalization")
    if not lines or any(not characters(line) for line in lines):
        raise ValueError("Lyric lines must contain words")
    end_to_word = {w["end_character"]: i for i, w in enumerate(words)}
    phrases, offset, first_syllable = [], 0, 0
    for line in lines:
        offset += len(characters(line))
        if offset not in end_to_word:
            raise ValueError("Lyric line boundary splits a .str word")
        word_index = end_to_word[offset]
        end_syllable = (
            words[word_index + 1]["first_syllable"] if word_index + 1 < len(words) else len(rows)
        )
        first_note, last_note = groups[first_syllable][0], groups[end_syllable - 1][-1]
        phrases.append(
            dict(
                text=line,
                first_note=first_note,
                last_note=last_note,
                first_syllable=first_syllable,
                end_syllable=end_syllable,
            )
        )
        first_syllable = end_syllable
    return groups, phrases


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_song(root, song, overlap_row, overlap_csv, *, tempo=120.0, max_notes=40):
    paths = dict(
        stress=root / "stress-format-space" / f"{song}.str",
        lyrics=root / "rs_lyrics" / f"{song}.txt",
    )
    candidates = sorted((root / "melismas").glob(f"{song}_*.mel"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one matching .mel, found {len(candidates)}")
    paths["melody"] = candidates[0]
    melody = parse_melody(paths["melody"].read_text(encoding="utf-8-sig"))
    if any(m["meter"] != (4, 4) for m in melody.measures):
        raise ValueError("Not entirely 4/4 (including bars without vocal onsets)")
    rows = read_stress(paths["stress"])
    groups, phrases = align_sources(melody, rows, paths["lyrics"].read_text(encoding="utf-8-sig"))
    source_note_count = len(melody.notes)
    melody = syllabic_melody(melody, rows, groups)
    groups = [[i] for i in range(len(rows))]
    for phrase in phrases:
        phrase["first_note"] = phrase["first_syllable"]
        phrase["last_note"] = phrase["end_syllable"] - 1
    sizes = [p["last_note"] - p["first_note"] + 1 for p in phrases]
    if max(sizes) > max_notes:
        raise ValueError(f"Lyric phrase has {max(sizes)} notes; configured limit is {max_notes}")
    # Represent every grid subdivision exactly where MIDI's 15-bit PPQ permits it.
    resolution = math.lcm(480, *(n[k].denominator for n in melody.notes for k in ("onset", "end")))
    if resolution > 32767:
        raise ValueError(f"Exact timing requires MIDI PPQ {resolution}, above 32767")
    midi = miditoolkit.MidiFile(ticks_per_beat=resolution)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    midi.time_signature_changes = []
    previous_meter = None
    for measure in melody.measures:
        if measure["meter"] != previous_meter:
            n, d = measure["meter"]
            midi.time_signature_changes.append(
                miditoolkit.TimeSignature(n, d, int(measure["start"] * resolution))
            )
            previous_meter = measure["meter"]
    instrument = miditoolkit.Instrument(0, name="Vocal melody")
    note_info = []
    for index, note in enumerate(melody.notes):
        start, end = (int(note[k] * resolution) for k in ("onset", "end"))
        instrument.notes.append(miditoolkit.Note(80, note["pitch"], start, end))
        note_info.append(
            dict(
                index=index,
                pitch=note["pitch"],
                start_tick=start,
                end_tick=end,
                syllable_index=note["syllable_index"],
                syllable_start=True,
                source_note_index=note["source_note_index"],
                source_melisma_note_count=note["source_melisma_note_count"],
            )
        )
    midi.instruments = [instrument]
    for i, phrase in enumerate(phrases):
        # Repository grouping uses note.start <= marker.time. Last-note onset
        # avoids swallowing a contiguous first note of the next phrase.
        tick = note_info[phrase["last_note"]]["start_tick"]
        midi.markers.append(miditoolkit.Marker(f"phrase_end_{i:03d}", tick))
        phrase["marker_tick"] = tick
    annotations = {
        "schema": SCHEME,
        "song_id": f"rolling-stone:{song}",
        "title": song.replace("_", " "),
        "dali_overlap": overlap_row,
        "note_policy": NOTE_POLICY,
        "source_note_count": source_note_count,
        "omitted_melisma_continuations": source_note_count - len(rows),
        "sources": {
            k: {"path": str(p.resolve()), "sha256": checksum(p)}
            for k, p in {**paths, "overlap_csv": overlap_csv}.items()
        },
        "timing": {
            "ticks_per_beat": resolution,
            "tempo_bpm": tempo,
            "tempo_source": "synthetic_constant_not_recording_timing",
            "duration_rule": "min(next_syllabic_onset, next_explicit_rest, onset+local_measure)",
            "durations_observed": False,
        },
        "meter_changes": [
            {"numerator": s.numerator, "denominator": s.denominator, "tick": s.time}
            for s in midi.time_signature_changes
        ],
        "phrase_source": "nonempty_rs_lyrics_lines_exact_character_alignment_to_str_words",
        "notes": note_info,
        "phrases": phrases,
        "syllables": [
            {
                **{k: v for k, v in row.items() if k != "onset_measure"},
                "onset_measure": str(row["onset_measure"]),
                "note_indices": indices,
            }
            for row, indices in zip(rows, groups, strict=True)
        ],
        "limitations": [
            "Tempo is synthetic; offsets are estimated, not performed note durations.",
            "Phrase boundaries come from lyric lines, not verified musical phrase annotations.",
            "Only .str syllabic notes are emitted; non-initial melisma notes are omitted. "
            "Retained notes span an estimated duration, not the original pitch contour.",
            "corpus_stress uses RS/CMU conventions (3=nonword), not bridge IPA labels. "
            "IPA vowel-length targets and word-local IPA alignment are not prepared here.",
            "DALI absence is taken from the supplied CSV, not an upstream pretraining audit.",
        ],
    }
    return midi, annotations


def validate_roundtrip(path, annotations):
    from prosodia_lyricist.midi import midi_melody_record, midi_records

    loaded = miditoolkit.MidiFile(str(path))
    if len(loaded.instruments) != 1 or len(loaded.instruments[0].notes) != len(
        annotations["notes"]
    ):
        raise ValueError("MIDI roundtrip lost notes")
    expected = [(n["pitch"], n["start_tick"], n["end_tick"]) for n in annotations["notes"]]
    actual = [(n.pitch, n.start, n.end) for n in loaded.instruments[0].notes]
    if actual != expected:
        raise ValueError("MIDI roundtrip changed notes/timing")
    if len(actual) != len(annotations["syllables"]) or any(
        syllable["note_indices"] != [i] or actual[i][0] != syllable["pitch"]
        for i, syllable in enumerate(annotations["syllables"])
    ):
        raise ValueError("MIDI must have exactly one note per .str syllable, with matching pitch")
    expected_sizes = [p["last_note"] - p["first_note"] + 1 for p in annotations["phrases"]]
    source = midi_melody_record(path)
    counts = [len(line["melody"]["midi_pitches"]) for line in source["lines"]]
    historical = midi_records(path, stress_source="actual", max_syllables=max(expected_sizes))
    if counts != expected_sizes or [len(r["syllables"]) for r in historical] != expected_sizes:
        raise ValueError("Phrase markers changed grouping in the current input parsers")
    if source["unmarked_notes_omitted"] or historical[-1]["unmarked_notes_omitted"]:
        raise ValueError("MIDI roundtrip omitted trailing notes")
    if [(s.numerator, s.denominator, s.time) for s in loaded.time_signature_changes] != [
        (s["numerator"], s["denominator"], s["tick"]) for s in annotations["meter_changes"]
    ]:
        raise ValueError("MIDI roundtrip changed meter")


def prepare(root, overlap_csv, output, *, songs=None, limit=2, tempo=120.0, max_notes=40):
    root, overlap_csv, output = Path(root), Path(overlap_csv), Path(output)
    if not math.isfinite(tempo) or tempo <= 0 or max_notes < 1 or limit < 1:
        raise ValueError("tempo, max_notes and limit must be positive")
    with overlap_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"subset_song", "status"}.issubset(reader.fieldnames or []):
            raise ValueError("Overlap CSV needs subset_song and status columns")
        rows = list(reader)
    by_song = {r["subset_song"]: r for r in rows}
    if len(by_song) != len(rows):
        raise ValueError("Duplicate songs in overlap CSV")
    selected = songs or sorted(s for s, r in by_song.items() if r["status"] == "absent")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate requested songs")
    for song in selected:
        if not re.fullmatch(r"[a-z0-9_]+", song):
            raise ValueError(f"Invalid song slug: {song!r}")
        if song not in by_song or by_song[song]["status"] != "absent":
            raise ValueError(f"{song}: not marked absent in the DALI overlap CSV")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    converted, skipped = [], []
    for song in selected:
        try:
            midi, annotation = build_song(
                root, song, by_song[song], overlap_csv, tempo=tempo, max_notes=max_notes
            )
        except (ValueError, FileNotFoundError) as exc:
            if songs:
                raise ValueError(f"{song}: {exc}") from exc
            skipped.append({"song": song, "reason": str(exc)})
            print(f"Skip {song}: {exc}")
            continue
        path = output / f"{song}.mid"
        midi.dump(str(path))
        validate_roundtrip(path, annotation)
        annotation["midi_sha256"] = checksum(path)
        (output / f"{song}.annotations.json").write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output / f"{song}.lyrics.txt").write_text(
            "".join(p["text"] + "\n" for p in annotation["phrases"]), encoding="utf-8"
        )
        counts = [p["last_note"] - p["first_note"] + 1 for p in annotation["phrases"]]
        entry = {
            "song": song,
            "midi": path.name,
            "annotations": f"{song}.annotations.json",
            "lyrics": f"{song}.lyrics.txt",
            "notes": len(annotation["notes"]),
            "syllables": len(annotation["syllables"]),
            "phrases": len(counts),
            "max_phrase_notes": max(counts),
            "omitted_melisma_continuations": annotation["omitted_melisma_continuations"],
            "midi_sha256": annotation["midi_sha256"],
        }
        converted.append(entry)
        print(
            f"Converted {song}: {entry['notes']} notes, {entry['syllables']} syllables, "
            f"{entry['phrases']} phrases (largest {max(counts)} notes)"
        )
        if not songs and len(converted) >= limit:
            break
    manifest = dict(
        schema=SCHEME,
        note_policy=NOTE_POLICY,
        selection="status=absent; entirely 4/4; source alignment validated",
        overlap_csv=str(overlap_csv.resolve()),
        overlap_csv_sha256=checksum(overlap_csv),
        source_root=str(root.resolve()),
        tempo_bpm=tempo,
        max_phrase_notes=max_notes,
        requested_songs=songs,
        requested_limit=None if songs else limit,
        converted=converted,
        skipped=skipped,
        evaluation_ready=False,
        evaluation_remaining="Prepare bridge IPA targets and word-local note alignment; "
        "review lyric phrase boundaries and estimated timing before scoring.",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if len(converted) < (len(songs) if songs else limit):
        raise ValueError(
            f"Only {len(converted)} songs converted; inspect {output / 'manifest.json'}"
        )
    return manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/rolling-stone")
    parser.add_argument(
        "--overlap-csv", type=Path, default=root / "data/rolling_stone_80_dali_v2_overlap.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data/rolling-stone/midi-pilot")
    parser.add_argument("--songs", nargs="+", help="Explicit absent song slugs; overrides --limit")
    parser.add_argument(
        "--limit", type=int, default=2, help="First N passing songs, alphabetically"
    )
    parser.add_argument("--tempo", type=float, default=120.0, help="Synthetic constant BPM")
    parser.add_argument(
        "--max-phrase-notes",
        type=int,
        default=40,
        help="Reject, never split/truncate, longer lyric lines",
    )
    args = parser.parse_args()
    try:
        prepare(
            args.data_dir,
            args.overlap_csv,
            args.output_dir,
            songs=args.songs,
            limit=args.limit,
            tempo=args.tempo,
            max_notes=args.max_phrase_notes,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
