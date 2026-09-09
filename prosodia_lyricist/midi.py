"""MIDI phrase conversion retained for evaluation experiments.

Markers are phrase ends, as in the original example MIDI. Beat/bar marker
interpretation is deliberately left for the later evaluation workflow.
"""

from .dali import normalize_text


def metric_stress(start, duration, resolution):
    """Original project's duration-dependent 4-beat heuristic (not observed beats)."""
    durations = {resolution * factor for factor in (1, 0.5, 0.25, 0.125, 0.0625, 2, 4)}
    unit = 4 * duration if duration in durations else 4 * resolution
    beat = int((start % unit) // (unit / 4))
    return "strong" if beat in (0, 2) else "weak"


def midi_records(path, *, title="", track=0, max_syllables=64, stress_source="heuristic"):
    import miditoolkit

    if stress_source not in ("heuristic", "unknown"):
        raise ValueError("MIDI stress_source must be heuristic or unknown")
    midi = miditoolkit.MidiFile(str(path))
    if not 0 <= track < len(midi.instruments):
        raise ValueError("MIDI melody track is out of range")
    notes = sorted(midi.instruments[track].notes, key=lambda note: (note.start, note.end))
    if not notes or midi.instruments[track].is_drum:
        raise ValueError("Select a non-empty, non-drum melody track")
    if any(note.end <= note.start for note in notes):
        raise ValueError("MIDI contains a non-positive note duration")
    if any(left.end > right.start for left, right in zip(notes, notes[1:])):
        raise ValueError("MIDI melody track must be monophonic")
    boundaries = sorted({marker.time for marker in midi.markers})
    if not boundaries:
        raise ValueError("MIDI needs phrase-end markers; see examples/imagine.mid")
    if boundaries[-1] < notes[-1].end:
        boundaries.append(notes[-1].end)
    groups, current, cursor = [], [], 0
    # Phrase-end markers include notes starting exactly at the marker, matching the old pipeline.
    for boundary in boundaries:
        current = []
        while cursor < len(notes) and notes[cursor].start <= boundary:
            current.append(notes[cursor])
            cursor += 1
        if current:
            groups.append(current)
    records = []
    for index, group in enumerate(groups):
        if len(group) > max_syllables:
            raise ValueError(
                f"MIDI phrase {index} has {len(group)} notes; limit is {max_syllables}"
            )
        mean_duration = sum(note.end - note.start for note in group) / len(group)
        syllables = [
            {
                "stress": metric_stress(note.start, note.end - note.start, midi.ticks_per_beat)
                if stress_source == "heuristic"
                else "unknown",
                "length": "long" if note.end - note.start > mean_duration else "short",
            }
            for note in group
        ]
        records.append(
            {"id": f"midi:{index}", "title": normalize_text(title), "syllables": syllables}
        )
    return records
