"""4/4 MIDI templates; explicit supplement conventions are in docs/evaluation.md."""

import math

from .dali import normalize_text


def metric_stress(start, duration, resolution):
    """Original project's duration-dependent 4-beat heuristic (not observed beats)."""
    durations = {resolution * factor for factor in (1, 0.5, 0.25, 0.125, 0.0625, 2, 4)}
    unit = 4 * duration if duration in durations else 4 * resolution
    beat = int((start % unit) // (unit / 4))
    return "strong" if beat in (0, 2) else "weak"


def supplement_stress(start, duration, resolution):
    """Figure 1's duration-dependent binary pattern, quantized to nearest slot.

    Use the released helper's ordinary note types and quarter-note fallback for
    dotted/irregular durations. Resolve the supplement's unspecified rounding as
    half-up on that note-type grid, anchored at MIDI tick zero. See the fidelity
    discussion in docs/evaluation.md; this is not a literal reading of Eq. (1).
    """
    durations = {resolution * factor for factor in (1, 0.5, 0.25, 0.125, 0.0625, 2, 4)}
    unit = duration if duration in durations else resolution
    position = math.floor(start / unit + 0.5)
    return "strong" if position % 4 in (0, 2) else "weak"


def midi_records(path, *, title="", track=0, max_syllables=64, stress_source="supplement"):
    import miditoolkit

    if stress_source not in ("supplement", "heuristic", "unknown"):
        raise ValueError("MIDI stress_source must be supplement, heuristic or unknown")
    midi = miditoolkit.MidiFile(str(path))
    if stress_source == "supplement" and any(
        (sig.numerator, sig.denominator) != (4, 4) for sig in midi.time_signature_changes
    ):
        raise ValueError("The XAI supplement method supports only 4/4 MIDI")
    if not 0 <= track < len(midi.instruments):
        raise ValueError("MIDI melody track is out of range")
    notes = sorted(midi.instruments[track].notes, key=lambda note: (note.start, note.end))
    if not notes or midi.instruments[track].is_drum:
        raise ValueError("Select a non-empty, non-drum melody track")
    if any(note.start < 0 or note.end <= note.start for note in notes):
        raise ValueError("MIDI contains a negative onset or non-positive note duration")
    if any(left.end > right.start for left, right in zip(notes, notes[1:])):
        raise ValueError("MIDI melody track must be monophonic")
    boundaries = sorted({marker.time for marker in midi.markers})
    if not boundaries:
        raise ValueError("MIDI needs phrase-end markers; see examples/imagine.mid")
    final_marker = boundaries[-1]
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
    melody_mean = sum(note.end - note.start for note in notes) / len(notes)
    for index, group in enumerate(groups):
        if len(group) > max_syllables:
            raise ValueError(
                f"MIDI phrase {index} has {len(group)} notes; limit is {max_syllables}"
            )
        # Supplement Eq. (3): mean of the melody. Retain the old phrase mean
        # only for explicitly selected historical/unknown modes.
        mean_duration = (
            melody_mean
            if stress_source == "supplement"
            else sum(note.end - note.start for note in group) / len(group)
        )
        strength = supplement_stress if stress_source == "supplement" else metric_stress
        syllables = [
            {
                "stress": strength(note.start, note.end - note.start, midi.ticks_per_beat)
                if stress_source != "unknown"
                else "unknown",
                "length": "long" if note.end - note.start > mean_duration else "short",
                "note": {
                    "pitch": note.pitch,
                    "start_tick": note.start,
                    "end_tick": note.end,
                    "duration_ticks": note.end - note.start,
                    "measure": note.start // (4 * midi.ticks_per_beat) + 1,
                    "quarter_beat": (note.start % (4 * midi.ticks_per_beat)) / midi.ticks_per_beat
                    + 1,
                },
            }
            for note in group
        ]
        records.append(
            {
                "id": f"midi:{index}",
                "title": normalize_text(title),
                "syllables": syllables,
                "ticks_per_beat": midi.ticks_per_beat,
                "length_threshold_ticks": mean_duration,
                "stress_source": stress_source,
                "meter": "4/4" if stress_source == "supplement" else "unchecked",
                "meter_assumed": not bool(midi.time_signature_changes),
                "unmarked_tail": group[0].start > final_marker,
            }
        )
    return records
