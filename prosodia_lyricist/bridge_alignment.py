"""Observed DALI word/note links for direct note-head supervision."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ALIGNMENT_SCHEME = "dali_word_note_starts_v1"
ALIGNMENT_POLICIES = ("strict", "prefix")


def _characters(text):
    # Match the IPA parser's spelling normalization, allowing punctuation and
    # hyphen/token boundaries to differ without moving labels between words.
    text = str(text).casefold().replace("cuz", "cause")
    return "".join(c for c in text if c.isalnum())


def prepare_note_alignment(record, words):
    """Link IPA slots to the first note of each raw sung group, within words.

    Word counts are deliberately NOT repaired here. Training selects strict or
    prefix matching later. Tokenization splits are accepted only when the full
    normalized character streams match and no IPA token crosses a raw word.
    """
    from .dali import sung_syllables

    targets, character_offset, syllable_offset = [], 0, 0
    for word in record["words"]:
        size = len(_characters(word["text"]))
        count = word["syllable_count"]
        targets.append(
            (
                character_offset,
                character_offset + size,
                list(range(syllable_offset, syllable_offset + count)),
            )
        )
        character_offset += size
        syllable_offset += count
    if syllable_offset != len(record["syllables"]):
        raise ValueError("IPA word counts differ from the line skeleton")
    raw_text = "".join(_characters(text) for text, _ in words)
    ipa_text = "".join(_characters(w["text"]) for w in record["words"])
    result = {
        "scheme": ALIGNMENT_SCHEME,
        "note_count": sum(len(notes) for _, notes in words),
        "words": [],
    }
    if raw_text != ipa_text:
        return {**result, "unmatched_reason": "word_text_mismatch"}
    character_offset, note_offset = 0, 0
    for text, notes in words:
        end = character_offset + len(_characters(text))
        groups = sung_syllables(notes)
        starts, position = [], note_offset
        for group in groups:
            starts.append(position)
            position += group["note_count"]
        indices = []
        for lo, hi, slots in targets:
            if slots and lo < end and hi > character_offset:
                if lo < character_offset or hi > end:
                    return {**result, "words": [], "unmatched_reason": "cross_word_token"}
                indices.extend(slots)
        result["words"].append({"note_starts": starts, "syllable_indices": indices})
        character_offset, note_offset = end, position
    return result


def recover_note_alignment(song, dali_dir):
    """Enrich an old prepared song in memory; never rerun IPA or rewrite splits."""
    from .dali import aligned_words, interval, parent_index, read_annotation
    from .melody_checkpoint import file_sha256
    from .melody_encoder.encoding import hz_to_midi_pitch

    if not dali_dir:
        raise ValueError("Note heads need word/note alignment: prepare again or supply dali_dir")
    directory = Path(dali_dir)
    song_id = song["song_id"]
    candidates = [directory / f"{song_id}{suffix}" for suffix in (".gz", ".json")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError(
            f"Missing raw DALI annotation for {song_id} in {directory}; "
            "prepare again with note alignment, or supply the local --dali-dir"
        )
    checksum = file_sha256(path)
    info, annot = read_annotation(path)
    if info["id"] != song_id:
        raise ValueError("Raw DALI identity differs from prepared song")
    notes_by_word, words_by_line = defaultdict(list), defaultdict(list)
    for note in annot["notes"]:
        notes_by_word[parent_index(note, len(annot["words"]))].append(note)
    for index, word in enumerate(annot["words"]):
        words_by_line[parent_index(word, len(annot["lines"]))].append((index, word))
    for line in song["lines"]:
        if "note_alignment" in line:
            continue
        start, end = interval(annot["lines"][line["line_id"]])
        words = aligned_words(words_by_line[line["line_id"]], notes_by_word, start, end)
        notes = [note for _, group in words for note in group]
        expected = line["melody"]
        values = {
            "midi_pitches": [
                hz_to_midi_pitch(float(np.asarray(n["freq"], dtype=np.float32).reshape(-1)[0]))
                for n in notes
            ],
            "onset_seconds": [float(n["time"][0]) for n in notes],
            "note_duration_seconds": [float(n["time"][1] - n["time"][0]) for n in notes],
        }
        if any(
            len(values[key]) != len(expected[key])
            or not np.allclose(values[key], expected[key], rtol=0, atol=1e-6)
            for key in values
        ):
            raise ValueError("Raw DALI notes differ from the prepared melody; prepare again")
        line["note_alignment"] = prepare_note_alignment(line, words)
    return checksum


def aligned_note_indices(lines, policy="strict"):
    """One source index per serialized IPA position; -1 means no supervision."""
    if policy not in ALIGNMENT_POLICIES:
        raise ValueError("alignment_policy must be strict or prefix")
    indices, counts, note_offset = [], Counter(), 0
    for line in lines:
        alignment = line.get("note_alignment", {})
        n, p = len(line["melody"]["midi_pitches"]), len(line["syllables"])
        if alignment.get("scheme") != ALIGNMENT_SCHEME or alignment.get("note_count") != n:
            raise ValueError("Missing/incompatible prepared note alignment; prepare again")
        selected, used_notes, used_slots = [-1] * p, set(), set()
        counts.update(ipa_slots=p, notes=n)
        if alignment.get("unmatched_reason"):
            counts["unmatched_text_lines"] += 1
        for word in alignment["words"]:
            starts, slots = word["note_starts"], word["syllable_indices"]
            for values, bound, used in ((starts, n, used_notes), (slots, p, used_slots)):
                if any(type(v) is not int or not 0 <= v < bound or v in used for v in values):
                    raise ValueError("Invalid or duplicate word/note alignment index")
                if any(a >= b for a, b in zip(values, values[1:])):
                    raise ValueError("Unordered word/note alignment indices")
                used.update(values)
            counts["words"] += 1
            if len(starts) != len(slots):
                counts["mismatched_words"] += 1
                if policy == "strict":
                    continue
            else:
                counts["matched_words"] += 1
            for note, slot in zip(starts, slots):
                selected[slot] = note_offset + note
        counts["supervised_slots"] += sum(i >= 0 for i in selected)
        counts["masked_slots"] += sum(i < 0 for i in selected)
        counts["fully_aligned_lines"] += int(all(i >= 0 for i in selected))
        indices.extend([-1, *selected])  # Fixed line prefix has no note target.
        note_offset += n
    return indices, counts
