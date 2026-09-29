"""DALI melody extraction and split provenance shared with prosodia-direct."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dali import AnnotationError, interval, parent_index
from .melody_checkpoint import file_sha256
from .melody_encoder.encoding import hz_to_midi_pitch
from .prepare import song_groups


def canonical_split(value):
    value = {"val": "valid", "validation": "valid"}.get(value, value)
    if value not in ("train", "valid", "test"):
        raise ValueError(f"Unknown split {value!r}; expected train, val/valid, test")
    return value


def read_melody_manifest(path, *, require_lyrics=True):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "dali_id", "split"}
        if require_lyrics:
            required |= {"melody_path", "line_text", "line_count", "note_count"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(
                "Expected current two-pool note/lyric manifest; rerun "
                "try-contrastive/scripts/prepare_dali_dataset.py (old frame data is unsupported)"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Empty melody manifest")
    songs, splits, ids = {}, {}, set()
    for row in rows:
        sample, song = row["sample_id"], row["dali_id"]
        if not song or not sample or sample in ids:
            raise ValueError("Manifest has empty song/sample IDs or duplicate sample IDs")
        ids.add(sample)
        row["split"] = canonical_split(row["split"])
        if splits.setdefault(song, row["split"]) != row["split"]:
            raise ValueError(f"Song {song} crosses splits")
        audio = row.get("raw_audio_path") or row.get("audio_path")
        info = {
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "audio": {"url": str((path.parent / audio).resolve()) if audio else ""},
        }
        if song in songs and songs[song] != info:
            raise ValueError(f"Inconsistent song identity for {song}")
        songs[song] = info
        if require_lyrics:
            row["melody_path"] = str((path.parent / row["melody_path"]).resolve())
            lines = row["line_text"].splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError(f"Missing aligned lyric text for {sample}")
            if len(lines) != int(row["line_count"]):
                raise ValueError(f"Lyric line_count mismatch for {sample}")
    groups = song_groups(songs)
    group_splits = {}
    for song, group in groups.items():
        if group_splits.setdefault(group, splits[song]) != splits[song]:
            raise ValueError(f"Duplicate song/audio group {group} crosses splits")
    return rows, {song: {**info, "split": splits[song]} for song, info in songs.items()}


def audit_pretraining_splits(songs, manifest):
    """Require shared songs (including duplicate identities) to keep their split.

    This is conservative: it also prevents moving pretraining validation songs
    into downstream training and does not claim to detect undocumented exposure.
    """
    _, upstream = read_melody_manifest(manifest, require_lyrics=False)
    combined = {f"up:{song}": info for song, info in upstream.items()}
    combined.update({f"down:{song}": info for song, info in songs.items()})
    groups = song_groups(combined)
    memberships = defaultdict(set)
    for key, info in combined.items():
        memberships[groups[key]].add(info["split"])
    for song, info in songs.items():
        if song in upstream and info["split"] != upstream[song]["split"]:
            raise ValueError(f"Pretraining/downstream split mismatch for song {song}")
        if len(memberships[groups[f"down:{song}"]]) != 1:
            raise ValueError(f"Pretraining/downstream duplicate group crosses splits: {song}")
    return {
        "manifest": str(Path(manifest).resolve()),
        "sha256": file_sha256(manifest),
        "shared_songs": len(songs.keys() & upstream.keys()),
        "downstream_only_songs": len(songs.keys() - upstream.keys()),
    }


def preserve_pretraining_splits(songs, splits, manifest):
    """Reuse upstream assignments for shared songs and duplicate identity groups."""
    _, upstream = read_melody_manifest(manifest, require_lyrics=False)
    combined = {f"up:{key}": info for key, info in upstream.items()}
    combined.update({f"down:{key}": info for key, info in songs.items()})
    groups = song_groups(combined)
    assigned = {}
    for key, info in upstream.items():
        group = groups[f"up:{key}"]
        if assigned.setdefault(group, info["split"]) != info["split"]:
            raise ValueError("Pretraining duplicate groups cross splits")
    result = {}
    for key in songs:
        group_split = assigned.get(groups[f"down:{key}"])
        same_id_split = upstream.get(key, {}).get("split")
        if group_split and same_id_split and group_split != same_id_split:
            raise ValueError(f"Conflicting pretraining identity for {key}")
        result[key] = same_id_split or group_split or splits[key]
    audit_pretraining_splits(
        {key: {**info, "split": result[key]} for key, info in songs.items()}, manifest
    )
    return result


def attach_melody(records, annot):
    """Retain every aligned note (including melisma) using upstream pitch rounding."""
    by_line = defaultdict(list)
    for note in annot["notes"]:
        word = annot["words"][parent_index(note, len(annot["words"]))]
        line = parent_index(word, len(annot["lines"]))
        start, end = interval(note)
        try:
            frequency = np.asarray(note["freq"], dtype=np.float32).reshape(-1)
            pitch = hz_to_midi_pitch(float(frequency[0]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnnotationError("Missing or invalid note frequency") from exc
        by_line[line].append((pitch, start, end - start))
    previous_end = -1.0
    for record in records:
        if record["start"] < previous_end - 1e-6:
            raise AnnotationError("Overlapping or unordered melody lines")
        previous_end = record["end"]
        values = by_line[record["line_id"]]
        if not values:
            raise AnnotationError("Melody line has no notes")
        record["melody"] = dict(
            zip(
                ("midi_pitches", "onset_seconds", "note_duration_seconds"),
                map(list, zip(*values)),
                strict=True,
            )
        )
        paragraph = annot["lines"][record["line_id"]].get("index")
        record["paragraph_id"] = (
            int(paragraph) if isinstance(paragraph, (int, np.integer)) and paragraph >= 0 else None
        )
