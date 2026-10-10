"""DALI melody extraction and split provenance shared with prosodia-direct."""

import csv
import json
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


def read_melody_manifest(path, *, require_lyrics=True, allow_split_overlap=False):
    """Read identities; exposure audits may retain every split of an upstream song."""
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
        splits.setdefault(song, set()).add(row["split"])
        if len(splits[song]) > 1 and not allow_split_overlap:
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
    if not allow_split_overlap:
        groups = song_groups(songs)
        group_splits = {}
        for song, group in groups.items():
            if group_splits.setdefault(group, splits[song]) != splits[song]:
                raise ValueError(f"Duplicate song/audio group {group} crosses splits")
    return rows, {
        song: {**info, "split": "train" if "train" in splits[song] else sorted(splits[song])[0],
               "splits": sorted(splits[song])}
        for song, info in songs.items()
    }


def pretraining_identity_groups(songs, upstream):
    """Join IDs, artist/title, audio and persisted groups across both datasets."""
    combined = {f"up:{song}": info for song, info in upstream.items()}
    combined.update({f"down:{song}": info for song, info in songs.items()})
    parents = song_groups(combined)

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        parents[root(a)] = root(b)

    for song in songs.keys() & upstream.keys():
        join(f"up:{song}", f"down:{song}")
    known_groups = {}
    for song, info in songs.items():
        key = f"down:{song}"
        if "group" in info:
            join(key, known_groups.setdefault(info["group"], key))
    groups = defaultdict(lambda: {"upstream": [], "downstream": [], "identities": {}})
    for side, prefix, entries in (("upstream", "up", upstream), ("downstream", "down", songs)):
        for song, info in sorted(entries.items()):
            key = f"{prefix}:{song}"
            group = groups[root(key)]
            for split in info.get("splits", [info["split"]]):
                group[side].append({"song_id": song, "split": split})
            group["identities"][key] = {
                "artist": info.get("artist", ""), "title": info.get("title", ""),
                "audio": info.get("audio", {}).get("url", ""),
            }
    return list(groups.values())


def pretraining_overlap_report(songs, upstream):
    """Only upstream training exposure threatens downstream held-out evaluation."""
    report = {
        "policy": "protect_downstream_heldout_v1", "conflicts": [],
        "allowed_overlaps": [], "upstream_split_overlaps": [],
        "shared_songs": len(songs.keys() & upstream.keys()),
        "downstream_only_songs": len(songs.keys() - upstream.keys()),
    }
    for group in pretraining_identity_groups(songs, upstream):
        up = {row["split"] for row in group["upstream"]}
        down = {row["split"] for row in group["downstream"]}
        if len(up) > 1:
            report["upstream_split_overlaps"].append(group)
        if "train" in up and down & {"valid", "test"}:
            report["conflicts"].append({
                "kind": "pretraining_train_in_downstream_heldout", **group,
            })
        elif len(down) > 1:
            report["conflicts"].append({"kind": "downstream_duplicate_crosses_splits", **group})
        elif up and down and len(up | down) > 1:
            report["allowed_overlaps"].append(group)
    return report


def audit_pretraining_splits(songs, manifest, *, upstream=None):
    if upstream is None:
        _, upstream = read_melody_manifest(
            manifest, require_lyrics=False, allow_split_overlap=True,
        )
    report = pretraining_overlap_report(songs, upstream)
    if report["conflicts"]:
        raise ValueError("Unsafe downstream split group: " + json.dumps(report["conflicts"][0]))
    return {
        **report,
        "manifest": str(Path(manifest).resolve()),
        "sha256": file_sha256(manifest),
    }


def preserve_pretraining_splits(songs, splits, manifest, *, upstream=None):
    """Force upstream-trained identity groups into train; retain other random splits."""
    if upstream is None:
        _, upstream = read_melody_manifest(
            manifest, require_lyrics=False, allow_split_overlap=True,
        )
    result = dict(splits)
    downstream = {song: {**info, "split": splits[song]} for song, info in songs.items()}
    for group in pretraining_identity_groups(downstream, upstream):
        ids = sorted({row["song_id"] for row in group["downstream"]})
        if not ids:
            continue
        # Upstream identities can join formerly separate downstream random groups.
        trained = any(row["split"] == "train" for row in group["upstream"])
        split = "train" if trained else splits[ids[0]]
        result.update(dict.fromkeys(ids, split))
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
