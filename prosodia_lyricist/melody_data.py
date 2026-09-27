"""Paired note/lyric windows from the current try-contrastive preparation manifest."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dali import AnnotationError, interval, parent_index
from .data import read_manifest
from .features import MAX_LINES, TARGET_KEYS, encode_targets, token_id
from .melody_checkpoint import file_sha256
from .melody_encoder.encoding import MELODY_FEATURE_DIM, encode_note_sequence, hz_to_midi_pitch
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


def song_units(record, unit):
    if unit == "song":
        return [record]
    if unit != "paragraph":
        raise ValueError("data.unit must be song or paragraph")
    groups = []
    seen = set()
    previous = None
    for line in record["lines"]:
        paragraph = line.get("paragraph_id")
        if paragraph is None:
            raise ValueError("Paragraph mode requires DALI paragraph IDs on every line")
        if paragraph != previous:
            if paragraph in seen:
                raise ValueError("Non-contiguous DALI paragraph IDs")
            seen.add(paragraph)
            groups.append([])
            previous = paragraph
        groups[-1].append(line)
    return [{**record, "lines": lines} for lines in groups]


def encode_melody_source(record, tokenizer, *, lines_per_window=1, include_title=True):
    """No lyric text, IPA labels or word boundaries enter the source."""
    if not isinstance(lines_per_window, int) or lines_per_window < 1:
        raise ValueError("lines_per_window must be a positive integer")
    lines = record["lines"]
    if not 1 <= len(lines) <= MAX_LINES:
        raise ValueError(f"Expected 1..{MAX_LINES} melody lines")
    ids = [tokenizer.bos_token_id]
    if include_title:
        ids += [token_id(tokenizer, "<title>")]
        ids += tokenizer.encode(record.get("title", ""), add_special_tokens=False)
    windows, positions = [], []
    for offset in range(0, len(lines), lines_per_window):
        pitches, onsets, durations, slots = [], [], [], []
        for index in range(offset, min(offset + lines_per_window, len(lines))):
            ids.append(token_id(tokenizer, f"<sent_{index}>"))
            melody = lines[index]["melody"]
            count = len(melody["midi_pitches"])
            if count < 1 or any(
                len(melody[key]) != count
                for key in (
                    "onset_seconds",
                    "note_duration_seconds",
                )
            ):
                raise ValueError("Invalid melody arrays")
            slots.extend(range(len(ids), len(ids) + count))
            ids.extend([tokenizer.pad_token_id] * count)  # Replaced by note vectors, mask=1.
            pitches.extend(melody["midi_pitches"])
            onsets.extend(melody["onset_seconds"])
            durations.extend(melody["note_duration_seconds"])
        windows.append(
            encode_note_sequence(
                np.array(pitches),
                np.array(onsets) - onsets[0],
                np.array(durations),
            )
        )
        positions.append(slots)
    ids.append(tokenizer.eos_token_id)
    return {"input_ids": ids, "windows": windows, "positions": positions}


class MelodySongDataset(Dataset):
    def __init__(
        self,
        directory,
        split,
        tokenizer,
        *,
        max_source_length,
        max_target_length,
        lines_per_window=1,
        include_title=True,
        unit="song",
        max_window_length=4096,
        limit=None,
    ):
        manifest = read_manifest(directory)
        if not manifest["config"].get("include_melody") or (
            manifest["config"]["stress_source"] != "ipa"
        ):
            raise ValueError("Prepare again with include_melody: true and stress_source: ipa")
        path = Path(directory) / f"{split}.jsonl"
        if file_sha256(path) != manifest["sha256"][path.name]:
            raise ValueError("Prepared data differs from its manifest")
        self.examples, self.skipped, self.songs = [], 0, set()
        with path.open() as handle:
            for row in handle:
                record = json.loads(row)
                if manifest["songs"][record["song_id"]]["split"] != split:
                    raise ValueError("Song appears in incorrect split")
                for part in song_units(record, unit):
                    if len(part["lines"]) > MAX_LINES:
                        self.skipped += 1
                        continue
                    source = encode_melody_source(
                        part,
                        tokenizer,
                        lines_per_window=lines_per_window,
                        include_title=include_title,
                    )
                    targets = encode_targets(part, tokenizer, manifest["config"]["max_syllables"])
                    if (
                        len(source["input_ids"]) > max_source_length
                        or len(targets["labels"]) > max_target_length
                        or max(map(len, source["windows"])) > max_window_length
                    ):
                        self.skipped += 1
                        continue
                    self.examples.append({**source, **targets})
                    self.songs.add(record["song_id"])
                if limit and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(f"No usable {unit} examples in {split}; inspect lengths/preparation")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class MelodyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        width = max(len(e["input_ids"]) for e in examples)
        window_width = max(len(w) for e in examples for w in e["windows"])
        count = sum(len(e["windows"]) for e in examples)
        batch = {
            "input_ids": torch.full((len(examples), width), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros(len(examples), width, dtype=torch.long),
            "melody_features": torch.zeros(count, window_width, MELODY_FEATURE_DIM),
            "melody_attention_mask": torch.zeros(count, window_width, dtype=torch.long),
            "note_positions": torch.full((count, window_width), -1, dtype=torch.long),
        }
        target_keys = [key for key in TARGET_KEYS if key in examples[0]]
        for key in target_keys:
            batch[key] = torch.full(
                (len(examples), max(len(e[key]) for e in examples)),
                -100,
                dtype=torch.long,
            )
        cursor = 0
        for row, example in enumerate(examples):
            n = len(example["input_ids"])
            batch["input_ids"][row, :n] = torch.tensor(example["input_ids"])
            batch["attention_mask"][row, :n] = 1
            for window, positions in zip(example["windows"], example["positions"], strict=True):
                n = len(window)
                batch["melody_features"][cursor, :n] = torch.from_numpy(window)
                batch["melody_attention_mask"][cursor, :n] = 1
                batch["note_positions"][cursor, :n] = torch.tensor(positions) + row * width
                cursor += 1
            for key in target_keys:
                batch[key][row, : len(example[key])] = torch.tensor(example[key])
        return batch
