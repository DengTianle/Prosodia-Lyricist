"""Melody windows as inputs; variable-length IPA templates as supervision."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import read_manifest
from .melody_checkpoint import file_sha256
from .melody_encoder.encoding import MELODY_FEATURE_DIM, encode_note_sequence

PAD, BOS, SLOT = range(3)
PAIRS = (("strong", "long"), ("strong", "short"), ("weak", "long"), ("weak", "short"))
PAIR_OFFSET = 3
FIRST_LINE = PAIR_OFFSET + len(PAIRS)


def bridge_vocabulary(lines_per_window):
    return (
        ["pad", "bos", "slot"]
        + [f"{s}_{n}" for s, n in PAIRS]
        + [f"line_{index}" for index in range(lines_per_window)]
    )


def encode_bridge_source(lines):
    """Only note arrays and phrase membership enter the pretrained melody tower."""
    if not lines:
        raise ValueError("Expected at least one melody phrase")
    pitches, onsets, durations, line_ids = [], [], [], []
    for index, line in enumerate(lines, 1):
        melody = line["melody"]
        pitch, onset, duration = (
            np.asarray(melody[key])
            for key in ("midi_pitches", "onset_seconds", "note_duration_seconds")
        )
        if (
            pitch.ndim != 1
            or not len(pitch)
            or pitch.shape != onset.shape
            or (pitch.shape != duration.shape)
        ):
            raise ValueError("Invalid melody arrays")
        pitches.extend(pitch)
        onsets.extend(onset)
        durations.extend(duration)
        line_ids.extend([index] * len(pitch))
    return {
        "melody_features": encode_note_sequence(
            np.asarray(pitches), np.asarray(onsets) - onsets[0], np.asarray(durations)
        ),
        "note_line_ids": line_ids,
        "line_count": len(lines),
    }


def encode_bridge_targets(lines, max_syllables):
    """DALI line membership fixes the prefixes; IPA supplies syllables within each line."""
    labels = []
    for index, line in enumerate(lines):
        if not 1 <= len(line["syllables"]) <= max_syllables:
            raise ValueError("IPA template exceeds the per-phrase syllable limit")
        labels.append(FIRST_LINE + index)
        for syllable in line["syllables"]:
            pair = syllable["stress"], syllable["length"]
            if pair not in PAIRS:
                raise ValueError("Bridge targets require binary IPA stress and vowel length")
            labels.append(PAIR_OFFSET + PAIRS.index(pair))
    return labels


def decode_bridge_tokens(tokens):
    lines, syllables, padded = [], None, False
    for token in tokens:
        if token == PAD:
            padded = True
            continue
        if padded:
            raise ValueError("Template padding must be trailing")
        if token >= FIRST_LINE:
            if token != FIRST_LINE + len(lines):
                raise ValueError("Template line IDs must be consecutive, starting at line_0")
            if syllables == []:
                raise ValueError("Empty template phrase")
            syllables = []
            lines.append({"syllables": syllables})
        elif PAIR_OFFSET <= token < FIRST_LINE and syllables is not None:
            stress, length = PAIRS[token - PAIR_OFFSET]
            syllables.append({"stress": stress, "length": length})
        else:
            raise ValueError("Expected a line prefix followed by prosody pairs")
    if not syllables:
        raise ValueError("Incomplete predicted template")
    return lines


class BridgeDataset(Dataset):
    def __init__(self, directory, split, *, lines_per_window, max_notes, limit=None):
        if lines_per_window < 1 or max_notes < 1:
            raise ValueError("Window sizes must be positive")
        manifest = read_manifest(directory)
        data = manifest["config"]
        if not data.get("include_melody") or data["stress_source"] != "ipa":
            raise ValueError("Prepare with include_melody: true and stress_source: ipa")
        path = Path(directory) / f"{split}.jsonl"
        if file_sha256(path) != manifest["sha256"][path.name]:
            raise ValueError("Prepared data differs from its manifest; prepare again")
        self.examples, self.skipped, self.songs = [], 0, set()
        with path.open(encoding="utf-8") as handle:
            for row in handle:
                song = json.loads(row)
                if manifest["songs"][song["song_id"]]["split"] != split:
                    raise ValueError("Song appears in incorrect split")
                for offset in range(0, len(song["lines"]), lines_per_window):
                    lines = song["lines"][offset : offset + lines_per_window]
                    source = encode_bridge_source(lines)
                    if len(source["note_line_ids"]) > max_notes:
                        self.skipped += 1
                        continue
                    self.examples.append(
                        {**source, "labels": encode_bridge_targets(lines, data["max_syllables"])}
                    )
                    self.songs.add(song["song_id"])
                    if limit is not None and len(self.examples) >= limit:
                        break
                if limit is not None and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(f"No usable bridge windows in {split}; inspect preparation/limits")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def bridge_collate(examples):
    width = max(len(e["note_line_ids"]) for e in examples)
    batch = {
        "melody_features": torch.zeros(len(examples), width, MELODY_FEATURE_DIM),
        "melody_attention_mask": torch.zeros(len(examples), width, dtype=torch.long),
        "note_line_ids": torch.zeros(len(examples), width, dtype=torch.long),
        "line_counts": torch.tensor([e["line_count"] for e in examples]),
    }
    if "labels" in examples[0]:
        batch["labels"] = torch.full(
            (len(examples), max(len(e["labels"]) for e in examples)), -100, dtype=torch.long
        )
        batch["syllable_counts"] = torch.zeros(
            len(examples), int(batch["line_counts"].max()), dtype=torch.long
        )
    for row, example in enumerate(examples):
        n = len(example["note_line_ids"])
        batch["melody_features"][row, :n] = torch.from_numpy(example["melody_features"])
        batch["melody_attention_mask"][row, :n] = 1
        batch["note_line_ids"][row, :n] = torch.tensor(example["note_line_ids"])
        if "labels" in batch:
            batch["labels"][row, : len(example["labels"])] = torch.tensor(example["labels"])
            counts = [len(line["syllables"]) for line in decode_bridge_tokens(example["labels"])]
            if len(counts) != example["line_count"]:
                raise ValueError("Target line skeleton does not match the melody lines")
            batch["syllable_counts"][row, : len(counts)] = torch.tensor(counts)
    return batch
