"""Globally encoded melody inputs and whole-song IPA template supervision."""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .bridge_alignment import ALIGNMENT_POLICIES, aligned_note_indices, recover_note_alignment
from .data import read_manifest
from .ipa import REPO_RULES, REPO_STRESSES
from .melody_checkpoint import file_sha256
from .melody_encoder.encoding import MELODY_FEATURE_DIM, encode_note_sequence
from .meter import METER_FEATURE_DIM, encode_meter
from .runtime import log_stage

logger = logging.getLogger(__name__)

PAD, SLOT = range(2)
PAIRS = (("strong", "long"), ("strong", "short"), ("weak", "long"), ("weak", "short"))
PAIR_OFFSET = 2


def prosody_pairs(strength_labels=("strong", "weak")):
    if tuple(strength_labels) not in (("strong", "weak"), ("strong", "substrong", "weak")):
        raise ValueError("Unsupported bridge strength vocabulary")
    return tuple((s, n) for s in strength_labels for n in ("long", "short"))


def bridge_vocabulary(max_lines, *, pairs=PAIRS):
    return (
        ["pad", "slot"]
        + [f"{s}_{n}" for s, n in pairs]
        + [f"line_{index}" for index in range(max_lines)]
    )


def melody_arrays(lines):
    """Read only source note arrays; line numbering follows the supplied sequence."""
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
    pitches, onsets, durations = (
        np.asarray(values, dtype=np.float64) for values in (pitches, onsets, durations)
    )
    if (
        not all(np.isfinite(values).all() for values in (pitches, onsets, durations))
        or not np.equal(pitches, np.trunc(pitches)).all()
        or (np.diff(onsets) < 0).any()
        or (durations <= 0).any()
    ):
        raise ValueError("Invalid or unordered melody pitch/timing arrays")
    return pitches, onsets, durations, line_ids


def encode_bridge_song(lines):
    """Encode the entire song once with the original 177-D representation.

    Pitch reference, log-duration bins and IOI bins are shared by all phrases.
    Phrase IDs and optional observed meter are separate conditioning inputs.
    """
    pitches, onsets, durations, line_ids = melody_arrays(lines)
    meter, observed = encode_meter(lines)
    return {
        "melody_features": encode_note_sequence(pitches, onsets - onsets[0], durations),
        "meter_features": meter,
        "meter_mask": observed,
        "note_line_ids": line_ids,
        "line_count": len(lines),
    }


def encode_bridge_targets(lines, max_syllables, *, pairs=PAIRS):
    """DALI line membership fixes the prefixes; IPA supplies syllables within each line."""
    labels = []
    for index, line in enumerate(lines):
        if not 1 <= len(line["syllables"]) <= max_syllables:
            raise ValueError("IPA template exceeds the per-phrase syllable limit")
        labels.append(PAIR_OFFSET + len(pairs) + index)
        for syllable in line["syllables"]:
            pair = syllable["stress"], syllable["length"]
            if pair not in pairs:
                raise ValueError("Bridge targets differ from the IPA stress/length vocabulary")
            labels.append(PAIR_OFFSET + pairs.index(pair))
    return labels


def decode_bridge_tokens(tokens, *, pairs=PAIRS):
    lines, syllables, padded = [], None, False
    first_line = PAIR_OFFSET + len(pairs)
    for token in tokens:
        if token == PAD:
            padded = True
            continue
        if padded:
            raise ValueError("Template padding must be trailing")
        if token >= first_line:
            if token != first_line + len(lines):
                raise ValueError("Template line IDs must be consecutive, starting at line_0")
            if syllables == []:
                raise ValueError("Empty template phrase")
            syllables = []
            lines.append({"syllables": syllables})
        elif PAIR_OFFSET <= token < first_line and syllables is not None:
            stress, length = pairs[token - PAIR_OFFSET]
            syllables.append({"stress": stress, "length": length})
        else:
            raise ValueError("Expected a line prefix followed by prosody pairs")
    if not syllables:
        raise ValueError("Incomplete predicted template")
    return lines


class BridgeDataset(Dataset):
    def __init__(
        self,
        directory,
        split,
        *,
        limit=None,
        max_song_notes=2048,
        max_song_lines=256,
        max_target_length=4096,
        strength_labels=None,
        alignment_policy="strict",
        dali_dir=None,
    ):
        if alignment_policy is not None and alignment_policy not in ALIGNMENT_POLICIES:
            raise ValueError("alignment_policy must be strict or prefix")
        if (
            max_song_notes < 1
            or max_song_lines < 1
            or (max_target_length is not None and max_target_length < 1)
            or (limit is not None and limit < 1)
        ):
            raise ValueError("Song/target limits must be positive")
        manifest = read_manifest(directory)
        data = manifest["config"]
        if not data.get("include_melody") or data["stress_source"] != "ipa":
            raise ValueError("Prepare with include_melody: true and stress_source: ipa")
        prepared_labels = (
            REPO_STRESSES if data.get("prosody_mode", "paper") == "actual" else ("strong", "weak")
        )
        if strength_labels is not None and tuple(strength_labels) != prepared_labels:
            raise ValueError("Prepared IPA rules differ from the bridge's stress/length vocabulary")
        if prepared_labels == REPO_STRESSES and (
            manifest.get("pronunciation", {}).get("rules") != REPO_RULES
        ):
            raise ValueError("Prepared IPA rules must record XAI historical pronunciation")
        self.strength_labels = prepared_labels
        self.pairs = prosody_pairs(prepared_labels)
        path = Path(directory) / f"{split}.jsonl"
        with log_stage("Verifying prepared %s checksum: %s", split, path):
            if file_sha256(path) != manifest["sha256"][path.name]:
                raise ValueError("Prepared data differs from its manifest; prepare again")
        self.examples, self.skipped, self.songs = [], 0, set()
        self.alignment_counts, self.raw_alignment_sha256 = Counter(), {}
        self.skipped_alignment = 0
        self.skipped_limits = dict(song_notes=0, song_lines=0, target_tokens=0)
        with log_stage("Encoding bridge %s data", split), path.open(encoding="utf-8") as handle:
            for row in tqdm(handle, desc=f"Encoding {split}", unit="song", leave=False):
                song = json.loads(row)
                if manifest["songs"][song["song_id"]]["split"] != split:
                    raise ValueError("Song appears in incorrect split")
                lines = song["lines"]
                if not lines:
                    raise ValueError("Empty bridge song")
                note_counts = [len(line["melody"]["midi_pitches"]) for line in lines]
                over = {
                    "song_notes": sum(note_counts) > max_song_notes,
                    "song_lines": len(lines) > max_song_lines,
                    "target_tokens": max_target_length is not None
                    and (sum(len(line["syllables"]) + 1 for line in lines) > max_target_length),
                }
                if any(over.values()):
                    self.skipped += 1
                    for reason, exceeded in over.items():
                        self.skipped_limits[reason] += int(exceeded)
                    continue
                alignment = {}
                if alignment_policy is not None:
                    if any("note_alignment" not in line for line in lines):
                        if data.get("source") == "lakh":
                            raise ValueError("Missing prepared Lakh note alignment; prepare again")
                        checksum = recover_note_alignment(song, dali_dir or data.get("dali_dir"))
                        self.raw_alignment_sha256[song["song_id"]] = checksum
                    indices, counts = aligned_note_indices(lines, alignment_policy)
                    self.alignment_counts.update(counts)
                    if not counts["supervised_slots"]:
                        self.skipped += 1
                        self.skipped_alignment += 1
                        continue
                    alignment["target_note_indices"] = indices
                durations = np.asarray(
                    [
                        duration
                        for line in lines
                        for duration in line["melody"]["note_duration_seconds"]
                    ],
                    dtype=np.float64,
                )
                self.examples.append(
                    {
                        **encode_bridge_song(lines),
                        **alignment,
                        "labels": encode_bridge_targets(
                            lines, data["max_syllables"], pairs=self.pairs
                        ),
                        # Evaluation metadata; never passed to the model.
                        "note_duration_seconds": durations,
                        "song_mean_note_duration_seconds": float(durations.mean()),
                    }
                )
                self.songs.add(song["song_id"])
                if limit is not None and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(f"No usable bridge songs in {split}; inspect limits")
        array_bytes = sum(
            value.nbytes
            for example in self.examples
            for value in example.values()
            if isinstance(value, np.ndarray)
        )
        logger.info(
            "%s: %d songs, %.1f MiB feature arrays; skipped %s",
            split,
            len(self.examples),
            array_bytes / 1024**2,
            self.skipped_limits,
        )
        if alignment_policy is not None:
            logger.info(
                "%s note alignment (%s): %s; %d examples without targets skipped",
                split,
                alignment_policy,
                dict(self.alignment_counts),
                self.skipped_alignment,
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def bridge_collate(examples, *, pairs=PAIRS):
    if not examples:
        raise ValueError("Cannot collate an empty bridge batch")
    for key in ("labels", "target_note_indices"):
        if any((key in e) != (key in examples[0]) for e in examples):
            raise ValueError("Cannot mix labeled/unlabeled or aligned/unaligned examples")
    width = max(len(e["note_line_ids"]) for e in examples)
    batch = {
        "melody_features": torch.zeros(len(examples), width, MELODY_FEATURE_DIM),
        "melody_attention_mask": torch.zeros(len(examples), width, dtype=torch.long),
        "note_line_ids": torch.zeros(len(examples), width, dtype=torch.long),
        "line_counts": torch.tensor([e["line_count"] for e in examples]),
        "meter_features": torch.zeros(len(examples), width, METER_FEATURE_DIM),
        "meter_mask": torch.zeros(len(examples), width, dtype=torch.bool),
    }
    if "labels" in examples[0]:
        batch["labels"] = torch.full(
            (len(examples), max(len(e["labels"]) for e in examples)), -100, dtype=torch.long
        )
        batch["syllable_counts"] = torch.zeros(
            len(examples), int(batch["line_counts"].max()), dtype=torch.long
        )
        if "target_note_indices" in examples[0]:
            batch["target_note_indices"] = torch.full_like(batch["labels"], -1)
    for row, example in enumerate(examples):
        n = len(example["note_line_ids"])
        batch["melody_features"][row, :n] = torch.from_numpy(example["melody_features"])
        batch["melody_attention_mask"][row, :n] = 1
        batch["note_line_ids"][row, :n] = torch.tensor(example["note_line_ids"])
        if "meter_features" in example:
            batch["meter_features"][row, :n] = torch.from_numpy(example["meter_features"])
            batch["meter_mask"][row, :n] = torch.from_numpy(example["meter_mask"])
        if "labels" in batch:
            batch["labels"][row, : len(example["labels"])] = torch.tensor(example["labels"])
            counts = [
                len(line["syllables"])
                for line in decode_bridge_tokens(example["labels"], pairs=pairs)
            ]
            if len(counts) != example["line_count"]:
                raise ValueError("Target line skeleton does not match the melody lines")
            batch["syllable_counts"][row, : len(counts)] = torch.tensor(counts)
            if "target_note_indices" in batch:
                indices = example["target_note_indices"]
                if len(indices) != len(example["labels"]):
                    raise ValueError("Note alignment must match the IPA target skeleton")
                batch["target_note_indices"][row, : len(indices)] = torch.tensor(indices)
    return batch
