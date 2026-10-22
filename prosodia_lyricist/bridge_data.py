"""Window-preserving melody inputs and whole-song IPA template supervision."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import read_manifest
from .ipa import REPO_RULES, REPO_STRESSES
from .melody_checkpoint import file_sha256
from .melody_encoder.encoding import MELODY_FEATURE_DIM, encode_note_sequence
from .runtime import log_stage

logger = logging.getLogger(__name__)

PAD, BOS, SLOT = range(3)
PAIRS = (("strong", "long"), ("strong", "short"), ("weak", "long"), ("weak", "short"))
PAIR_OFFSET = 3
FIRST_LINE = PAIR_OFFSET + len(PAIRS)

# Continuous song coordinates supplement, never replace, the pretrained 177-D inputs.
SONG_FEATURE_SCHEME = "song_pitch_timing_v1"
SONG_TIME_PERIODS = 2.0 ** np.arange(-1, 15, dtype=np.float64)
SONG_SCALARS = (
    "pitch_from_song_start_octaves", "previous_pitch_interval_octaves",
    "log2_duration_over_scale", "log2_positive_ioi_over_scale",
    "has_previous_note", "has_positive_ioi", "log2_scale_seconds",
)
SONG_FEATURE_DIM = len(SONG_SCALARS) + 2 * len(SONG_TIME_PERIODS)


def prosody_pairs(strength_labels=("strong", "weak")):
    if tuple(strength_labels) not in (("strong", "weak"), ("strong", "substrong", "weak")):
        raise ValueError("Unsupported bridge strength vocabulary")
    return tuple((s, n) for s in strength_labels for n in ("long", "short"))


def bridge_vocabulary(lines_per_window, *, pairs=PAIRS):
    return (
        ["pad", "bos", "slot"]
        + [f"{s}_{n}" for s, n in pairs]
        + [f"line_{index}" for index in range(lines_per_window)]
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


def encode_bridge_source(lines):
    """Legacy single window, with exactly the pretrained feature normalization."""
    pitches, onsets, durations, line_ids = melody_arrays(lines)
    return {
        "melody_features": encode_note_sequence(
            pitches, onsets - onsets[0], durations
        ),
        "note_line_ids": line_ids,
        "line_count": len(lines),
    }


def encode_bridge_song(lines, encoder_lines_per_window=2):
    """Gather locally encoded windows in song order and add common musical coordinates.

    Scale is the median positive song IOI (median duration if none exists), not
    an estimated beat. Zero/missing IOIs have explicit indicators and log value
    zero. Fourier time periods are 0.5 .. 16384 scale units. Pitch is in octaves;
    changing the song key or shifting its time origin leaves these features intact.
    """
    if not isinstance(encoder_lines_per_window, int) or encoder_lines_per_window < 1:
        raise ValueError("encoder_lines_per_window must be a positive integer")
    pitches, onsets, durations, line_ids = melody_arrays(lines)
    windows = [
        encode_bridge_source(lines[start : start + encoder_lines_per_window])
        for start in range(0, len(lines), encoder_lines_per_window)
    ]
    intervals = np.diff(onsets, prepend=onsets[0])
    positive = intervals > 0
    scale = np.median(intervals[positive]) if positive.any() else np.median(durations)
    log_ioi = np.zeros_like(intervals)
    log_ioi[positive] = np.log2(intervals[positive] / scale)
    previous = np.arange(len(pitches)) > 0
    scalars = np.stack([
        (pitches - pitches[0]) / 12,
        np.diff(pitches, prepend=pitches[0]) / 12,
        np.log2(durations / scale), log_ioi, previous, positive,
        np.full(len(pitches), np.log2(scale)),
    ], axis=-1)
    phase = (2 * np.pi * ((onsets - onsets[0]) / scale))[:, None] / SONG_TIME_PERIODS
    features = np.concatenate((scalars, np.sin(phase), np.cos(phase)), axis=-1)
    return {
        "melody_features": np.concatenate([w["melody_features"] for w in windows]),
        "song_features": features.astype(np.float32),
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
        self, directory, split, *, lines_per_window, max_notes, limit=None,
        bridge_scope="window", max_song_notes=2048, max_song_lines=256, max_target_length=None,
        strength_labels=None,
    ):
        if lines_per_window < 1 or max_notes < 1:
            raise ValueError("Window sizes must be positive")
        if bridge_scope not in ("window", "song"):
            raise ValueError("Unknown bridge scope")
        if max_song_notes < 1 or max_song_lines < 1 or (
            max_target_length is not None and max_target_length < 1
        ) or (limit is not None and limit < 1):
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
        if data.get("lines_per_window", lines_per_window) != lines_per_window:
            raise ValueError("Prepared encoder window size differs from the bridge")
        self.strength_labels = prepared_labels
        self.pairs = prosody_pairs(prepared_labels)
        path = Path(directory) / f"{split}.jsonl"
        with log_stage("Verifying prepared %s checksum: %s", split, path):
            if file_sha256(path) != manifest["sha256"][path.name]:
                raise ValueError("Prepared data differs from its manifest; prepare again")
        self.examples, self.skipped, self.songs = [], 0, set()
        self.bridge_scope, self.encoder_windows = bridge_scope, 0
        self.skipped_limits = dict(window_notes=0, song_notes=0, song_lines=0, target_tokens=0)
        with log_stage("Encoding bridge %s data", split), path.open(encoding="utf-8") as handle:
            for row in tqdm(handle, desc=f"Encoding {split}", unit="song", leave=False):
                song = json.loads(row)
                if manifest["songs"][song["song_id"]]["split"] != split:
                    raise ValueError("Song appears in incorrect split")
                width = len(song["lines"]) if bridge_scope == "song" else lines_per_window
                if not width:
                    raise ValueError("Empty bridge song")
                song_mean_duration = float(np.mean([
                    duration for line in song["lines"]
                    for duration in line["melody"]["note_duration_seconds"]
                ]))
                for offset in range(0, len(song["lines"]), width):
                    lines = song["lines"][offset : offset + width]
                    note_counts = [len(line["melody"]["midi_pitches"]) for line in lines]
                    over = {
                        "window_notes": any(
                            sum(note_counts[start : start + lines_per_window]) > max_notes
                            for start in range(0, len(lines), lines_per_window)
                        ),
                        "song_notes": bridge_scope == "song" and sum(note_counts) > max_song_notes,
                        "song_lines": bridge_scope == "song" and len(lines) > max_song_lines,
                        "target_tokens": max_target_length is not None and (
                            sum(len(line["syllables"]) + 1 for line in lines) > max_target_length
                        ),
                    }
                    if any(over.values()):
                        self.skipped += 1
                        for reason, exceeded in over.items():
                            self.skipped_limits[reason] += int(exceeded)
                        continue
                    source = (
                        encode_bridge_song(lines, lines_per_window)
                        if bridge_scope == "song" else encode_bridge_source(lines)
                    )
                    self.examples.append(
                        {
                            **source,
                            "labels": encode_bridge_targets(
                                lines, data["max_syllables"], pairs=self.pairs
                            ),
                            # Evaluation metadata; bridge_collate never passes it to the model.
                            "note_duration_seconds": np.asarray([
                                duration for line in lines
                                for duration in line["melody"]["note_duration_seconds"]
                            ], dtype=np.float64),
                            "song_mean_note_duration_seconds": song_mean_duration,
                        }
                    )
                    self.songs.add(song["song_id"])
                    self.encoder_windows += (len(lines) + lines_per_window - 1) // lines_per_window
                    if limit is not None and len(self.examples) >= limit:
                        break
                if limit is not None and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(f"No usable bridge {bridge_scope} examples in {split}; inspect limits")
        array_bytes = sum(
            value.nbytes for example in self.examples for value in example.values()
            if isinstance(value, np.ndarray)
        )
        logger.info(
            "%s: %d %s examples, %d encoder windows, %.1f MiB feature arrays; skipped %s",
            split, len(self.examples), bridge_scope, self.encoder_windows,
            array_bytes / 1024**2, self.skipped_limits,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def bridge_collate(examples, *, pairs=PAIRS):
    if not examples:
        raise ValueError("Cannot collate an empty bridge batch")
    for key in ("song_features", "labels"):
        if any((key in e) != (key in examples[0]) for e in examples):
            raise ValueError("Cannot mix bridge scopes or labeled/unlabeled examples")
    width = max(len(e["note_line_ids"]) for e in examples)
    batch = {
        "melody_features": torch.zeros(len(examples), width, MELODY_FEATURE_DIM),
        "melody_attention_mask": torch.zeros(len(examples), width, dtype=torch.long),
        "note_line_ids": torch.zeros(len(examples), width, dtype=torch.long),
        "line_counts": torch.tensor([e["line_count"] for e in examples]),
    }
    if "song_features" in examples[0]:
        batch["song_features"] = torch.zeros(len(examples), width, SONG_FEATURE_DIM)
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
        if "song_features" in batch:
            batch["song_features"][row, :n] = torch.from_numpy(example["song_features"])
        if "labels" in batch:
            batch["labels"][row, : len(example["labels"])] = torch.tensor(example["labels"])
            counts = [
                len(line["syllables"])
                for line in decode_bridge_tokens(example["labels"], pairs=pairs)
            ]
            if len(counts) != example["line_count"]:
                raise ValueError("Target line skeleton does not match the melody lines")
            batch["syllable_counts"][row, : len(counts)] = torch.tensor(counts)
    return batch
