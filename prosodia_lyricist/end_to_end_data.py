"""Join melody inputs and checkpoint-specific lyric targets from the same song."""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .bridge_data import bridge_collate, encode_bridge_song, prosody_pairs
from .data import ProsodyCollator, read_manifest
from .features import SOURCE_KEYS, TARGET_KEYS, encode_example, encode_source
from .ipa import REPO_RULES, REPO_STRESSES
from .melody_checkpoint import file_sha256

logger = logging.getLogger(__name__)
SOURCE_FIELDS = (*SOURCE_KEYS, "slot_indices")
MELODY_FIELDS = (
    "melody_features",
    "melody_attention_mask",
    "note_line_ids",
    "line_counts",
    "song_features",
)
CONDITION_FIELDS = (*SOURCE_FIELDS, *MELODY_FIELDS, "syllable_counts", "remainder_ids")


def encode_slot_source(record, counts, tokenizer, max_syllables, *, decoder_mode="explainable"):
    """Use only title/keywords/counts. Placeholder labels never reach the model as prosody."""
    if len(counts) != len(record["lines"]):
        raise ValueError("Slot counts must match the melody phrases")
    neutral = {
        "title": record.get("title", ""),
        "lines": [
            {
                "keyword": line.get("keyword", ""),
                "syllables": [{"stress": "weak", "length": "short"}] * count,
            }
            for line, count in zip(record["lines"], counts, strict=True)
        ],
    }
    source_encoder = encode_source
    if decoder_mode == "xai":
        from .xai_features import encode_source as source_encoder
    elif decoder_mode != "explainable":
        raise ValueError("Joint decoder_mode must be explainable or xai")
    source = source_encoder(neutral, tokenizer, max_syllables)
    # Bridge queries include one line prefix before each line's slots. Source
    # prompts have variable BPE lengths, so map positions explicitly, never by token ID.
    indices, offset = [], 0
    for count in counts:
        indices.extend(range(offset + 1, offset + count + 1))
        offset += count + 1
    positions = iter(indices)
    source["slot_indices"] = [next(positions) if n else -1 for n in source["length_ids"]]
    source["length_ids"] = [0] * len(source["input_ids"])
    return source


def encode_song(
    record,
    tokenizer,
    *,
    max_syllables,
    encoder_lines_per_window,
    count_source="ipa",
    with_targets=True,
    decoder_mode="explainable",
    strength_labels=("strong", "weak"),
):
    if count_source not in ("ipa", "notes"):
        raise ValueError("count_source must be ipa or notes")
    lines = record["lines"]
    pairs = prosody_pairs(strength_labels)
    counts = [
        len(line["syllables"]) if count_source == "ipa" else len(line["melody"]["midi_pitches"])
        for line in lines
    ]
    example = {
        "melody": encode_bridge_song(lines, encoder_lines_per_window),
        "source": encode_slot_source(
            record, counts, tokenizer, max_syllables, decoder_mode=decoder_mode
        ),
        "syllable_counts": counts,
        "song_id": record.get("song_id"),
        # Evaluation metadata, never collated into model inputs.
        "note_duration_seconds": [
            d for line in lines for d in line["melody"]["note_duration_seconds"]
        ],
    }
    if with_targets:
        # Encode targets from the SAME record, then discard the gold source template.
        if decoder_mode == "xai":
            from .xai_features import encode_targets

            example["targets"] = encode_targets(record, tokenizer)
        else:
            encoded = encode_example(record, tokenizer, max_syllables)
            example["targets"] = {key: encoded[key] for key in TARGET_KEYS}
        labels = []
        for line, count in zip(lines, counts, strict=True):
            labels.append(-100)  # A structural prefix has no prosody label.
            if count == len(line["syllables"]):
                labels.extend(pairs.index((s["stress"], s["length"])) for s in line["syllables"])
            else:
                # Note-count evaluation has no inferred note-to-syllable alignment.
                labels.extend([-100] * count)
        example["bridge_labels"] = labels
        example["reference_lines"] = [
            {"text": line["text"], "syllables": line["syllables"]} for line in lines
        ]
    return example


class EndToEndDataset(Dataset):
    def __init__(
        self,
        directory,
        split,
        tokenizer,
        *,
        max_syllables,
        encoder_lines_per_window,
        max_window_notes,
        max_song_notes,
        max_song_lines,
        max_bridge_length,
        max_source_length,
        max_target_length,
        count_source="ipa",
        limit=None,
        decoder_mode="explainable",
        strength_labels=("strong", "weak"),
        prosody_mode="paper",
    ):
        limits = (
            max_syllables,
            encoder_lines_per_window,
            max_window_notes,
            max_song_notes,
            max_song_lines,
            max_bridge_length,
            max_source_length,
            max_target_length,
        )
        if any(not isinstance(n, int) or n < 1 for n in limits) or (
            limit is not None and (not isinstance(limit, int) or limit < 1)
        ):
            raise ValueError("Dataset limits must be positive integers")
        if count_source not in ("ipa", "notes"):
            raise ValueError("count_source must be ipa or notes")
        manifest = read_manifest(directory)
        data = manifest["config"]
        if data.get("prosody_mode", "paper") != prosody_mode:
            raise ValueError("Prepared IPA rules differ from the model; prepare again")
        if prosody_mode == "actual" and (
            tuple(strength_labels) != REPO_STRESSES
            or manifest.get("pronunciation", {}).get("rules") != REPO_RULES
        ):
            raise ValueError(
                "Repo training requires historical pronunciation and three stress classes"
            )
        if not data.get("include_melody") or data["stress_source"] != "ipa":
            raise ValueError("Prepare with include_melody: true and stress_source: ipa")
        if data["max_syllables"] != max_syllables or (
            data["lines_per_window"] != encoder_lines_per_window
        ):
            raise ValueError("Prepared syllable/window limits differ from the model")
        path = Path(directory) / f"{split}.jsonl"
        if file_sha256(path) != manifest["sha256"][path.name]:
            raise ValueError("Prepared data differs from its manifest; prepare again")
        self.examples, self.songs, self.skipped, self.encoder_windows = [], set(), 0, 0
        self.count_source = count_source
        self.skipped_limits = dict(
            window_notes=0,
            song_notes=0,
            song_lines=0,
            syllables=0,
            bridge_tokens=0,
            source_tokens=0,
            target_tokens=0,
        )
        with path.open(encoding="utf-8") as handle:
            for row in tqdm(handle, desc=f"Encoding joint {split}", unit="song", leave=False):
                record = json.loads(row)
                if manifest["songs"][record["song_id"]]["split"] != split:
                    raise ValueError("Song appears in incorrect split")
                lines = record["lines"]
                if not lines:
                    raise ValueError("Empty melody/lyric song")
                notes = [len(line["melody"]["midi_pitches"]) for line in lines]
                counts = (
                    [len(line["syllables"]) for line in lines] if count_source == "ipa" else notes
                )
                over = dict(
                    window_notes=any(
                        sum(notes[i : i + encoder_lines_per_window]) > max_window_notes
                        for i in range(0, len(lines), encoder_lines_per_window)
                    ),
                    song_notes=sum(notes) > max_song_notes,
                    song_lines=len(lines) > max_song_lines,
                    syllables=any(not 1 <= n <= max_syllables for n in counts),
                    bridge_tokens=sum(counts) + len(counts) > max_bridge_length,
                )
                if not any(over.values()):
                    example = encode_song(
                        record,
                        tokenizer,
                        max_syllables=max_syllables,
                        encoder_lines_per_window=encoder_lines_per_window,
                        count_source=count_source,
                        decoder_mode=decoder_mode,
                        strength_labels=strength_labels,
                    )
                    over.update(
                        source_tokens=len(example["source"]["input_ids"]) > max_source_length,
                        target_tokens=len(example["targets"]["labels"]) > max_target_length,
                    )
                if any(over.values()):
                    self.skipped += 1
                    for reason, exceeded in over.items():
                        self.skipped_limits[reason] += int(exceeded)
                    continue
                self.examples.append(example)
                self.songs.add(record["song_id"])
                self.encoder_windows += (
                    len(lines) + encoder_lines_per_window - 1
                ) // encoder_lines_per_window
                if limit is not None and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(f"No usable joint {split} songs; inspect limits")
        logger.info("Joint %s: %d songs; skipped %s", split, len(self), self.skipped_limits)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class EndToEndCollator:
    def __init__(self, pad_token_id):
        self.prosody = ProsodyCollator(pad_token_id)

    def __call__(self, examples):
        melody = bridge_collate([e["melody"] for e in examples])
        labeled = "targets" in examples[0]
        if any(("targets" in e) != labeled for e in examples):
            raise ValueError("Cannot mix labeled and unlabeled joint examples")
        source = self.prosody(
            [
                {
                    **{key: value for key, value in e["source"].items() if key != "slot_indices"},
                    **e.get("targets", {}),
                }
                for e in examples
            ]
        )
        source["slot_indices"] = torch.full_like(source["input_ids"], -1)
        melody["syllable_counts"] = torch.zeros(
            len(examples),
            int(melody["line_counts"].max()),
            dtype=torch.long,
        )
        if labeled:
            source["bridge_labels"] = torch.full(
                (len(examples), max(len(e["bridge_labels"]) for e in examples)),
                -100,
                dtype=torch.long,
            )
        for row, example in enumerate(examples):
            n = len(example["source"]["slot_indices"])
            source["slot_indices"][row, :n] = torch.tensor(example["source"]["slot_indices"])
            counts = example["syllable_counts"]
            melody["syllable_counts"][row, : len(counts)] = torch.tensor(counts)
            if labeled:
                labels = example["bridge_labels"]
                source["bridge_labels"][row, : len(labels)] = torch.tensor(labels)
        return {**melody, **source}
