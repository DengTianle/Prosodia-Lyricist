"""Tokenized JSONL datasets and padding for compound prosody features."""

import hashlib
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .features import SOURCE_KEYS, encode_example
from .prepare import SCHEMA_VERSION

logger = logging.getLogger(__name__)


def read_manifest(directory):
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Prepared data schema differs; run prosodia-prepare again")
    groups = {}
    for info in manifest["songs"].values():
        previous = groups.setdefault(info["group"], info["split"])
        if previous != info["split"]:
            raise ValueError("A song group appears in more than one data split")
    return manifest


class LyricDataset(Dataset):
    def __init__(
        self, directory, split, tokenizer, *, max_source_length, max_target_length, limit=None
    ):
        manifest = read_manifest(directory)
        path = Path(directory) / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
            raise ValueError(f"{path} differs from its preparation manifest; prepare again")
        self.examples = []
        self.skipped = 0
        with path.open(encoding="utf-8") as handle:
            for row in tqdm(handle, desc=f"Tokenizing {split}", unit="line"):
                record = json.loads(row)
                if manifest["songs"][record["song_id"]]["split"] != split:
                    raise ValueError(f"Song in incorrect split: {record['song_id']}")
                example = encode_example(record, tokenizer, manifest["config"]["max_syllables"])
                if (
                    len(example["input_ids"]) > max_source_length
                    or len(example["labels"]) > max_target_length
                ):
                    self.skipped += 1
                    continue
                self.examples.append(example)
                if limit is not None and len(self.examples) >= limit:
                    break
        if not self.examples:
            raise ValueError(
                f"No usable examples in {split}; prepare more songs or raise length limits"
            )
        if self.skipped:
            logger.warning(
                "Skipped %d %s lines exceeding token limits (no truncation)", self.skipped, split
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class ProsodyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        batch = {}
        for key in (*SOURCE_KEYS, "labels"):
            padding = self.pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
            width = max(len(example[key]) for example in examples)
            batch[key] = torch.tensor(
                [example[key] + [padding] * (width - len(example[key])) for example in examples],
                dtype=torch.long,
            )
        return batch
