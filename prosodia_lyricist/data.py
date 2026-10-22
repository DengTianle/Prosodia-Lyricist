"""Prepared-data manifest validation and XAI source/target padding."""

import json
from pathlib import Path

import torch

from .prepare import SCHEMA_VERSION


def read_manifest(directory):
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] not in (4, SCHEMA_VERSION):
        raise ValueError("Prepared data schema differs; run prosodia-prepare again")
    groups = {}
    for info in manifest["songs"].values():
        previous = groups.setdefault(info["group"], info["split"])
        if previous != info["split"]:
            raise ValueError("A song group appears in more than one data split")
    return manifest


class ProsodyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        batch = {}
        keys = tuple(examples[0])
        if any(set(example) != set(keys) for example in examples):
            raise ValueError("Cannot mix target formats in one batch")
        for key in keys:
            padding = self.pad_token_id if key == "input_ids" else (-100 if "labels" in key else 0)
            width = max(len(example[key]) for example in examples)
            batch[key] = torch.tensor(
                [example[key] + [padding] * (width - len(example[key])) for example in examples],
                dtype=torch.long,
            )
        return batch
