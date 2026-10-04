"""Evaluate a saved bridge on an integrity-checked prepared held-out split."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bridge_data import BridgeDataset, bridge_collate, decode_bridge_tokens
from .bridge_model import ProsodyBridge
from .bridge_train import evaluate_templates, run_bridge_epoch
from .data import read_manifest
from .evaluation import METRIC_VERSION
from .melody_checkpoint import file_sha256
from .runtime import select_device


def evaluate_bridge(
    checkpoint,
    prepared_dir,
    *,
    split="test",
    batch_size=16,
    device="auto",
    precision="fp32",
    num_workers=0,
    limit=None,
    progress=False,
):
    """Return teacher-forced and free-running IPA-count metrics without training config."""
    if split not in ("valid", "test"):
        raise ValueError("Use the prepared valid or test split")
    if batch_size < 1 or num_workers < 0 or (limit is not None and limit < 1):
        raise ValueError("batch_size/limit must be positive and num_workers nonnegative")
    device = select_device(device)
    if precision not in ("fp32", "bf16") or (
        precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported())
    ):
        raise ValueError("Use fp32 on CPU/MPS; bf16 requires supported CUDA hardware")
    checkpoint, prepared_dir = Path(checkpoint).resolve(), Path(prepared_dir).resolve()
    model = ProsodyBridge.load(checkpoint)
    manifest = read_manifest(prepared_dir)
    dataset = BridgeDataset(
        prepared_dir,
        split,
        **model.dataset_options,
        limit=limit,
    )
    if any(
        len(line["syllables"]) > model.max_syllables
        for example in dataset.examples
        for line in decode_bridge_tokens(example["labels"])
    ):
        raise ValueError("Prepared IPA targets exceed the checkpoint's per-phrase slot limit")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bridge_collate,
        num_workers=num_workers,
    )
    model.to(device)
    with tqdm(loader, desc="Teacher-forced evaluation", unit="batch", disable=not progress) as bar:
        teacher = run_bridge_epoch(model, bar, device, precision=precision)
    with tqdm(loader, desc="IPA-count generation", unit="batch", disable=not progress) as bar:
        generated = evaluate_templates(model, bar, device, precision=precision)
    generated["accuracy"] = {
        "strength": generated.pop("strength_accuracy"),
        "length": generated.pop("length_accuracy"),
        "combined": generated.pop("pair_accuracy"),
    }
    return {
        "schema_version": 3,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": file_sha256(checkpoint / "bridge_weights.pt"),
        "data": {
            "prepared_dir": str(prepared_dir),
            "split": split,
            "split_sha256": manifest["sha256"][f"{split}.jsonl"],
            "reference_labels": "prepared IPA stress/vowel length",
            "example_scope": model.bridge_scope,
            "limit_examples": limit,
            "examples": len(dataset),
            "songs": len(dataset.songs),
            "encoder_windows": dataset.encoder_windows,
            "skipped_examples": dataset.skipped,
            "skipped_limits": dataset.skipped_limits,
            **({
                "limit_windows": limit, "windows": len(dataset),
                "skipped_note_limit_windows": dataset.skipped_limits["window_notes"],
            } if model.bridge_scope == "window" else {}),
        },
        "runtime": {
            "device": str(device),
            "precision": precision,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model_limits": {
            "lines_per_window": model.lines_per_window,
            "max_notes": model.max_notes,
            "max_syllables": model.max_syllables,
            "max_target_length": model.max_target_length,
            **({
                "encoder_lines_per_window": model.encoder_lines_per_window,
                "max_window_notes": model.max_notes,
                "max_song_notes": model.max_song_notes, "max_song_lines": model.max_song_lines,
            } if model.bridge_scope == "song" else {}),
        },
        "bridge_scope": model.bridge_scope,
        "output_design": model.output_design,
        "target_scheme": model.target_scheme,
        "teacher_forced": {
            "loss": teacher["loss"],
            "loss_components": teacher["components"],
            "accuracy": {
                "strength": teacher["teacher_forced_strength_accuracy"],
                "length": teacher["teacher_forced_length_accuracy"],
                "combined": teacher["teacher_forced_pair_accuracy"],
            },
            "slots": teacher["tokens"],
            "phrases": teacher["phrases"],
            "batches": teacher["batches"],
            "slot_count_source": "ipa",
        },
        "generation": generated,
        "metric_definitions": {
            "loss": (
                "Sum of mean binary strength and length cross-entropies per IPA slot; "
                "prefixes/padding excluded."
                if model.output_design == "separate"
                else "Mean joint pair cross-entropy per IPA slot; prefixes/padding excluded."
            ),
            "teacher_forced_accuracy": "Micro accuracy over IPA slots with gold prior labels.",
            "generation_accuracy": (
                "Greedy free-running labels with reference IPA counts, without gold prior labels. "
                "Micro accuracy over IPA slots, compared by phrase/slot order. "
                "MIDI inference still uses note counts."
            ),
            "combined_accuracy": "Both strength and length must match at the same slot.",
            "prosody_bleu": (
                f"{METRIC_VERSION}: phrase-mean unsmoothed BLEU-4 of (strength,length) "
                "symbols, same as final lyric evaluation. A phrase with either sequence "
                "shorter than four slots scores zero."
            ),
            "coverage": (
                "Teacher forcing excludes entire examples above checkpoint source/target "
                "limits. A song example is never partially retained. Generation also excludes "
                "examples whose IPA counts exceed the per-phrase slot or total target limit. Count "
                "diagnostics cover all teacher-forced phrases. Skips are reported separately. "
                "With --limit, counts cover only the scanned subset."
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, help="First N retained songs (windows for legacy checkpoints)"
    )
    parser.add_argument("--output", type=Path, help="Also save the JSON summary to this path")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_bridge(
        args.checkpoint,
        args.prepared_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        num_workers=args.num_workers,
        limit=args.limit,
        progress=not args.no_progress,
    )
    summary = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return report


if __name__ == "__main__":
    main()
