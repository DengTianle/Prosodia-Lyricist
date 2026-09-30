"""Supervised training of a contrastive melody encoder's IPA-template bridge."""

import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bridge_data import (
    FIRST_LINE,
    PAIR_OFFSET,
    BridgeDataset,
    bridge_collate,
    decode_bridge_tokens,
)
from .bridge_model import ProsodyBridge
from .data import read_manifest
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .melody_data import read_melody_manifest
from .runtime import seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)


def run_bridge_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    scheduler=None,
    accumulation_steps=1,
    precision="fp32",
    gradient_clip=1.0,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    totals = {"prosody": 0.0, "counts": 0.0}
    tokens, phrases, correct, batches, updates = 0, 0, 0, 0, 0
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            group_tokens = sum(int(batch["syllable_counts"].sum()) for batch in group)
            group_phrases = sum(int(batch["line_counts"].sum()) for batch in group)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {key: value.to(device) for key, value in batch.items()}
                active = batch["labels"].ge(PAIR_OFFSET) & batch["labels"].lt(FIRST_LINE)
                count = int(active.sum())
                line_count = int(batch["line_counts"].sum())
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if precision == "bf16"
                    else (nullcontext())
                ):
                    output = model(**batch)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite bridge loss")
                if training:
                    # Each objective has its own valid-target denominator, including tails.
                    (
                        output.loss_components["prosody"] * (count / group_tokens)
                        + output.loss_components["counts"] * (line_count / group_phrases)
                    ).backward()
                correct += int(
                    ((output.logits.argmax(-1) + PAIR_OFFSET).eq(batch["labels"]) & active).sum()
                )
                totals["prosody"] += output.loss_components["prosody"].item() * count
                totals["counts"] += output.loss_components["counts"].item() * line_count
                tokens += count
                phrases += line_count
                batches += 1
            if training:
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                scheduler.step()
                updates += 1
    if not tokens:
        raise ValueError("No template targets")
    components = {"prosody": totals["prosody"] / tokens, "counts": totals["counts"] / phrases}
    return {
        "loss": sum(components.values()),
        "components": components,
        "teacher_forced_pair_accuracy": correct / tokens,
        "tokens": tokens,
        "phrases": phrases,
        "batches": batches,
        "optimizer_steps": updates,
    }


@torch.inference_mode()
def evaluate_templates(model, loader, device, *, max_batches=None):
    """Free-running accuracy, including the learned number of syllables per phrase."""
    model.eval()
    phrases, exact, count_matches, count_error = 0, 0, 0, 0
    for batch in islice(loader, max_batches):
        labels = batch.pop("labels")
        # Automatic validation must not receive the ground-truth skeleton lengths.
        batch.pop("syllable_counts")
        batch = {key: value.to(device) for key, value in batch.items()}
        result = model.generate(**batch)
        for predicted, truth in zip(result.sequences.tolist(), labels.tolist(), strict=True):
            expected = decode_bridge_tokens([token for token in truth if token != -100])
            actual = decode_bridge_tokens(predicted)
            for a, b in zip(actual, expected, strict=True):
                phrases += 1
                exact += a == b
                count_matches += len(a["syllables"]) == len(b["syllables"])
                count_error += abs(len(a["syllables"]) - len(b["syllables"]))
    return {
        "phrases": phrases,
        "exact_template_accuracy": exact / phrases,
        "syllable_count_accuracy": count_matches / phrases,
        "syllable_count_mae": count_error / phrases,
    }


def train_bridge(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    for key in ("epochs", "batch_size", "patience", "gradient_accumulation_steps"):
        if not isinstance(settings.get(key, 1), int) or settings.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "melody_learning_rate"):
        if not 0 < settings[key] < math.inf:
            raise ValueError(f"{key} must be positive and finite")
    unfreeze = settings.get("melody_unfreeze_epoch", 1)
    if unfreeze is not None and (not isinstance(unfreeze, int) or unfreeze < 0):
        raise ValueError("melody_unfreeze_epoch must be null or a nonnegative integer")
    clip = settings.get("gradient_clip", 1.0)
    if clip is not None and not 0 < clip < math.inf:
        raise ValueError("gradient_clip must be positive and finite")
    seed_everything(settings["seed"])
    device = select_device(settings["device"])
    precision = "fp32" if smoke_test else settings.get("precision", "fp32")
    if precision not in ("fp32", "bf16") or (
        precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported())
    ):
        raise ValueError("Use fp32 on CPU/MPS; bf16 requires supported CUDA hardware")
    manifest = read_manifest(data["prepared_dir"])
    if manifest["config"] != data:
        raise ValueError("Data configuration differs from preparation; prepare again")
    if data.get("song_ids_file") and (
        file_sha256(data["song_ids_file"]) != manifest["selection"]["sha256"]
    ):
        raise ValueError("Song allowlist changed; prepare again")
    checkpoint = cfg.get("melody_checkpoint")
    tower = None
    if checkpoint:
        audit = manifest.get("pretraining_split_audit")
        if (
            not data.get("pretraining_manifest")
            or not audit
            or (audit["sha256"] != file_sha256(data["pretraining_manifest"]))
        ):
            raise ValueError("Prepare with the unchanged melody pretraining manifest")
        tower, melody_config, provenance = load_contrastive_melody(checkpoint)
        if provenance["pretraining_train_split"] != "train" or (
            provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported contrastive pretraining split semantics")
        rows, _ = read_melody_manifest(data["pretraining_manifest"], require_lyrics=False)
        sizes = {int(row["line_count"]) for row in rows if row.get("line_count")}
        if sizes and sizes != {data["lines_per_window"]}:
            raise ValueError("lines_per_window must match the contrastive pretraining manifest")
    elif smoke_test:
        melody_config = dict(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
            projection_dim=None,
            max_length=cfg.get("max_notes", 512),
        )
        provenance = {"random_smoke_test_tower": True}
    else:
        raise ValueError("Set model.melody_checkpoint to the contrastive two-pool checkpoint")
    decoder = cfg.get("bridge_decoder", {})
    if smoke_test:
        decoder = dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0.0)
    model = ProsodyBridge(
        melody_config,
        max_syllables=data["max_syllables"],
        lines_per_window=data["lines_per_window"],
        max_notes=cfg.get("max_notes"),
        provenance=provenance,
        **decoder,
    )
    if tower is not None:
        model.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
        del tower
    max_notes = model.max_notes
    datasets = {
        split: BridgeDataset(
            data["prepared_dir"],
            split,
            lines_per_window=data["lines_per_window"],
            max_notes=max_notes,
            limit=8 if smoke_test else None,
        )
        for split in ("train", "valid")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=min(2, settings["batch_size"]) if smoke_test else settings["batch_size"],
            collate_fn=bridge_collate,
            shuffle=split == "train",
            generator=torch.Generator().manual_seed(settings["seed"]),
            num_workers=0 if smoke_test else settings.get("num_workers", 0),
        )
        for split, ds in datasets.items()
    }
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("melody_encoder.")
                ],
                "lr": settings["learning_rate"],
            },
            {"params": model.melody_encoder.parameters(), "lr": settings["melody_learning_rate"]},
        ],
        betas=tuple(settings.get("adam_betas", (0.9, 0.98))),
        eps=settings.get("adam_epsilon", 1e-5),
        weight_decay=settings.get("weight_decay", 0.0),
    )
    epochs, max_batches = (1, 2) if smoke_test else (settings["epochs"], None)
    accumulation = settings.get("gradient_accumulation_steps", 1)
    steps = math.ceil(
        min(len(loaders["train"]), max_batches or len(loaders["train"])) / accumulation
    )
    scheduler = build_scheduler(optimizer, settings, epochs * steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"smoke-{stamp}" if smoke_test else stamp
    output = Path(output_dir) if output_dir else Path(settings["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "conditioning": "bridge",
        "bridge_target_scheme": "line_skeleton_v2",
        "config": config,
        "smoke_test": smoke_test,
        "data_schema_version": manifest["schema_version"],
        "pronunciation": manifest["pronunciation"],
        "data_sha256": manifest["sha256"],
        "pretraining_split_audit": manifest.get("pretraining_split_audit"),
        "melody_provenance": provenance,
        "precision": precision,
        "total_steps": steps * epochs,
        "windows": {s: len(ds) for s, ds in datasets.items()},
        "songs": {s: len(ds.songs) for s, ds in datasets.items()},
        "skipped_note_limits": {s: ds.skipped for s, ds in datasets.items()},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    best, stale = math.inf, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in tqdm(range(epochs), desc="Bridge epochs"):
            model.set_melody_trainable(
                bool(provenance.get("random_smoke_test_tower"))
                or (unfreeze is not None and epoch >= unfreeze)
            )
            training = run_bridge_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation_steps=accumulation,
                precision=precision,
                gradient_clip=clip,
                max_batches=max_batches,
            )
            validation = run_bridge_epoch(
                model,
                loaders["valid"],
                device,
                precision=precision,
                max_batches=max_batches,
            )
            validation["generation"] = evaluate_templates(
                model,
                loaders["valid"],
                device,
                max_batches=max_batches,
            )
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": training,
                        "valid": validation,
                        "learning_rates": scheduler.get_last_lr(),
                        "melody_frozen": model.melody_frozen,
                    }
                )
                + "\n"
            )
            handle.flush()
            logger.info(
                "Epoch %d train %.4f valid %.4f", epoch + 1, training["loss"], validation["loss"]
            )
            if validation["loss"] < best:
                best, stale = validation["loss"], 0
                model.save(output / "best")
                (output / "best" / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    logger.info("Best bridge checkpoint: %s", output / "best")
    return output
