"""Supervised training of a contrastive melody encoder's IPA-template bridge."""

import json
import logging
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import islice, zip_longest
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
from .evaluation import prosody_bleu
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .melody_data import read_melody_manifest
from .runtime import log_stage, seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)


def _random_melody_config(config):
    """Resolve an explicit tower architecture without opening a pretrained checkpoint."""
    required = {"d_model", "num_layers", "num_heads", "dim_feedforward"}
    allowed = required | {"dropout", "pooling", "max_length"}
    if not isinstance(config, dict) or not required.issubset(config) or config.keys() - allowed:
        raise ValueError(
            "model.random_melody_encoder requires d_model, num_layers, num_heads, "
            "dim_feedforward; optional keys are dropout, pooling, max_length"
        )
    config = {"dropout": 0.1, "pooling": "cls", "max_length": 4096, **config}
    for name in (*sorted(required), "max_length"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"model.random_melody_encoder.{name} must be a positive integer")
    if config["d_model"] % config["num_heads"]:
        raise ValueError("model.random_melody_encoder.d_model must be divisible by num_heads")
    if not isinstance(config["dropout"], (int, float)) or not 0 <= config["dropout"] < 1:
        raise ValueError("model.random_melody_encoder.dropout must be in [0, 1)")
    if config["pooling"] not in ("cls", "mean"):
        raise ValueError("model.random_melody_encoder.pooling must be cls or mean")
    return {**config, "projection_dim": None}


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
    history_mask_probability=0.0,
):
    training = optimizer is not None
    model.train(training)
    forward_options = (
        {"history_mask_probability": history_mask_probability}
        if training and history_mask_probability else {}
    )
    total_loss = 0.0
    component_totals = {}
    tokens, phrases, correct, batches, updates = 0, 0, 0, 0, 0
    strength_correct, length_correct = 0, 0
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            group_tokens = sum(int(batch["syllable_counts"].sum()) for batch in group)
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
                    output = model(**batch, **forward_options)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite bridge loss")
                if training:
                    # Weight by valid prosody slots, including partial accumulation groups.
                    (output.loss * (count / group_tokens)).backward()
                if model.output_design == "separate":
                    predicted = (
                        2 * output.strength_logits.argmax(-1) + output.length_logits.argmax(-1)
                    )[active]
                else:
                    predicted = output.logits.argmax(-1)[active]
                expected = batch["labels"][active] - PAIR_OFFSET
                correct += int(predicted.eq(expected).sum())
                # PAIRS groups strong/weak by quotient and long/short by remainder.
                strength_correct += int((predicted // 2).eq(expected // 2).sum())
                length_correct += int((predicted % 2).eq(expected % 2).sum())
                total_loss += output.loss.item() * count
                for name, value in output.loss_components.items():
                    component_totals[name] = component_totals.get(name, 0.0) + value.item() * count
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
    loss = total_loss / tokens
    return {
        "loss": loss,
        "components": {name: value / tokens for name, value in component_totals.items()},
        "teacher_forced_strength_accuracy": strength_correct / tokens,
        "teacher_forced_length_accuracy": length_correct / tokens,
        "teacher_forced_pair_accuracy": correct / tokens,
        "tokens": tokens,
        "phrases": phrases,
        "batches": batches,
        "optimizer_steps": updates,
        "history_mask_probability": history_mask_probability if training else 0.0,
    }


@torch.inference_mode()
def evaluate_templates(model, loader, device, *, max_batches=None, precision="fp32"):
    """Free-running labels with IPA counts; note/IPA differences remain diagnostics."""
    model.eval()
    phrases, exact, count_matches, count_error, count_phrases = 0, 0, 0, 0, 0
    skipped_examples, skipped_phrases = 0, 0
    skipped_target_examples, skipped_target_phrases = 0, 0
    bleu_sum, short_phrases = 0.0, 0
    slots, strength_correct, length_correct, pair_correct = 0, 0, 0, 0
    for batch in islice(loader, max_batches):
        batch = dict(batch)
        labels = batch.pop("labels")
        # Supply the reference skeleton, but never feed gold prosody labels to generation.
        counts = batch["syllable_counts"]
        note_counts = model.note_counts(batch["note_line_ids"], batch["line_counts"])
        active = counts.gt(0)
        count_matches += int((note_counts.eq(counts) & active).sum())
        count_error += int((note_counts - counts).abs()[active].sum())
        count_phrases += int(active.sum())
        eligible = counts.le(model.max_syllables).all(-1)
        target_eligible = (counts.sum(-1) + batch["line_counts"]).le(model.max_target_length)
        skipped_examples += int((~eligible).sum())
        skipped_phrases += int(batch["line_counts"][~eligible].sum())
        # Report total-target exclusions separately; slot-limit failures take precedence.
        skipped_target_examples += int((eligible & ~target_eligible).sum())
        skipped_target_phrases += int(batch["line_counts"][eligible & ~target_eligible].sum())
        eligible &= target_eligible
        if not eligible.any():
            continue
        labels = labels[eligible]
        batch = {key: value[eligible] for key, value in batch.items()}
        # A filtered batch can have fewer phrases than its original padding width.
        batch["syllable_counts"] = batch["syllable_counts"][:, :int(batch["line_counts"].max())]
        batch = {key: value.to(device) for key, value in batch.items()}
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        ):
            result = model.generate(**batch)
        for predicted, truth in zip(result.sequences.tolist(), labels.tolist(), strict=True):
            expected = decode_bridge_tokens([token for token in truth if token != -100])
            actual = decode_bridge_tokens(predicted)
            for a, b in zip(actual, expected, strict=True):
                phrases += 1
                exact += a == b
                hypothesis, reference = a["syllables"], b["syllables"]
                bleu_sum += prosody_bleu(reference, hypothesis)
                short_phrases += min(len(reference), len(hypothesis)) < 4
                # Compare by phrase/slot order; missing and extra slots are incorrect.
                slots += max(len(hypothesis), len(reference))
                for predicted_slot, expected_slot in zip_longest(hypothesis, reference):
                    if predicted_slot is not None and expected_slot is not None:
                        strength_correct += predicted_slot["stress"] == expected_slot["stress"]
                        length_correct += predicted_slot["length"] == expected_slot["length"]
                        pair_correct += predicted_slot == expected_slot
    return {
        "phrases": phrases,
        "slot_count_source": "ipa",
        "strength_accuracy": strength_correct / slots if slots else None,
        "length_accuracy": length_correct / slots if slots else None,
        "pair_accuracy": pair_correct / slots if slots else None,
        "compared_slots": slots,
        "exact_template_accuracy": exact / phrases if phrases else None,
        "prosody_bleu": bleu_sum / phrases if phrases else None,
        "bleu_short_phrases": short_phrases,
        "count_diagnostic_phrases": count_phrases,
        "note_ipa_count_match_rate": count_matches / count_phrases,
        "note_ipa_count_mae": count_error / count_phrases,
        f"skipped_slot_limit_{'songs' if model.bridge_scope == 'song' else 'windows'}": (
            skipped_examples
        ),
        "skipped_target_limit_examples": skipped_target_examples,
        "skipped_target_limit_phrases": skipped_target_phrases,
        "example_scope": model.bridge_scope,
        "skipped_slot_limit_phrases": skipped_phrases,
    }


def train_bridge(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    initialization = cfg.get("melody_initialization", "pretrained")
    if initialization not in ("pretrained", "random"):
        raise ValueError("model.melody_initialization must be pretrained or random")
    random_config = (
        _random_melody_config(cfg.get("random_melody_encoder"))
        if initialization == "random" else None
    )
    logger.info(
        "Starting bridge training: pid=%d torch=%s source=%s batch_size=%s workers=%s",
        os.getpid(), torch.__version__, __file__, settings["batch_size"],
        0 if smoke_test else settings.get("num_workers", 0),
    )
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
    history_mask_probability = settings.get("history_mask_probability", 0.0)
    if not isinstance(history_mask_probability, (int, float)) or not (
        0 <= history_mask_probability <= 1
    ):
        raise ValueError("history_mask_probability must be a number between 0 and 1")
    logger.info("Training label-history masking probability: %.3f", history_mask_probability)
    with log_stage("Initializing bridge device and random seed"):
        seed_everything(settings["seed"])
        device = select_device(settings["device"])
        precision = "fp32" if smoke_test else settings.get("precision", "fp32")
        if precision not in ("fp32", "bf16") or (
            precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported())
        ):
            raise ValueError("Use fp32 on CPU/MPS; bf16 requires supported CUDA hardware")
    logger.info("Using %s with %s precision", device, precision)
    with log_stage("Reading prepared manifest: %s", data["prepared_dir"]):
        manifest = read_manifest(data["prepared_dir"])
    if manifest["config"] != data:
        raise ValueError("Data configuration differs from preparation; prepare again")
    if data.get("song_ids_file") and (
        file_sha256(data["song_ids_file"]) != manifest["selection"]["sha256"]
    ):
        raise ValueError("Song allowlist changed; prepare again")
    checkpoint = cfg.get("melody_checkpoint")
    encoder_lines = cfg.get("encoder_lines_per_window", data["lines_per_window"])
    if encoder_lines != data["lines_per_window"]:
        raise ValueError("encoder_lines_per_window must match the prepared pretraining windows")
    max_window_notes = cfg.get("max_window_notes", cfg.get("max_notes"))
    if cfg.get("max_notes") is not None and max_window_notes != cfg["max_notes"]:
        raise ValueError("Conflicting max_notes and max_window_notes settings")
    tower = None
    # Preserve prepared-data audits in either mode; random initialization does not
    # require pretraining provenance when the data was prepared without it.
    if data.get("pretraining_manifest") or (initialization == "pretrained" and checkpoint):
        audit = manifest.get("pretraining_split_audit")
        with log_stage(
            "Verifying pretraining manifest checksum: %s", data.get("pretraining_manifest")
        ):
            if (
                not data.get("pretraining_manifest")
                or not audit
                or (audit["sha256"] != file_sha256(data["pretraining_manifest"]))
            ):
                raise ValueError("Prepare with the unchanged melody pretraining manifest")
        with log_stage("Auditing pretraining CSV: %s", data["pretraining_manifest"]):
            rows, upstream_songs = read_melody_manifest(
                data["pretraining_manifest"], require_lyrics=False
            )
            sizes = {int(row["line_count"]) for row in rows if row.get("line_count")}
            if sizes and sizes != {encoder_lines}:
                raise ValueError("lines_per_window must match the contrastive pretraining manifest")
            # These potentially large Python containers are not used during training.
            del rows, upstream_songs
    if initialization == "random":
        melody_config = random_config
        provenance = {
            "initialization": "random", "weights_loaded": False,
            "checkpoint": None, "sha256": None,
            "architecture_source": "model.random_melody_encoder",
        }
    elif checkpoint:
        tower, melody_config, provenance = load_contrastive_melody(checkpoint)
        if provenance["pretraining_train_split"] != "train" or (
            provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported contrastive pretraining split semantics")
        provenance = {**provenance, "initialization": "pretrained", "weights_loaded": True}
    elif smoke_test:
        melody_config = dict(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
            projection_dim=None,
            max_length=max_window_notes or 512,
        )
        provenance = {
            "random_smoke_test_tower": True, "initialization": "random", "weights_loaded": False,
        }
    else:
        raise ValueError(
            "Set model.melody_checkpoint to the contrastive two-pool checkpoint, "
            "or use model.melody_initialization: random with model.random_melody_encoder"
        )
    logger.info(
        "Melody initialization: %s; pretrained weights loaded=%s",
        provenance["initialization"], provenance["weights_loaded"],
    )
    decoder = cfg.get("bridge_decoder", {})
    if smoke_test:
        decoder = dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0.0)
    with log_stage("Building bridge model"):
        model = ProsodyBridge(
            melody_config,
            max_syllables=data["max_syllables"],
            encoder_lines_per_window=encoder_lines,
            max_window_notes=max_window_notes,
            bridge_scope=cfg.get("bridge_scope", "song"),
            max_song_notes=cfg.get("max_song_notes", 2048),
            max_song_lines=cfg.get("max_song_lines", 256),
            max_target_length=cfg.get("max_target_length"),
            song_encoder_layers=(
                1 if smoke_test else cfg.get("song_encoder_layers", 2)
            ),
            provenance=provenance,
            **decoder,
        )
        if tower is not None:
            model.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
            del tower
    datasets = {
        split: BridgeDataset(
            data["prepared_dir"],
            split,
            **model.dataset_options,
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
    with log_stage("Moving bridge model to %s", device):
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
        "melody_initialization": provenance["initialization"],
        "bridge_target_scheme": model.target_scheme,
        "bridge_scope": model.bridge_scope,
        "training_count_source": "ipa",
        "inference_count_source": "notes",
        "output_design": model.output_design,
        "config": config,
        "smoke_test": smoke_test,
        "data_schema_version": manifest["schema_version"],
        "pronunciation": manifest["pronunciation"],
        "data_sha256": manifest["sha256"],
        "pretraining_split_audit": manifest.get("pretraining_split_audit"),
        "melody_provenance": provenance,
        "precision": precision,
        "total_steps": steps * epochs,
        "examples": {s: len(ds) for s, ds in datasets.items()},
        "encoder_windows": {s: ds.encoder_windows for s, ds in datasets.items()},
        "songs": {s: len(ds.songs) for s, ds in datasets.items()},
        "skipped_examples": {s: ds.skipped for s, ds in datasets.items()},
        "skipped_limits": {s: ds.skipped_limits for s, ds in datasets.items()},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info(
        "Bridge ready: %d train / %d valid batches, accumulation=%d; output=%s",
        len(loaders["train"]), len(loaders["valid"]), accumulation, output,
    )
    best, stale = math.inf, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in tqdm(range(epochs), desc="Bridge epochs"):
            model.set_melody_trainable(
                bool(provenance.get("random_smoke_test_tower"))
                or (unfreeze is not None and epoch >= unfreeze)
            )
            logger.info(
                "Epoch %d/%d starting; melody frozen=%s", epoch + 1, epochs, model.melody_frozen
            )
            with tqdm(loaders["train"], desc="Bridge train", leave=False) as progress:
                training = run_bridge_epoch(
                    model,
                    progress,
                    device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    accumulation_steps=accumulation,
                    precision=precision,
                    gradient_clip=clip,
                    max_batches=max_batches,
                    history_mask_probability=history_mask_probability,
                )
            with tqdm(loaders["valid"], desc="Bridge validate", leave=False) as progress:
                validation = run_bridge_epoch(
                    model,
                    progress,
                    device,
                    precision=precision,
                    max_batches=max_batches,
                )
            logger.info(
                "Epoch %d train %.4f valid %.4f; starting template generation "
                "(cached decoder cuDNN SDPA disabled)",
                epoch + 1, training["loss"], validation["loss"],
            )
            with (
                log_stage("Epoch %d template generation", epoch + 1),
                tqdm(loaders["valid"], desc="Bridge generate", leave=False) as progress,
            ):
                validation["generation"] = evaluate_templates(
                    model,
                    progress,
                    device,
                    max_batches=max_batches,
                    precision=precision,
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
            generation = validation["generation"]
            logger.info(
                "Epoch %d train %.4f valid %.4f; free generation accuracy (IPA counts): "
                "strength=%s length=%s pair=%s",
                epoch + 1, training["loss"], validation["loss"],
                *(f"{generation[key]:.4f}" if generation[key] is not None else "n/a"
                  for key in ("strength_accuracy", "length_accuracy", "pair_accuracy")),
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
