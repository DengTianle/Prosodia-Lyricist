"""Train prosody-conditioned BART on prepared DALI lyric lines."""

import argparse
import hashlib
import json
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from .config import load_config
from .data import LyricDataset, ProsodyCollator, read_manifest
from .features import configure_tokenizer
from .model import ProsodyBart
from .runtime import seed_everything, select_device

logger = logging.getLogger(__name__)


def run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    scheduler=None,
    gradient_clip=1.0,
    max_batches=None,
    loss_aggregation="tokens",
):
    if loss_aggregation not in ("tokens", "batches"):
        raise ValueError("loss_aggregation must be tokens or batches")
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens, batches = 0.0, 0, 0
    components = {}
    total_weight = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="Train" if training else "Validate", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            tokens = int(batch["labels"].ne(-100).sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stopping without accepting this epoch")
            if training:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip, error_if_nonfinite=True
                    )
                optimizer.step()
                scheduler.step()
            weight = tokens if loss_aggregation == "tokens" else 1
            total_weight += weight
            total_loss += loss.item() * weight
            for name, value in getattr(output, "loss_components", {}).items():
                components[name] = components.get(name, 0.0) + value.item() * weight
            total_tokens += tokens
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    if not total_tokens:
        raise ValueError("No target tokens in this epoch")
    return {
        "loss": total_loss / total_weight,
        "tokens": total_tokens,
        "batches": batches,
        "components": {k: v / total_weight for k, v in components.items()},
    }


def build_scheduler(optimizer, settings, total_steps):
    """The original literal -1 schedule is preserved as an explicit experiment choice."""
    mode = settings.get("schedule", "linear")
    if mode not in ("xai_original", "linear", "constant_after_warmup"):
        raise ValueError("Unknown schedule")
    warmup = settings.get("warmup_steps", int(total_steps * settings.get("warmup_ratio", 0)))
    if warmup < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if mode == "constant_after_warmup":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup if step < warmup else 1.0
        )
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=-1 if mode == "xai_original" else total_steps,
    )


def train(config, *, output_dir=None, smoke_test=False, local_files_only=False):
    settings = config["training"]
    if any(settings[key] < 1 for key in ("epochs", "batch_size", "patience")):
        raise ValueError("epochs, batch_size, and patience must be positive")
    if not 0 <= settings.get("warmup_ratio", 0) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if (settings["gradient_clip"] is not None and settings["gradient_clip"] <= 0) or settings[
        "learning_rate"
    ] <= 0:
        raise ValueError("gradient_clip and learning_rate must be positive")
    seed_everything(settings["seed"])
    device = select_device(settings["device"])
    logger.info("Using %s", device)
    prepared_dir = config["data"]["prepared_dir"]
    manifest = read_manifest(prepared_dir)
    if manifest["config"] != config["data"]:
        raise ValueError("Data configuration changed since preparation; run prosodia-prepare again")
    if config["data"].get("song_ids_file"):
        selection_hash = hashlib.sha256(
            Path(config["data"]["song_ids_file"]).read_bytes()
        ).hexdigest()
        if selection_hash != manifest["selection"]["sha256"]:
            raise ValueError("Song ID list changed since preparation; run prosodia-prepare again")
    max_syllables = manifest["config"]["max_syllables"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained"], local_files_only=local_files_only
    )
    configure_tokenizer(tokenizer, max_syllables)
    loss_weights = settings.get("loss_weights", {})
    scaffold = any(loss_weights.get(k, 0) > 0 for k in ("syllable", "remainder", "sentence"))
    datasets = {
        split: LyricDataset(
            prepared_dir,
            split,
            tokenizer,
            max_source_length=config["model"]["max_source_length"],
            max_target_length=config["model"]["max_target_length"],
            limit=16 if smoke_test else None,
            scaffold=scaffold,
        )
        for split in ("train", "valid")
    }
    generator = torch.Generator().manual_seed(settings["seed"])
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=min(settings["batch_size"], 2) if smoke_test else settings["batch_size"],
            shuffle=split == "train",
            collate_fn=ProsodyCollator(tokenizer.pad_token_id),
            num_workers=0 if smoke_test else settings["num_workers"],
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    if settings.get("fixed_batch_order", False):
        # Original builds and shuffles its batch sampler once, outside the epoch loop.
        indices = list(range(len(datasets["train"])))
        rng = random.Random(settings["seed"])
        rng.shuffle(indices)
        size = min(settings["batch_size"], 2) if smoke_test else settings["batch_size"]
        batches = [indices[i : i + size] for i in range(0, len(indices), size)]
        rng.shuffle(batches)
        loaders["train"] = DataLoader(
            datasets["train"],
            batch_sampler=batches,
            collate_fn=ProsodyCollator(tokenizer.pad_token_id),
            num_workers=0 if smoke_test else settings["num_workers"],
        )
    if smoke_test:
        # Exercises real BART tokenization/data/optimization without downloading base weights.
        bart_config = BartConfig(
            vocab_size=len(tokenizer),
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_position_embeddings=max(
                config["model"]["max_source_length"], config["model"]["max_target_length"]
            ),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.eos_token_id,
        )
        bart = BartForConditionalGeneration(bart_config)
    else:
        bart = BartForConditionalGeneration.from_pretrained(
            config["model"]["pretrained"],
            local_files_only=local_files_only,
        )
        bart.resize_token_embeddings(len(tokenizer))
    for key in ("max_source_length", "max_target_length"):
        if not 2 <= config["model"][key] <= bart.config.max_position_embeddings:
            raise ValueError(f"{key} exceeds BART's position limit or is less than 2")
    model = ProsodyBart(
        bart,
        max_syllables=max_syllables,
        dropout=config["model"]["dropout"],
        loss_weights=loss_weights,
    ).to(device)
    epochs = 1 if smoke_test else settings["epochs"]
    max_batches = 2 if smoke_test else None
    steps_per_epoch = min(len(loaders["train"]), max_batches or len(loaders["train"]))
    total_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        betas=tuple(settings.get("adam_betas", (0.9, 0.98))),
    )
    scheduler = build_scheduler(optimizer, settings, total_steps)
    if settings.get("schedule") == "xai_original":
        logger.warning("Original schedule selected: learning rate becomes zero after warmup")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(output_dir)
        if output_dir
        else Path(settings["output_dir"]) / (f"smoke-{run_id}" if smoke_test else run_id)
    )
    # A run always has its own directory; never overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "config": config,
        "smoke_test": smoke_test,
        "total_steps": total_steps,
        "data_counts": manifest["counts"],
        "data_sha256": manifest["sha256"],
        "data_selection": manifest["selection"],
        "tokenized_lines": {split: len(ds) for split, ds in datasets.items()},
        "skipped_token_limits": {split: ds.skipped for split, ds in datasets.items()},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    best_loss, stale_epochs = math.inf, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(epochs):
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clip=settings["gradient_clip"],
                max_batches=max_batches,
                loss_aggregation=settings.get("loss_aggregation", "batches"),
            )
            valid_metrics = run_epoch(
                model,
                loaders["valid"],
                device,
                max_batches=max_batches,
                loss_aggregation=settings.get("loss_aggregation", "batches"),
            )
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "valid": valid_metrics,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            logger.info(
                "Epoch %d: train %.4f, valid %.4f",
                epoch + 1,
                train_metrics["loss"],
                valid_metrics["loss"],
            )
            if valid_metrics["loss"] < best_loss:
                best_loss, stale_epochs = valid_metrics["loss"], 0
                model.save(output / "best", tokenizer)
                (output / "best" / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
            else:
                stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                logger.info("Stopping after %d epochs without validation improvement", stale_epochs)
                break
    logger.info("Best checkpoint: %s", output / "best")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="New run directory (must not exist)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Two train/valid batches with tiny random BART; no pretrained weights",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(
        load_config(args.config),
        output_dir=args.output_dir,
        smoke_test=args.smoke_test,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
