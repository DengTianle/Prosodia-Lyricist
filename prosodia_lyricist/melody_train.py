"""Song/paragraph training of direct melody conditioning and four output streams."""

import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

from .data import read_manifest
from .features import configure_tokenizer
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .melody_data import MelodyCollator, MelodySongDataset, read_melody_manifest
from .melody_model import MelodyBart
from .model import ProsodyBart
from .runtime import seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)


class LengthBatches(Sampler):
    """Shuffle songs, bucket by source/target length, then shuffle complete batches."""

    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        batches = []
        bucket_size = self.batch_size * 32
        for start in range(0, len(indices), bucket_size):
            bucket = sorted(
                indices[start : start + bucket_size],
                key=lambda i: max(
                    len(self.dataset[i]["input_ids"]), len(self.dataset[i]["labels"])
                ),
            )
            batches.extend(
                bucket[j : j + self.batch_size] for j in range(0, len(bucket), self.batch_size)
            )
        rng.shuffle(batches)
        return iter(batches)


def run_melody_epoch(
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
    """Token-weighted accumulation, including the last incomplete update group."""
    if accumulation_steps < 1 or precision not in ("fp32", "bf16"):
        raise ValueError("Invalid accumulation_steps or precision")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 training requires a CUDA device with BF16 support; use fp32")
    training = optimizer is not None
    model.train(training)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    totals, tokens, batches, updates = {}, 0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            group_tokens = sum(int(batch["labels"].ne(-100).sum()) for batch in group)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {key: value.to(device) for key, value in batch.items()}
                n = int(batch["labels"].ne(-100).sum())
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if precision == "bf16"
                    else nullcontext()
                ):
                    output = model(**batch)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite melody training loss")
                if training:
                    (output.loss * (n / group_tokens)).backward()
                for name, loss in {"loss": output.loss, **output.loss_components}.items():
                    totals[name] = totals.get(name, 0.0) + float(loss.detach()) * n
                tokens += n
                batches += 1
                del output
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
        raise ValueError("No target tokens")
    result = {
        "loss": totals.pop("loss") / tokens,
        "components": {name: value / tokens for name, value in totals.items()},
        "tokens": tokens,
        "batches": batches,
        "optimizer_steps": updates,
    }
    if device.type == "cuda":
        result.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
    return result


def train_melody(config, *, output_dir=None, smoke_test=False, local_files_only=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    if cfg.get("decoder_mode", "explainable") != "explainable":
        raise ValueError("Direct melody conditioning requires the explainable four-stream decoder")
    for key in ("epochs", "batch_size", "patience", "gradient_accumulation_steps"):
        if not isinstance(settings.get(key, 1), int) or settings.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "adapter_learning_rate", "melody_learning_rate"):
        if not 0 < settings[key] < math.inf:
            raise ValueError(f"{key} must be positive and finite")
    freeze_epochs = settings.get("freeze_decoder_epochs", 1)
    unfreeze = settings.get("melody_unfreeze_epoch")
    if not isinstance(freeze_epochs, int) or freeze_epochs < 0:
        raise ValueError("freeze_decoder_epochs must be a nonnegative integer")
    if unfreeze is not None and (not isinstance(unfreeze, int) or unfreeze < freeze_epochs):
        raise ValueError("melody_unfreeze_epoch must be null or >= freeze_decoder_epochs")
    if settings.get("schedule") not in ("linear", "constant_after_warmup"):
        raise ValueError("Use linear or constant_after_warmup for melody training")
    if settings.get("gradient_clip") is not None and settings["gradient_clip"] <= 0:
        raise ValueError("gradient_clip must be positive")
    seed_everything(settings["seed"])
    device = select_device(settings["device"])
    manifest = read_manifest(data["prepared_dir"])
    if manifest["config"] != data:
        raise ValueError("Data configuration differs from preparation; prepare again")
    if (
        data.get("song_ids_file")
        and file_sha256(data["song_ids_file"]) != (manifest["selection"]["sha256"])
    ):
        raise ValueError("Song allowlist changed; prepare again")
    checkpoint = cfg.get("melody_checkpoint")
    if not checkpoint and not smoke_test:
        raise ValueError("Set model.melody_checkpoint to a trained note-based two-pool checkpoint")
    if checkpoint:
        tower, melody_config, provenance = load_contrastive_melody(checkpoint)
        audit = manifest.get("pretraining_split_audit")
        if (
            not data.get("pretraining_manifest")
            or not audit
            or audit["sha256"] != (file_sha256(data["pretraining_manifest"]))
        ):
            raise ValueError(
                "Prepare with the unchanged melody pretraining manifest to audit splits"
            )
        if provenance["pretraining_train_split"] != "train" or (
            provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported upstream split semantics")
        upstream_rows, _ = read_melody_manifest(data["pretraining_manifest"], require_lyrics=False)
        window_sizes = {int(row["line_count"]) for row in upstream_rows if row.get("line_count")}
        if window_sizes and window_sizes != {data.get("lines_per_window", 1)}:
            raise ValueError("lines_per_window must match the melody pretraining manifest")
    else:
        tower = None
        melody_config = dict(
            d_model=16,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.0,
            projection_dim=None,
        )
        provenance = {"random_smoke_test_tower": True}
    max_syllables = data["max_syllables"]
    template_path = cfg.get("template_checkpoint")
    tokenizer_path = Path(template_path) / "tokenizer" if template_path else cfg["pretrained"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)
    if template_path:
        template = ProsodyBart.load(template_path)
        if type(template) is not ProsodyBart or template.max_syllables != max_syllables:
            raise ValueError("Expected compatible four-stream template-decoder checkpoint")
        provenance["template_weights_sha256"] = file_sha256(Path(template_path) / "weights.pt")
    else:
        configure_tokenizer(tokenizer, max_syllables)
        if smoke_test:
            bart = BartForConditionalGeneration(
                BartConfig(
                    vocab_size=len(tokenizer),
                    d_model=32,
                    encoder_layers=1,
                    decoder_layers=1,
                    encoder_attention_heads=2,
                    decoder_attention_heads=2,
                    encoder_ffn_dim=64,
                    decoder_ffn_dim=64,
                    max_position_embeddings=max(cfg["max_source_length"], cfg["max_target_length"]),
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=tokenizer.eos_token_id,
                )
            )
        else:
            bart = BartForConditionalGeneration.from_pretrained(
                cfg["pretrained"],
                local_files_only=local_files_only,
                dropout=cfg.get("transformer_dropout", 0.3),
            )
            bart.resize_token_embeddings(len(tokenizer))
        template = ProsodyBart(
            bart,
            max_syllables=max_syllables,
            dropout=cfg.get("dropout", 0.1),
            loss_weights=settings.get("loss_weights"),
        )
        if freeze_epochs:
            logger.warning(
                "No trained template checkpoint: newly initialized decoder features "
                "will also be frozen during adapter warmup"
            )
    source_config = {
        "lines_per_window": data.get("lines_per_window", 1),
        "include_title": data.get("include_title", True),
    }
    model = MelodyBart.from_template(
        template,
        melody_config=melody_config,
        provenance=provenance,
        source_config=source_config,
        loss_weights=settings.get("loss_weights"),
    )
    del template
    if tower is not None:
        model.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
        del tower
    for key in ("max_source_length", "max_target_length"):
        if not 3 <= cfg[key] <= model.bart.config.max_position_embeddings:
            raise ValueError(f"{key} exceeds BART positions or is too small")
    datasets = {
        split: MelodySongDataset(
            data["prepared_dir"],
            split,
            tokenizer,
            **source_config,
            unit=data.get("unit", "song"),
            max_source_length=cfg["max_source_length"],
            max_target_length=cfg["max_target_length"],
            max_window_length=melody_config.get("max_length", 4096),
            limit=8 if smoke_test else None,
        )
        for split in ("train", "valid")
    }
    collator = MelodyCollator(tokenizer.pad_token_id)
    batch_size = min(settings["batch_size"], 2) if smoke_test else settings["batch_size"]
    loaders = {
        "train": DataLoader(
            datasets["train"],
            collate_fn=collator,
            batch_sampler=LengthBatches(datasets["train"], batch_size, settings["seed"]),
            num_workers=settings.get("num_workers", 0),
        ),
        "valid": DataLoader(
            datasets["valid"],
            collate_fn=collator,
            batch_size=batch_size,
            num_workers=settings.get("num_workers", 0),
        ),
    }
    model.to(device)
    if settings.get("gradient_checkpointing", False):
        model.bart.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    epochs = 1 if smoke_test else settings["epochs"]
    accumulation = settings.get("gradient_accumulation_steps", 1)
    max_batches = 2 if smoke_test else None
    steps = math.ceil(
        min(len(loaders["train"]), max_batches or len(loaders["train"])) / accumulation
    )
    decoder_parameters = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(("adapter.", "melody_encoder."))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.adapter.parameters(), "lr": settings["adapter_learning_rate"]},
            {"params": decoder_parameters, "lr": settings["learning_rate"]},
            {"params": model.melody_encoder.parameters(), "lr": settings["melody_learning_rate"]},
        ],
        betas=tuple(settings.get("adam_betas", (0.9, 0.98))),
        eps=settings.get("adam_epsilon", 1e-5),
        weight_decay=settings.get("weight_decay", 0.0),
    )
    scheduler = build_scheduler(optimizer, settings, epochs * steps)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else Path(settings["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "conditioning": "melody",
        "decoder_mode": "explainable",
        "config": config,
        "smoke_test": smoke_test,
        "data_schema_version": manifest["schema_version"],
        "melody_provenance": provenance,
        "pretraining_split_audit": manifest.get("pretraining_split_audit"),
        "data_sha256": manifest["sha256"],
        "total_steps": epochs * steps,
        "examples": {s: len(ds) for s, ds in datasets.items()},
        "songs": {s: len(ds.songs) for s, ds in datasets.items()},
        "skipped_length_limits": {s: ds.skipped for s, ds in datasets.items()},
        "precision": "fp32" if smoke_test else settings.get("precision", "fp32"),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2))
    best, stale = math.inf, 0
    with (output / "metrics.jsonl").open("w") as handle:
        for epoch in range(epochs):
            model.set_trainable(
                melody=unfreeze is not None and epoch >= unfreeze, decoder=epoch >= freeze_epochs
            )
            training = run_melody_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation_steps=accumulation,
                precision=run["precision"],
                gradient_clip=settings.get("gradient_clip", 1.0),
                max_batches=max_batches,
            )
            validation = run_melody_epoch(
                model, loaders["valid"], device, precision=run["precision"], max_batches=max_batches
            )
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": training,
                        "valid": validation,
                        "learning_rates": scheduler.get_last_lr(),
                        "melody_frozen": model.melody_frozen,
                        "decoder_frozen": model.decoder_frozen,
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
                model.save(output / "best", tokenizer)
                (output / "best" / "run.json").write_text(json.dumps(run, indent=2))
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    return output
