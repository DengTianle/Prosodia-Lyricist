"""Train melody slot queries through a pretrained, four-stream lyric generator."""

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
from transformers import AutoTokenizer

from .audit_bridge_splits import check_training_splits
from .bridge_model import ProsodyBridge
from .bridge_train import _random_melody_config
from .data import read_manifest
from .end_to_end_data import EndToEndCollator, EndToEndDataset
from .end_to_end_model import EndToEndProsody
from .features import token_id
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .model import LOSS_NAMES, ProsodyBart
from .runtime import log_stage, seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)


def validate_precision(device, precision):
    if precision not in ("fp32", "bf16") or (
        precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported())
    ):
        raise ValueError("Use fp32 on CPU/MPS; bf16 requires supported CUDA hardware")


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def run_epoch(
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
    template_mode="soft",
):
    training = optimizer is not None
    if accumulation_steps < 1 or (training and template_mode != "soft"):
        raise ValueError("Use positive accumulation and soft embeddings for training")
    model.train(training)
    totals = dict.fromkeys((*LOSS_NAMES, "bridge_anchor"), 0.0)
    tokens, anchor_slots, word_events, batches, updates, songs = 0, 0, 0, 0, 0, 0
    correct = dict.fromkeys(("strength", "length", "pair"), 0)
    word_correct = dict.fromkeys(("syllables", "stresses", "lengths"), 0)
    iterator = iter(islice(loader, max_batches))
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            group_tokens = sum(int(b["labels"].ne(-100).sum()) for b in group)
            group_slots = sum(int(b["bridge_labels"].ne(-100).sum()) for b in group)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {key: value.to(device) for key, value in batch.items()}
                n = int(batch["labels"].ne(-100).sum())
                active = batch["bridge_labels"].ne(-100)
                k = int(active.sum())
                with autocast(precision):
                    output = model(**batch, template_mode=template_mode)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite end-to-end loss")
                if training:
                    # The four decoder streams have identical padding. The optional
                    # anchor has a DIFFERENT denominator: valid source prosody slots.
                    loss = sum(
                        model.template_model.loss_weights[name] * output.loss_components[name]
                        for name in LOSS_NAMES
                    ) * (n / group_tokens)
                    if model.anchor_weight and group_slots:
                        loss = loss + model.anchor_weight * output.loss_components[
                            "bridge_anchor"
                        ] * (k / group_slots)
                    loss.backward()
                for name in LOSS_NAMES:
                    totals[name] += output.loss_components[name].item() * n
                totals["bridge_anchor"] += output.loss_components["bridge_anchor"].item() * k
                predicted = output.bridge_output
                stress = predicted.strength_logits[active].argmax(-1)
                length = predicted.length_logits[active].argmax(-1)
                truth = batch["bridge_labels"][active]
                correct["strength"] += int(stress.eq(truth // 2).sum())
                correct["length"] += int(length.eq(truth % 2).sum())
                correct["pair"] += int((2 * stress + length).eq(truth).sum())
                events = batch["syllable_labels"].gt(0)
                for name, key in zip(
                    word_correct, ("syllable_labels", "stress_labels", "length_labels"), strict=True
                ):
                    word_correct[name] += int(
                        output.prosody_logits[name].argmax(-1)[events].eq(batch[key][events]).sum()
                    )
                word_events += int(events.sum())
                tokens += n
                anchor_slots += k
                batches += 1
                songs += len(batch["labels"])
            if training:
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                updates += 1
    if not tokens:
        raise ValueError("No lyric targets")
    components = {name: totals[name] / tokens for name in LOSS_NAMES}
    components["bridge_anchor"] = totals["bridge_anchor"] / anchor_slots if anchor_slots else 0.0
    return {
        "loss": sum(model.template_model.loss_weights[k] * components[k] for k in LOSS_NAMES)
        + model.anchor_weight * components["bridge_anchor"],
        "components": components,
        "tokens": tokens,
        "songs": songs,
        "batches": batches,
        "optimizer_steps": updates,
        "bridge_accuracy": {
            name: n / anchor_slots if anchor_slots else None for name, n in correct.items()
        },
        "bridge_compared_slots": anchor_slots,
        "teacher_forced_word_event_accuracy": {
            name: n / word_events if word_events else None for name, n in word_correct.items()
        },
        "word_events": word_events,
    }


def build_model(config):
    data, cfg, settings = config["data"], config["model"], config["training"]
    checkpoint = cfg.get("template_checkpoint")
    if not checkpoint:
        raise ValueError("Set model.template_checkpoint to a pretrained explainable checkpoint")
    checkpoint = Path(checkpoint)
    with log_stage("Loading pretrained template decoder: %s", checkpoint):
        template = ProsodyBart.load(checkpoint)
        if not isinstance(template, ProsodyBart) or (checkpoint / "melody.json").exists():
            raise ValueError(
                "Use an explainable template-only checkpoint, not a legacy/direct model"
            )
        if (checkpoint / "end_to_end.json").exists():
            raise ValueError("template_checkpoint must be the original template-only pretrain")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    if data["max_syllables"] != template.max_syllables:
        raise ValueError("Prepared data and template checkpoint syllable vocabularies differ")
    if len(tokenizer) != template.bart.config.vocab_size:
        raise ValueError("Pretrained tokenizer differs from the template model vocabulary")
    for name in ("max_source_length", "max_target_length"):
        if not 2 <= cfg[name] <= template.bart.config.max_position_embeddings:
            raise ValueError(f"{name} must fit the pretrained BART position limit")
    initialization = cfg.get("melody_initialization", "pretrained")
    tower = None
    if initialization == "pretrained":
        if not cfg.get("melody_checkpoint"):
            raise ValueError("Set model.melody_checkpoint to the contrastive checkpoint")
        tower, melody_config, provenance = load_contrastive_melody(cfg["melody_checkpoint"])
        if provenance["pretraining_train_split"] != "train" or (
            provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported contrastive pretraining split semantics")
        provenance = {**provenance, "initialization": initialization, "weights_loaded": True}
    elif initialization == "random":
        melody_config = _random_melody_config(cfg.get("random_melody_encoder"))
        provenance = dict(
            initialization="random", weights_loaded=False, checkpoint=None, sha256=None
        )
    else:
        raise ValueError("melody_initialization must be pretrained or random")
    encoder_lines = cfg.get("encoder_lines_per_window", data["lines_per_window"])
    if encoder_lines != data["lines_per_window"]:
        raise ValueError("encoder_lines_per_window differs from the prepared windows")
    bridge = ProsodyBridge(
        melody_config,
        bridge_scope="song",
        query_mode="parallel",
        max_syllables=data["max_syllables"],
        encoder_lines_per_window=encoder_lines,
        max_window_notes=cfg.get("max_window_notes", 512),
        max_song_notes=cfg.get("max_song_notes", 2048),
        max_song_lines=cfg.get("max_song_lines", 256),
        max_target_length=cfg.get("max_bridge_length", 4096),
        song_encoder_layers=cfg.get("song_encoder_layers", 2),
        provenance=provenance,
        **cfg.get("slot_encoder", {}),
    )
    if tower is not None:
        bridge.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
    previous = checkpoint / "run.json"
    template_run = json.loads(previous.read_text()) if previous.exists() else {}
    model = EndToEndProsody(
        bridge,
        template,
        stress_token_ids=[token_id(tokenizer, f"<{s}>") for s in ("strong", "weak")],
        temperature=cfg.get("temperature", 1.0),
        anchor_weight=settings.get("bridge_anchor_weight", 0),
        loss_weights={**dict.fromkeys(LOSS_NAMES, 1.0), **(settings.get("loss_weights") or {})},
        provenance={
            "template_checkpoint": str(checkpoint.resolve()),
            "template_weights_sha256": file_sha256(checkpoint / "weights.pt"),
            "template_pretraining_data_sha256": template_run.get("data_sha256"),
            "melody": provenance,
        },
    )
    return model, tokenizer


def train_end_to_end(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    for key in ("epochs", "batch_size", "patience", "gradient_accumulation_steps"):
        value = settings.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "melody_learning_rate", "template_learning_rate"):
        if not 0 < settings[key] < math.inf:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("melody_unfreeze_epoch", "template_unfreeze_epoch"):
        value = settings.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{key} must be null or a zero-based epoch")
    finetune = settings.get("template_finetune", "encoder")
    if finetune not in ("encoder", "all"):
        raise ValueError("template_finetune must be encoder or all")
    clip = settings.get("gradient_clip", 1.0)
    if clip is not None and not 0 < clip < math.inf:
        raise ValueError("gradient_clip must be positive and finite")
    if "history_mask_probability" in settings:
        raise ValueError("There is no bridge label history in end-to-end training")
    seed_everything(settings["seed"])
    device = select_device(settings["device"])
    precision = "fp32" if smoke_test else settings.get("precision", "fp32")
    validate_precision(device, precision)
    manifest = read_manifest(data["prepared_dir"])
    # Moving a prepared dataset between machines changes paths, not its features.
    paths = {"dali_dir", "prepared_dir", "pretraining_manifest", "song_ids_file"}
    if {k: v for k, v in data.items() if k not in paths} != {
        k: v for k, v in manifest["config"].items() if k not in paths
    }:
        raise ValueError("Data preparation settings differ from the manifest")
    if (
        data.get("song_ids_file")
        and file_sha256(data["song_ids_file"]) != manifest["selection"]["sha256"]
    ):
        raise ValueError("Song selection changed; prepare again")
    split_check = check_training_splits(
        data,
        manifest,
        mode=settings.get("pretraining_audit_mode", "strict"),
        required=cfg.get("melody_initialization", "pretrained") == "pretrained",
        encoder_lines=data["lines_per_window"],
    )
    model, tokenizer = build_model(config)
    datasets = {
        split: EndToEndDataset(
            data["prepared_dir"],
            split,
            tokenizer,
            **model.dataset_options,
            max_source_length=cfg["max_source_length"],
            max_target_length=cfg["max_target_length"],
            limit=4 if smoke_test else None,
        )
        for split in ("train", "valid")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=min(2, settings["batch_size"]) if smoke_test else settings["batch_size"],
            shuffle=split == "train",
            collate_fn=EndToEndCollator(tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(settings["seed"]),
            num_workers=0 if smoke_test else settings.get("num_workers", 0),
            pin_memory=device.type == "cuda",
        )
        for split, ds in datasets.items()
    }
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.bridge.named_parameters()
                    if not n.startswith("melody_encoder.")
                ],
                "lr": settings["learning_rate"],
            },
            {
                "params": model.bridge.melody_encoder.parameters(),
                "lr": settings["melody_learning_rate"],
            },
            {"params": model.template_model.parameters(), "lr": settings["template_learning_rate"]},
        ],
        betas=tuple(settings.get("adam_betas", (0.9, 0.98))),
        eps=settings.get("adam_epsilon", 1e-5),
        weight_decay=settings.get("weight_decay", 0.01),
    )
    epochs, max_batches = (1, 2) if smoke_test else (settings["epochs"], None)
    accumulation = settings.get("gradient_accumulation_steps", 1)
    steps = math.ceil(
        min(len(loaders["train"]), max_batches or len(loaders["train"])) / accumulation
    )
    scheduler = build_scheduler(optimizer, settings, epochs * steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(output_dir)
        if output_dir
        else Path(settings["output_dir"]) / (f"smoke-{stamp}" if smoke_test else stamp)
    )
    output.mkdir(parents=True, exist_ok=False)
    run = dict(
        conditioning="end_to_end",
        config=config,
        smoke_test=smoke_test,
        data_schema_version=manifest["schema_version"],
        data_sha256=manifest["sha256"],
        pronunciation=manifest["pronunciation"],
        pretraining_split_check=split_check,
        provenance=model.provenance,
        training_count_source="ipa",
        inference_count_source="notes",
        checkpoint_selection=(
            "combined validation loss (weighted decoder losses plus bridge anchor)"
        ),
        template_split_verification=(
            "same_prepared_data"
            if model.provenance["template_pretraining_data_sha256"] == (manifest["sha256"])
            else "unverified: template pretraining used different or unrecorded data hashes"
        ),
        precision=precision,
        total_steps=steps * epochs,
        examples={s: len(ds) for s, ds in datasets.items()},
        skipped_limits={s: ds.skipped_limits for s, ds in datasets.items()},
    )
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    best, stale = math.inf, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in range(epochs):
            unfreeze = settings.get("melody_unfreeze_epoch")
            model.bridge.set_melody_trainable(unfreeze is not None and epoch >= unfreeze)
            unfreeze = settings.get("template_unfreeze_epoch")
            model.set_template_trainable(
                finetune if unfreeze is not None and epoch >= unfreeze else "frozen"
            )
            logger.info(
                "Epoch %d: melody frozen=%s, template trainable=%s",
                epoch + 1,
                model.bridge.melody_frozen,
                model.template_trainable,
            )
            training = run_epoch(
                model,
                tqdm(loaders["train"], desc="Joint train"),
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                accumulation_steps=accumulation,
                precision=precision,
                gradient_clip=clip,
                max_batches=max_batches,
            )
            validation = run_epoch(
                model,
                tqdm(loaders["valid"], desc="Joint validate"),
                device,
                precision=precision,
                max_batches=max_batches,
            )
            handle.write(
                json.dumps(
                    dict(
                        epoch=epoch + 1,
                        train=training,
                        valid=validation,
                        learning_rates=scheduler.get_last_lr(),
                        melody_frozen=model.bridge.melody_frozen,
                        template_trainable=model.template_trainable,
                    )
                )
                + "\n"
            )
            handle.flush()
            score = validation["loss"]
            logger.info(
                "Epoch %d combined loss: train=%.4f valid=%.4f; output=%s",
                epoch + 1,
                training["loss"],
                score,
                output,
            )
            if score < best:
                best, stale = score, 0
                model.save(output / "best", tokenizer)
                (output / "best" / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    logger.info("Best end-to-end checkpoint: %s", output / "best")
    return output
