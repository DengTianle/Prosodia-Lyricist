"""Supervised training and fine-tuning of a global melody-to-template bridge."""

import json
import logging
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .audit_bridge_splits import check_training_splits
from .bridge_data import (
    PAIR_OFFSET,
    BridgeDataset,
    bridge_collate,
)
from .bridge_metrics import ClassificationMetrics, TemplateMetrics
from .bridge_model import ProsodyBridge
from .data import read_manifest
from .ipa import REPO_STRESSES
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .runtime import log_stage, seed_everything, select_device
from .split_integrity import read_content_index, require_clean_splits, training_exposure
from .train import build_scheduler

logger = logging.getLogger(__name__)


def _length_class_weights(value):
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
            for w in value
        )
    ):
        raise ValueError(
            "length_class_weights must be null or two positive finite weights [long, short]"
        )
    # Weighted means depend only on the ratio; normalize to avoid overflow.
    scale = max(value)
    return tuple(float(w / scale) for w in value)


def _supervised_slots(batch, first_line):
    active = batch["labels"].ge(PAIR_OFFSET) & batch["labels"].lt(first_line)
    if "target_note_indices" in batch:
        active &= batch["target_note_indices"].ge(0)
    return active


def _melody_encoder_config(config):
    """Resolve an explicit tower architecture without opening a pretrained checkpoint."""
    required = {"d_model", "num_layers", "num_heads", "dim_feedforward"}
    allowed = required | {"dropout", "pooling", "max_length"}
    if not isinstance(config, dict) or not required.issubset(config) or config.keys() - allowed:
        raise ValueError(
            "model.melody_encoder requires d_model, num_layers, num_heads, "
            "dim_feedforward; optional keys are dropout, pooling, max_length"
        )
    config = {"dropout": 0.1, "pooling": "cls", "max_length": 4096, **config}
    for name in (*sorted(required), "max_length"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"model.melody_encoder.{name} must be a positive integer")
    if config["d_model"] % config["num_heads"]:
        raise ValueError("model.melody_encoder.d_model must be divisible by num_heads")
    if not isinstance(config["dropout"], (int, float)) or not 0 <= config["dropout"] < 1:
        raise ValueError("model.melody_encoder.dropout must be in [0, 1)")
    if config["pooling"] not in ("cls", "mean"):
        raise ValueError("model.melody_encoder.pooling must be cls or mean")
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
    template_metrics=False,
    length_class_weights=None,
):
    training = optimizer is not None
    weight_values = _length_class_weights(length_class_weights)
    weights = (
        torch.tensor(weight_values, device=device, dtype=torch.float32)
        if weight_values is not None
        else None
    )
    model.train(training)
    total_loss = 0.0
    component_totals = {}
    weighted_length_total, length_weight_total = 0.0, 0.0
    tokens, phrases, correct, batches, updates = 0, 0, 0, 0, 0
    strength_correct, length_correct = 0, 0
    classification = ClassificationMetrics(
        model.strength_labels, collect_length_scores=not training
    )
    templates = TemplateMetrics(pairs=model.pairs) if template_metrics else None
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            first_line = model.first_line
            group_masks = [_supervised_slots(batch, first_line) for batch in group]
            group_tokens = sum(int(mask.sum()) for mask in group_masks)
            if not group_tokens:
                continue
            if weights is not None:
                group_longs = sum(
                    int(((batch["labels"][mask] - PAIR_OFFSET) % 2).eq(0).sum())
                    for batch, mask in zip(group, group_masks, strict=True)
                )
                group_length_weight = weight_values[0] * group_longs + weight_values[1] * (
                    group_tokens - group_longs
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {key: value.to(device) for key, value in batch.items()}
                active = _supervised_slots(batch, first_line)
                count = int(active.sum())
                if not count:
                    continue
                expected = batch["labels"][active] - PAIR_OFFSET
                line_count = int(batch["line_counts"].sum())
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if precision == "bf16"
                    else (nullcontext())
                ):
                    output = model(**batch)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite bridge loss")
                if weights is not None:
                    weighted_length = F.cross_entropy(
                        output.length_logits[active].float(),
                        expected % 2,
                        weight=weights,
                        reduction="sum",
                    )
                    if not torch.isfinite(weighted_length):
                        raise FloatingPointError("Non-finite weighted length CE")
                    longs = int(expected.remainder(2).eq(0).sum())
                    batch_weight = weight_values[0] * longs + weight_values[1] * (count - longs)
                    weighted_length_total += weighted_length.item()
                    length_weight_total += batch_weight
                if training:
                    # Each head uses its own denominator across the accumulation group.
                    objective = (
                        output.loss_components["strength"] * (count / group_tokens)
                        + weighted_length / group_length_weight
                        if weights is not None
                        else output.loss * (count / group_tokens)
                    )
                    objective.backward()
                predicted = (
                    2 * output.strength_logits.argmax(-1) + output.length_logits.argmax(-1)
                )[active]
                classification.update(
                    predicted,
                    expected,
                    strength_logits=output.strength_logits[active],
                    length_logits=output.length_logits[active],
                )
                correct += int(predicted.eq(expected).sum())
                # Each stress class occupies two pair IDs, ordered long then short.
                strength_correct += int((predicted // 2).eq(expected // 2).sum())
                length_correct += int((predicted % 2).eq(expected % 2).sum())
                total_loss += output.loss.item() * count
                for name, value in output.loss_components.items():
                    component_totals[name] = component_totals.get(name, 0.0) + value.item() * count
                if templates is not None:
                    note_counts = model.note_counts(batch["note_line_ids"], batch["line_counts"])
                    templates.diagnose_counts(batch["syllable_counts"], note_counts)
                    templates.record(
                        model.template_sequences(output).tolist(),
                        batch["labels"].tolist(),
                        note_counts.tolist(),
                        target_mask=output.target_mask.tolist(),
                    )
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
                if scheduler is not None:
                    scheduler.step()
                updates += 1
    if not tokens:
        raise ValueError("No template targets")
    loss = total_loss / tokens
    result = {
        "loss": loss,
        "components": {name: value / tokens for name, value in component_totals.items()},
        **{
            f"{name}_accuracy": value / tokens
            for name, value in (
                ("strength", strength_correct),
                ("length", length_correct),
                ("pair", correct),
            )
        },
        "tokens": tokens,
        "phrases": phrases,
        "batches": batches,
        "optimizer_steps": updates,
        "classification": classification.report(),
    }
    if templates is not None:
        result["templates"] = templates.report()
    if weights is not None:
        result["length_class_weights"] = list(length_class_weights)
        result["objective_components"] = {
            "strength": result["components"]["strength"],
            "length": weighted_length_total / length_weight_total,
        }
        result["objective_loss"] = sum(result["objective_components"].values())
    return result


def train_bridge(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    length_class_weights = settings.get("length_class_weights")
    _length_class_weights(length_class_weights)
    audit_mode = settings.get("pretraining_audit_mode", "strict")
    if audit_mode not in ("strict", "warning"):
        raise ValueError("training.pretraining_audit_mode must be strict or warning")
    initialization = cfg.get("melody_initialization", "random")
    if initialization not in ("random", "masked", "bridge"):
        raise ValueError("model.melody_initialization must be random, masked, or bridge")
    random_config = (
        _melody_encoder_config(cfg.get("melody_encoder")) if initialization == "random" else None
    )
    logger.info(
        "Starting bridge training: pid=%d torch=%s source=%s batch_size=%s workers=%s",
        os.getpid(),
        torch.__version__,
        __file__,
        settings["batch_size"],
        0 if smoke_test else settings.get("num_workers", 0),
    )
    for key in ("epochs", "batch_size", "patience", "gradient_accumulation_steps"):
        if not isinstance(settings.get(key, 1), int) or settings.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "melody_learning_rate"):
        if not 0 < settings.get(key, settings["learning_rate"]) < math.inf:
            raise ValueError(f"{key} must be positive and finite")
    unfreeze = settings.get("melody_unfreeze_epoch", 0)
    if unfreeze is not None and (not isinstance(unfreeze, int) or unfreeze < 0):
        raise ValueError("melody_unfreeze_epoch must be null or a nonnegative integer")
    clip = settings.get("gradient_clip", 1.0)
    if clip is not None and not 0 < clip < math.inf:
        raise ValueError("gradient_clip must be positive and finite")
    obsolete = {
        "bridge_scope",
        "query_mode",
        "query_attention",
        "encoder_lines_per_window",
        "max_window_notes",
        "max_notes",
        "song_encoder_layers",
        "song_encoder",
        "bridge_readout",
        "bridge_decoder",
        "random_melody_encoder",
    } & cfg.keys()
    if obsolete or "history_mask_probability" in settings:
        raise ValueError(f"Remove obsolete window/query bridge settings: {sorted(obsolete)}")
    if "meter_conditioning" in cfg and not isinstance(cfg["meter_conditioning"], bool):
        raise ValueError("meter_conditioning must be boolean")
    fusion = cfg.get("contrastive_fusion")
    fusion = {} if fusion is None else fusion
    if not isinstance(fusion, dict) or fusion.keys() - {
        "checkpoint",
        "unfreeze_epoch",
        "learning_rate",
    }:
        raise ValueError("contrastive_fusion accepts checkpoint, unfreeze_epoch and learning_rate")
    contrastive_unfreeze = fusion.get("unfreeze_epoch")
    if contrastive_unfreeze is not None and (
        isinstance(contrastive_unfreeze, bool)
        or not isinstance(contrastive_unfreeze, int)
        or contrastive_unfreeze < 0
    ):
        raise ValueError("contrastive_fusion.unfreeze_epoch must be null or a nonnegative integer")
    contrastive_lr = fusion.get(
        "learning_rate", settings.get("melody_learning_rate", settings["learning_rate"])
    )
    if not 0 < contrastive_lr < math.inf:
        raise ValueError("contrastive_fusion.learning_rate must be positive and finite")
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
    # Storage relocation and obsolete window metadata do not change note arrays.
    # Prepared checksums, recovered raw notes and upstream splits are checked separately.
    ignored = {"lines_per_window", "pretraining_manifest", "prepared_dir", "dali_dir", "lakh_file"}
    if {key: value for key, value in manifest["config"].items() if key not in ignored} != {
        key: value for key, value in data.items() if key not in ignored
    }:
        raise ValueError("Data configuration differs from preparation; prepare again")
    if data.get("song_ids_file") and (
        file_sha256(data["song_ids_file"]) != manifest["selection"]["sha256"]
    ):
        raise ValueError("Song allowlist changed; prepare again")
    with log_stage("Checking supervised content identities across prepared splits"):
        content_index = read_content_index(data["prepared_dir"])
        supervised_split_check = require_clean_splits(content_index)
    checkpoint = cfg.get("melody_checkpoint")
    strength_labels = (
        REPO_STRESSES if data.get("prosody_mode", "paper") == "actual" else ("strong", "weak")
    )
    provenance = {"initialization": initialization, "weights_loaded": initialization != "random"}
    masked_metadata = None
    if initialization == "bridge":
        if not checkpoint:
            raise ValueError("Set model.melody_checkpoint to a global bridge best/ directory")
        model = ProsodyBridge.load(checkpoint)
        if model.supervised_exposure is None:
            raise ValueError(
                "Legacy bridge has no supervised training exposure index. Its ancestor "
                "DALI/Lakh identities cannot be verified; retrain from random initialization "
                "on repaired splits before using it as a fine-tuning parent."
            )
        supervised_split_check = require_clean_splits(content_index, model.supervised_exposure)
        if model.strength_labels != strength_labels or model.max_syllables != data["max_syllables"]:
            raise ValueError("Fine-tuning data vocabulary/slot limits differ from the checkpoint")
        # The loaded architecture is authoritative; conflicting overrides are errors.
        for key in ("max_song_notes", "max_song_lines", "max_target_length", "alignment_policy"):
            if key in cfg and cfg[key] != getattr(model, key):
                raise ValueError(f"Fine-tuning {key} differs from the checkpoint")
        if (
            "melody_encoder" in cfg
            and _melody_encoder_config(cfg["melody_encoder"]) != model.melody_config
        ):
            raise ValueError("Fine-tuning melody_encoder differs from the checkpoint")
        provenance.update(
            checkpoint=str(Path(checkpoint).resolve()),
            sha256=file_sha256(Path(checkpoint) / "bridge_weights.pt"),
            source_provenance=model.provenance,
        )
        if cfg.get("meter_conditioning", model.meter_conditioning):
            model.enable_meter()
        elif model.meter_conditioning:
            raise ValueError("Cannot remove trained meter conditioning during fine-tuning")
    else:
        if initialization == "masked":
            from .bridge_pretrain import read_pretraining_metadata

            if not checkpoint:
                raise ValueError(
                    "Set model.melody_checkpoint to a masked-pretraining best/ directory"
                )
            masked_metadata = read_pretraining_metadata(checkpoint, manifest=manifest)
            melody_config = masked_metadata["melody_config"]
            if (
                "melody_encoder" in cfg
                and _melody_encoder_config(cfg["melody_encoder"]) != melody_config
            ):
                raise ValueError("Configured melody_encoder differs from masked pretraining")
            provenance.update(
                checkpoint=str(Path(checkpoint).resolve()),
                loaded_scope="full_encoder",
                sha256=file_sha256(Path(checkpoint) / "encoder_weights.pt"),
                source_provenance=masked_metadata["provenance"],
            )
        else:
            melody_config = random_config
            provenance.update(
                checkpoint=None, sha256=None, architecture_source="model.melody_encoder"
            )
        model = ProsodyBridge(
            melody_config,
            max_syllables=data["max_syllables"],
            strength_labels=strength_labels,
            max_song_notes=cfg.get("max_song_notes", 2048),
            max_song_lines=cfg.get("max_song_lines", 256),
            max_target_length=cfg.get("max_target_length", 4096),
            alignment_policy=cfg.get("alignment_policy", "strict"),
        )
        if masked_metadata is not None:
            from .bridge_pretrain import initialize_from_masked

            initialize_from_masked(model, checkpoint, masked_metadata)
        if cfg.get("meter_conditioning", False):
            model.enable_meter()
    with log_stage("Checking pretraining splits (%s mode)", audit_mode):
        if masked_metadata is not None and not fusion.get("checkpoint"):
            split_check = {
                "mode": audit_mode,
                "status": "not_applicable",
                "basis": "masked_pretraining_train_valid_checksums",
            }
        else:
            split_check = check_training_splits(
                data,
                manifest,
                mode=audit_mode,
                required=bool(fusion.get("checkpoint") or model.contrastive_encoder is not None),
            )
    if fusion.get("checkpoint"):
        if model.contrastive_encoder is not None:
            raise ValueError(
                "Fine-tuning already includes a contrastive tower; omit its checkpoint"
            )
        tower, contrastive_config, contrastive_provenance = load_contrastive_melody(
            fusion["checkpoint"]
        )
        if contrastive_provenance["pretraining_train_split"] != "train" or (
            contrastive_provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported contrastive pretraining split semantics")
        model.add_contrastive(contrastive_config, tower.state_dict())
        provenance["contrastive"] = contrastive_provenance
        del tower
    model.provenance = provenance
    model.supervised_exposure = training_exposure(content_index, model.supervised_exposure)
    datasets = {
        split: BridgeDataset(
            data["prepared_dir"],
            split,
            **model.dataset_options,
            dali_dir=data.get("dali_dir"),
            limit=8 if smoke_test else None,
        )
        for split in ("train", "valid")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=min(2, settings["batch_size"]) if smoke_test else settings["batch_size"],
            collate_fn=partial(bridge_collate, pairs=model.pairs),
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
                    p
                    for n, p in model.named_parameters()
                    if not n.startswith(("melody_encoder.", "contrastive_encoder."))
                ],
                "lr": settings["learning_rate"],
            },
            {
                "params": model.melody_encoder.parameters(),
                "lr": settings.get("melody_learning_rate", settings["learning_rate"]),
            },
            *(
                [{"params": model.contrastive_encoder.parameters(), "lr": contrastive_lr}]
                if model.contrastive_encoder is not None
                else []
            ),
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
        "length_class_weights": length_class_weights,
        "checkpoint_selection_metric": (
            "valid.objective_loss" if length_class_weights is not None else "valid.loss"
        ),
        "query_mode": model.query_mode,
        "query_attention": model.query_attention,
        "alignment_policy": model.alignment_policy,
        "alignment_counts": {s: dict(ds.alignment_counts) for s, ds in datasets.items()},
        "raw_alignment_sha256": {s: ds.raw_alignment_sha256 for s, ds in datasets.items()},
        "skipped_no_alignment": {s: ds.skipped_alignment for s, ds in datasets.items()},
        "output_labels": {"strength": list(model.strength_labels), "length": ["long", "short"]},
        "prosody_rules": model.prosody_rules,
        "config": config,
        "smoke_test": smoke_test,
        "data_schema_version": manifest["schema_version"],
        "pronunciation": manifest["pronunciation"],
        "data_sha256": manifest["sha256"],
        "pretraining_split_audit": manifest.get("pretraining_split_audit"),
        "pretraining_split_check": split_check,
        "supervised_split_check": supervised_split_check,
        "melody_provenance": provenance,
        "precision": precision,
        "total_steps": steps * epochs,
        "examples": {s: len(ds) for s, ds in datasets.items()},
        "source_scheme": "global_note_177d_v1",
        "meter_conditioning": model.meter_conditioning,
        "contrastive_fusion": model.contrastive_config is not None,
        "songs": {s: len(ds.songs) for s, ds in datasets.items()},
        "skipped_examples": {s: ds.skipped for s, ds in datasets.items()},
        "skipped_limits": {s: ds.skipped_limits for s, ds in datasets.items()},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info(
        "Bridge ready: %d train / %d valid batches, accumulation=%d; output=%s",
        len(loaders["train"]),
        len(loaders["valid"]),
        accumulation,
        output,
    )
    best, stale = math.inf, 0
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in tqdm(range(epochs), desc="Bridge epochs"):
            model.set_melody_trainable(unfreeze is not None and epoch >= unfreeze)
            model.set_contrastive_trainable(
                contrastive_unfreeze is not None and epoch >= contrastive_unfreeze
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
                    length_class_weights=length_class_weights,
                )
            with tqdm(loaders["valid"], desc="Bridge validate", leave=False) as progress:
                validation = run_bridge_epoch(
                    model,
                    progress,
                    device,
                    precision=precision,
                    max_batches=max_batches,
                    template_metrics=True,
                    length_class_weights=length_class_weights,
                )
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train": training,
                        "valid": validation,
                        "learning_rates": scheduler.get_last_lr(),
                        "melody_frozen": model.melody_frozen,
                        "contrastive_frozen": model.contrastive_frozen,
                    }
                )
                + "\n"
            )
            handle.flush()
            logger.info(
                "Epoch %d train %.4f valid %.4f; %s prediction accuracy (IPA targets): "
                "strength=%s length=%s pair=%s",
                epoch + 1,
                training["loss"],
                validation["loss"],
                "note",
                *(
                    f"{validation[key]:.4f}"
                    for key in ("strength_accuracy", "length_accuracy", "pair_accuracy")
                ),
            )
            length = validation["classification"]["length"]
            long_class = length["classes"]["long"]
            logger.info(
                "Length CE %.4f, label entropy %.4f; majority=%s (%.4f); "
                "predicted long=%.4f (mean probability %.4f), long recall=%s; "
                "balanced accuracy=%.4f; long ROC AUC=%s",
                validation["components"]["length"],
                length["label_entropy"],
                length["majority_class"],
                length["majority_accuracy"],
                long_class["predicted_fraction"],
                long_class["mean_probability"],
                long_class["recall"],
                length["balanced_accuracy"],
                length["ranking"]["roc_auc"],
            )
            if length_class_weights is not None:
                logger.info(
                    "Length class weights [long, short]=%s; weighted objective train %.4f "
                    "valid %.4f (valid weighted length CE %.4f); best/ uses this objective",
                    length_class_weights,
                    training["objective_loss"],
                    validation["objective_loss"],
                    validation["objective_components"]["length"],
                )
            selection_loss = validation.get("objective_loss", validation["loss"])
            if selection_loss < best:
                best, stale = selection_loss, 0
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
