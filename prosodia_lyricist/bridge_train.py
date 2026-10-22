"""Supervised training of a contrastive melody encoder's IPA-template bridge."""

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
    FIRST_LINE,
    PAIR_OFFSET,
    PAIRS,
    BridgeDataset,
    bridge_collate,
)
from .bridge_metrics import ClassificationMetrics, TemplateMetrics
from .bridge_model import ProsodyBridge
from .data import read_manifest
from .ipa import REPO_STRESSES
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .runtime import log_stage, seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)


def _length_class_weights(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(w, bool) or not isinstance(w, (int, float))
        or not math.isfinite(w) or w <= 0 for w in value
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
    template_metrics=False,
    length_class_weights=None,
):
    training = optimizer is not None
    weight_values = _length_class_weights(length_class_weights)
    if weight_values is not None and model.output_design != "separate":
        raise ValueError("Length class weights require separate stress/length heads")
    weights = (
        torch.tensor(weight_values, device=device, dtype=torch.float32)
        if weight_values is not None else None
    )
    model.train(training)
    forward_options = (
        {"history_mask_probability": history_mask_probability}
        if training and history_mask_probability else {}
    )
    total_loss = 0.0
    component_totals = {}
    weighted_length_total, length_weight_total = 0.0, 0.0
    tokens, phrases, correct, batches, updates = 0, 0, 0, 0, 0
    strength_correct, length_correct = 0, 0
    classification = ClassificationMetrics(
        getattr(model, "strength_labels", ("strong", "weak")), collect_length_scores=not training
    )
    parallel = getattr(model, "query_mode", "autoregressive") in ("parallel", "note")
    if template_metrics and not parallel:
        raise ValueError("Single-pass template metrics require parallel queries")
    templates = (
        TemplateMetrics(pairs=model.pairs, bridge_scope=model.bridge_scope)
        if template_metrics else None
    )
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            first_line = getattr(model, "first_line", FIRST_LINE)
            group_masks = [_supervised_slots(batch, first_line) for batch in group]
            group_tokens = sum(int(mask.sum()) for mask in group_masks)
            if not group_tokens:
                continue
            if weights is not None:
                group_longs = sum(
                    int(((batch["labels"][mask] - PAIR_OFFSET) % 2).eq(0).sum())
                    for batch, mask in zip(group, group_masks, strict=True)
                )
                group_length_weight = (
                    weight_values[0] * group_longs
                    + weight_values[1] * (group_tokens - group_longs)
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
                    output = model(**batch, **forward_options)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite bridge loss")
                if weights is not None:
                    weighted_length = F.cross_entropy(
                        output.length_logits[active].float(), expected % 2,
                        weight=weights, reduction="sum",
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
                        if weights is not None else output.loss * (count / group_tokens)
                    )
                    objective.backward()
                if model.output_design == "separate":
                    predicted = (
                        2 * output.strength_logits.argmax(-1) + output.length_logits.argmax(-1)
                    )[active]
                else:
                    predicted = output.logits.argmax(-1)[active]
                classification.update(
                    predicted, expected,
                    strength_logits=(
                        output.strength_logits[active]
                        if hasattr(output, "strength_logits") else None
                    ),
                    length_logits=(
                        output.length_logits[active] if hasattr(output, "length_logits") else None
                    ),
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
                        batch["labels"].tolist(), note_counts.tolist(),
                        target_mask=(
                            output.target_mask.tolist() if model.query_mode == "note" else None
                        ),
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
                scheduler.step()
                updates += 1
    if not tokens:
        raise ValueError("No template targets")
    loss = total_loss / tokens
    result = {
        "loss": loss,
        "components": {name: value / tokens for name, value in component_totals.items()},
        **{
            f"{'' if parallel else 'teacher_forced_'}{name}_accuracy": value / tokens
            for name, value in (
                ("strength", strength_correct), ("length", length_correct), ("pair", correct)
            )
        },
        "tokens": tokens,
        "phrases": phrases,
        "batches": batches,
        "optimizer_steps": updates,
        "classification": classification.report(),
    }
    if not parallel:
        result["history_mask_probability"] = history_mask_probability if training else 0.0
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


@torch.inference_mode()
def evaluate_templates(model, loader, device, *, max_batches=None, precision="fp32"):
    """Predict with reference IPA counts, without supplying any label history."""
    model.eval()
    if model.query_mode == "note":
        return run_bridge_epoch(
            model, loader, device, max_batches=max_batches, precision=precision,
            template_metrics=True,
        )["templates"]
    metrics = TemplateMetrics(
        pairs=getattr(model, "pairs", PAIRS), bridge_scope=model.bridge_scope
    )
    for batch in islice(loader, max_batches):
        batch = dict(batch)
        labels = batch.pop("labels")
        # Supply the reference skeleton, but never feed gold prosody labels to generation.
        counts = batch["syllable_counts"]
        note_counts = model.note_counts(batch["note_line_ids"], batch["line_counts"])
        metrics.diagnose_counts(counts, note_counts)
        eligible = counts.le(model.max_syllables).all(-1)
        target_eligible = (counts.sum(-1) + batch["line_counts"]).le(model.max_target_length)
        metrics.counts.update(
            skipped_examples=int((~eligible).sum()),
            skipped_phrases=int(batch["line_counts"][~eligible].sum()),
            skipped_target_examples=int((eligible & ~target_eligible).sum()),
            skipped_target_phrases=int(batch["line_counts"][eligible & ~target_eligible].sum()),
        )
        eligible &= target_eligible
        if not eligible.any():
            continue
        labels = labels[eligible]
        retained_note_counts = note_counts[eligible].tolist()
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
        metrics.record(result.sequences.tolist(), labels.tolist(), retained_note_counts)
    return metrics.report()


def train_bridge(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    length_class_weights = settings.get("length_class_weights")
    _length_class_weights(length_class_weights)
    audit_mode = settings.get("pretraining_audit_mode", "strict")
    if audit_mode not in ("strict", "warning"):
        raise ValueError("training.pretraining_audit_mode must be strict or warning")
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
    if history_mask_probability:
        raise ValueError(
            "Remove training.history_mask_probability: parallel queries have no label history"
        )
    mode = cfg.get("query_mode", "parallel" if (
        "bridge_readout" in cfg or "bridge_decoder" in cfg
    ) else "note")
    if mode not in ("note", "parallel") or (
        cfg.get("bridge_scope", "song") != "song"
    ):
        raise ValueError("Bridge training requires note heads or parallel song queries")
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
    # Only the upstream audit input may change without re-preparing features.
    ignored = {"pretraining_manifest"} if audit_mode == "warning" else set()
    if ({key: value for key, value in manifest["config"].items() if key not in ignored}
            != {key: value for key, value in data.items() if key not in ignored}):
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
    with log_stage("Checking pretraining splits (%s mode)", audit_mode):
        split_check = check_training_splits(
            data, manifest, mode=audit_mode,
            required=bool(initialization == "pretrained" and checkpoint),
            encoder_lines=encoder_lines,
        )
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
    if "bridge_readout" in cfg and "bridge_decoder" in cfg:
        raise ValueError("Use model.bridge_readout, without the old bridge_decoder alias")
    if mode == "note" and ("bridge_readout" in cfg or "bridge_decoder" in cfg):
        raise ValueError("Note heads use model.song_encoder; remove bridge_readout/bridge_decoder")
    readout = (
        cfg.get("song_encoder", {}) if mode == "note" else
        cfg.get("bridge_readout", cfg.get("bridge_decoder", {}))
    )
    if smoke_test:
        readout = dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0.0)
    with log_stage("Building bridge model"):
        model = ProsodyBridge(
            melody_config,
            max_syllables=data["max_syllables"],
            encoder_lines_per_window=encoder_lines,
            max_window_notes=max_window_notes,
            bridge_scope="song",
            query_mode=mode,
            query_attention="none" if mode == "note" else "cross",
            alignment_policy=cfg.get("alignment_policy", "strict"),
            strength_labels=(
                REPO_STRESSES if data.get("prosody_mode", "paper") == "actual"
                else ("strong", "weak")
            ),
            max_song_notes=cfg.get("max_song_notes", 2048),
            max_song_lines=cfg.get("max_song_lines", 256),
            max_target_length=cfg.get("max_target_length"),
            song_encoder_layers=(
                1 if smoke_test else cfg.get("song_encoder_layers", 2)
            ),
            provenance=provenance,
            **readout,
        )
        if tower is not None:
            model.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
            del tower
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
        "length_class_weights": length_class_weights,
        "checkpoint_selection_metric": (
            "valid.objective_loss" if length_class_weights is not None else "valid.loss"
        ),
        "query_mode": model.query_mode,
        "query_attention": model.query_attention,
        "alignment_policy": model.alignment_policy if mode == "note" else None,
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
                    }
                )
                + "\n"
            )
            handle.flush()
            logger.info(
                "Epoch %d train %.4f valid %.4f; %s prediction accuracy (IPA targets): "
                "strength=%s length=%s pair=%s",
                epoch + 1, training["loss"], validation["loss"], mode,
                *(f"{validation[key]:.4f}"
                  for key in ("strength_accuracy", "length_accuracy", "pair_accuracy")),
            )
            length = validation["classification"]["length"]
            long_class = length["classes"]["long"]
            logger.info(
                "Length CE %.4f, label entropy %.4f; majority=%s (%.4f); "
                "predicted long=%.4f (mean probability %.4f), long recall=%s; "
                "balanced accuracy=%.4f; long ROC AUC=%s",
                validation["components"]["length"], length["label_entropy"],
                length["majority_class"], length["majority_accuracy"],
                long_class["predicted_fraction"], long_class["mean_probability"],
                long_class["recall"],
                length["balanced_accuracy"],
                length["ranking"]["roc_auc"],
            )
            if length_class_weights is not None:
                logger.info(
                    "Length class weights [long, short]=%s; weighted objective train %.4f "
                    "valid %.4f (valid weighted length CE %.4f); best/ uses this objective",
                    length_class_weights, training["objective_loss"], validation["objective_loss"],
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
