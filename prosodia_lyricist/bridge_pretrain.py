"""Self-supervised masked-note reconstruction of the complete bridge encoder."""

import argparse
import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .audit_bridge_splits import check_training_splits
from .bridge_data import SONG_FEATURE_SCHEME, bridge_collate, encode_bridge_song
from .bridge_model import ProsodyBridge
from .bridge_train import _random_melody_config
from .config import load_config
from .data import read_manifest
from .melody_checkpoint import file_sha256, load_contrastive_melody
from .melody_encoder.encoding import (
    DURATION_BINS,
    DURATION_SLICE,
    MELODY_REPRESENTATION,
    ONSET_SHIFT_BINS,
    ONSET_SHIFT_SLICE,
    PITCH_CHANGE_BINS,
    PITCH_CHANGE_SLICE,
    PITCH_SIGN_INDEX,
)
from .runtime import log_stage, seed_everything, select_device
from .train import build_scheduler

logger = logging.getLogger(__name__)
OBJECTIVE = "masked_note_attributes_v1"
ATTRIBUTES = dict(
    pitch=PITCH_CHANGE_BINS, pitch_sign=2, duration=DURATION_BINS, onset_shift=ONSET_SHIFT_BINS
)


def encoder_configuration(encoder):
    """Architecture shared by reconstruction and downstream note classification."""
    return {
        "melody_config": encoder.melody_config,
        "encoder_config": {
            **{k: v for k, v in encoder.decoder_config.items() if k != "num_layers"},
            "encoder_lines_per_window": encoder.encoder_lines_per_window,
            "max_window_notes": encoder.max_notes,
            "max_song_notes": encoder.max_song_notes,
            "max_song_lines": encoder.max_song_lines,
            "song_encoder_layers": encoder.song_encoder_layers,
        },
    }


class MelodyPretrainDataset(Dataset):
    """Read melody arrays only: no IPA, word alignment, lyrics, or raw DALI needed."""

    def __init__(
        self, directory, split, *, encoder_lines_per_window, max_window_notes,
        max_song_notes, max_song_lines, limit=None,
    ):
        if split not in ("train", "valid"):
            raise ValueError("Masked pretraining uses only train and valid splits")
        if any(not isinstance(v, int) or v < 1 for v in (
            encoder_lines_per_window, max_window_notes, max_song_notes, max_song_lines,
        )) or (limit is not None and limit < 1):
            raise ValueError("Pretraining source limits must be positive integers")
        manifest = read_manifest(directory)
        if not manifest["config"].get("include_melody") or (
            manifest["config"].get("lines_per_window") != encoder_lines_per_window
        ):
            raise ValueError("Prepared melody/window configuration differs from the encoder")
        path = Path(directory) / f"{split}.jsonl"
        with log_stage("Verifying prepared %s checksum: %s", split, path):
            if file_sha256(path) != manifest["sha256"][path.name]:
                raise ValueError("Prepared data differs from its manifest; prepare again")
        self.examples, self.songs = [], set()
        self.encoder_windows = 0
        self.skipped = dict(window_notes=0, song_notes=0, song_lines=0, no_targets=0)
        with log_stage("Encoding masked-pretraining %s melodies", split):
            with path.open(encoding="utf-8") as handle:
                for row in tqdm(handle, desc=f"Melody {split}", unit="song", leave=False):
                    song = json.loads(row)
                    if manifest["songs"][song["song_id"]]["split"] != split:
                        raise ValueError("Song appears in incorrect split")
                    lines = song["lines"]
                    counts = [len(line["melody"]["midi_pitches"]) for line in lines]
                    windows = [sum(counts[i:i + encoder_lines_per_window])
                               for i in range(0, len(counts), encoder_lines_per_window)]
                    over = dict(
                        window_notes=any(n > max_window_notes for n in windows),
                        song_notes=sum(counts) > max_song_notes,
                        song_lines=len(lines) > max_song_lines,
                        no_targets=not windows or sum(counts) <= len(windows),
                    )
                    if any(over.values()):
                        for reason, exceeded in over.items():
                            self.skipped[reason] += int(exceeded)
                        continue
                    self.examples.append(encode_bridge_song(lines, encoder_lines_per_window))
                    self.songs.add(song["song_id"])
                    self.encoder_windows += len(windows)
                    if limit is not None and len(self.examples) >= limit:
                        break
        if not self.examples:
            raise ValueError(f"No usable masked-pretraining songs in {split}; inspect limits")
        logger.info("%s: %d melody songs; skipped %s", split, len(self), self.skipped)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def eligible_notes(batch, encoder_lines_per_window):
    active = batch["melody_attention_mask"].bool()
    windows = (batch["note_line_ids"] - 1) // encoder_lines_per_window
    anchors = torch.ones_like(active)
    anchors[:, 1:] = windows[:, 1:] != windows[:, :-1]
    # The first note defines local pitch zero (and IOI zero). Keep that reference
    # visible rather than train trivial targets or leak its pitch through offsets.
    return active & ~anchors


def sample_note_mask(batch, probability, encoder_lines_per_window, *, generator):
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not (
        0 < probability <= 1
    ):
        raise ValueError("mask_probability must be in (0, 1]")
    eligible = eligible_notes(batch, encoder_lines_per_window).cpu()
    selected = torch.rand(eligible.shape, generator=generator) < probability
    selected &= eligible
    for row in range(len(selected)):
        candidates = eligible[row].nonzero().flatten()
        if not len(candidates):
            raise ValueError("Each pretraining song needs a non-anchor note")
        if not selected[row].any():
            choice = torch.randint(len(candidates), (), generator=generator)
            selected[row, candidates[choice]] = True
    return selected.to(batch["melody_attention_mask"].device)


def corrupt_notes(batch, selected):
    """Hide each note in BOTH feature views, including dependent successor IOIs."""
    local, song = batch["melody_features"].clone(), batch["song_features"].clone()
    local[selected] = 0
    song[selected] = 0
    following = torch.zeros_like(selected)
    following[:, 1:] = selected[:, :-1] & batch["melody_attention_mask"][:, 1:].bool()
    # A visible next note's pitch interval + absolute pitch would reveal the
    # masked pitch. Its IOI + absolute onset would likewise reveal masked timing.
    local[..., ONSET_SHIFT_SLICE].masked_fill_(following[..., None], 0)
    for channel in (1, 3, 5):  # previous pitch interval, log IOI, positive-IOI flag
        song[..., channel].masked_fill_(following, 0)
    return {
        "melody_features": local,
        "song_features": song,
        **{key: batch[key] for key in (
            "melody_attention_mask", "note_line_ids", "line_counts",
        )},
    }


class MaskedMelodyModel(nn.Module):
    def __init__(self, melody_config, encoder_config, *, provenance=None):
        super().__init__()
        self.encoder = ProsodyBridge(
            melody_config, bridge_scope="song", query_mode="note", **encoder_config
        )
        # Reuse precisely the downstream encoding path, without retaining or
        # training either IPA classifier. These reconstruction heads are temporary.
        del self.encoder.strength_head, self.encoder.length_head
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.encoder.decoder_config["d_model"], size)
            for name, size in ATTRIBUTES.items()
        })
        self.provenance = provenance or {}

    def forward(self, *, selected, **batch):
        eligible = eligible_notes(batch, self.encoder.encoder_lines_per_window)
        if selected.dtype != torch.bool or selected.shape != eligible.shape or (
            (selected & ~eligible).any() or not selected.any()
        ):
            raise ValueError("Reconstruction targets must be valid non-anchor notes")
        clean = batch["melody_features"][selected]
        targets = dict(
            pitch=clean[:, PITCH_CHANGE_SLICE].argmax(-1),
            pitch_sign=clean[:, PITCH_SIGN_INDEX].long(),
            duration=clean[:, DURATION_SLICE].argmax(-1),
            onset_shift=clean[:, ONSET_SHIFT_SLICE].argmax(-1),
        )
        hidden = self.encoder.encode(**corrupt_notes(batch, selected))[selected]
        logits = {name: head(hidden) for name, head in self.heads.items()}
        sums = {name: F.cross_entropy(logits[name].float(), target, reduction="sum")
                for name, target in targets.items()}
        count = int(selected.sum())
        return SimpleNamespace(
            loss=sum(sums.values()) / count, loss_sums=sums, count=count,
            logits=logits, targets=targets,
        )

    def save(self, directory, *, data_sha256, masking):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), directory / "encoder_weights.pt")
        torch.save(self.heads.state_dict(), directory / "reconstruction_weights.pt")
        metadata = {
            "format_version": 1, "objective": OBJECTIVE,
            "melody_representation": MELODY_REPRESENTATION,
            "song_feature_scheme": SONG_FEATURE_SCHEME,
            **encoder_configuration(self.encoder),
            "attributes": ATTRIBUTES, "masking": masking,
            "data_sha256": {k: data_sha256[k] for k in ("train.jsonl", "valid.jsonl")},
            "provenance": self.provenance,
        }
        (directory / "pretrain.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        metadata = read_pretraining_metadata(directory)
        model = cls(metadata["melody_config"], metadata["encoder_config"],
                    provenance=metadata["provenance"])
        model.encoder.load_state_dict(torch.load(
            Path(directory) / "encoder_weights.pt", map_location="cpu", weights_only=True
        ), strict=True)
        model.heads.load_state_dict(torch.load(
            Path(directory) / "reconstruction_weights.pt", map_location="cpu", weights_only=True
        ), strict=True)
        return model


def read_pretraining_metadata(directory, *, manifest=None):
    directory = Path(directory)
    meta = json.loads((directory / "pretrain.json").read_text(encoding="utf-8"))
    if (meta.get("format_version") != 1 or meta.get("objective") != OBJECTIVE
            or meta.get("melody_representation") != MELODY_REPRESENTATION
            or meta.get("song_feature_scheme") != SONG_FEATURE_SCHEME
            or meta.get("attributes") != ATTRIBUTES):
        raise ValueError("Unsupported masked-melody checkpoint or feature representation")
    if manifest is not None and any(
        meta["data_sha256"].get(k) != manifest["sha256"].get(k)
        for k in ("train.jsonl", "valid.jsonl")
    ):
        raise ValueError("Masked pretraining train/valid splits differ from prepared data")
    return meta


def initialize_from_masked(bridge, directory, metadata):
    if bridge.query_mode != "note" or encoder_configuration(bridge) != {
        k: metadata[k] for k in ("melody_config", "encoder_config")
    }:
        raise ValueError("Bridge encoder architecture differs from masked pretraining")
    state = torch.load(Path(directory) / "encoder_weights.pt",
                       map_location="cpu", weights_only=True)
    expected = set(bridge.state_dict()) - {
        f"{head}.{param}"
        for head in ("strength_head", "length_head") for param in ("weight", "bias")
    }
    if set(state) != expected:
        raise ValueError("Masked checkpoint must contain exactly the encoder weights")
    bridge.load_state_dict({**bridge.state_dict(), **state}, strict=True)


def run_pretrain_epoch(
    model, loader, device, *, mask_probability=0.15, mask_generator,
    optimizer=None, scheduler=None, accumulation_steps=1, precision="fp32",
    gradient_clip=1.0, max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    losses, correct = dict.fromkeys(ATTRIBUTES, 0.0), dict.fromkeys(ATTRIBUTES, 0)
    targets = {name: torch.zeros(size, dtype=torch.long) for name, size in ATTRIBUTES.items()}
    predictions = {name: torch.zeros_like(counts) for name, counts in targets.items()}
    tokens, notes, batches, updates = 0, 0, 0, 0
    iterator = iter(islice(loader, max_batches)) if max_batches is not None else iter(loader)
    with torch.enable_grad() if training else torch.no_grad():
        while group := list(islice(iterator, accumulation_steps if training else 1)):
            masks = [sample_note_mask(
                batch, mask_probability, model.encoder.encoder_lines_per_window,
                generator=mask_generator,
            ) for batch in group]
            group_tokens = sum(int(mask.sum()) for mask in masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch, selected in zip(group, masks, strict=True):
                batch = {key: value.to(device) for key, value in batch.items()}
                with (torch.autocast("cuda", dtype=torch.bfloat16)
                      if precision == "bf16" else nullcontext()):
                    output = model(**batch, selected=selected.to(device))
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("Non-finite masked-note reconstruction loss")
                if training:
                    (sum(output.loss_sums.values()) / group_tokens).backward()
                for name in ATTRIBUTES:
                    losses[name] += output.loss_sums[name].item()
                    predicted = output.logits[name].argmax(-1)
                    correct[name] += int(predicted.eq(output.targets[name]).sum())
                    targets[name] += torch.bincount(
                        output.targets[name].detach().cpu(), minlength=ATTRIBUTES[name]
                    )
                    predictions[name] += torch.bincount(
                        predicted.detach().cpu(), minlength=ATTRIBUTES[name]
                    )
                tokens += output.count
                notes += int(batch["melody_attention_mask"].sum())
                batches += 1
            if training:
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip, error_if_nonfinite=True
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                updates += 1
    if not tokens:
        raise ValueError("No masked notes were evaluated")
    components = {name: value / tokens for name, value in losses.items()}
    return {
        "loss": sum(components.values()), "components": components,
        "accuracy": {name: value / tokens for name, value in correct.items()},
        "majority_accuracy": {name: int(counts.max()) / tokens for name, counts in targets.items()},
        "target_counts": {name: counts.tolist() for name, counts in targets.items()},
        "prediction_counts": {name: counts.tolist() for name, counts in predictions.items()},
        "masked_notes": tokens, "source_notes": notes, "masked_fraction": tokens / notes,
        "batches": batches, "updates": updates,
    }


def train_pretrain(config, *, output_dir=None, smoke_test=False):
    data, cfg, settings = config["data"], config["model"], config["training"]
    initialization = cfg.get("melody_initialization", "random")
    if initialization not in ("random", "pretrained"):
        raise ValueError("Pretraining melody_initialization must be random or pretrained")
    if cfg.get("bridge_scope", "song") != "song" or cfg.get("query_mode", "note") != "note":
        raise ValueError("Masked pretraining requires the song encoder with note heads")
    for name in ("epochs", "batch_size", "patience", "gradient_accumulation_steps"):
        value = settings.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    probability = settings.get("mask_probability", 0.15)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not (
        0 < probability <= 1
    ):
        raise ValueError("mask_probability must be in (0, 1]")
    if not 0 < settings["learning_rate"] < math.inf:
        raise ValueError("learning_rate must be positive and finite")
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
    encoder_lines = cfg.get("encoder_lines_per_window", manifest["config"]["lines_per_window"])
    tower = None
    if initialization == "random":
        melody_config = _random_melody_config(cfg.get("random_melody_encoder"))
        provenance = {"initialization": "random", "weights_loaded": False}
        split_check = {"status": "not_applicable", "reason": "random initialization"}
    else:
        split_check = check_training_splits(
            {**manifest["config"], **data}, manifest,
            mode=settings.get("pretraining_audit_mode", "strict"), required=True,
            encoder_lines=encoder_lines,
        )
        tower, melody_config, provenance = load_contrastive_melody(cfg["melody_checkpoint"])
        if provenance["pretraining_train_split"] != "train" or (
            provenance["pretraining_valid_split"] not in ("val", "valid", "validation")
        ):
            raise ValueError("Unsupported contrastive pretraining split semantics")
        provenance = {**provenance, "initialization": "pretrained", "weights_loaded": True}
    model = MaskedMelodyModel(melody_config, {
        **cfg.get("song_encoder", {}),
        "encoder_lines_per_window": encoder_lines,
        "max_window_notes": cfg.get("max_window_notes", 512),
        "max_song_notes": cfg.get("max_song_notes", 2048),
        "max_song_lines": cfg.get("max_song_lines", 256),
        "song_encoder_layers": cfg.get("song_encoder_layers", 2),
    }, provenance=provenance)
    if tower is not None:
        model.encoder.melody_encoder.load_state_dict(tower.state_dict(), strict=True)
        del tower
    # All encoder layers learn immediately, at the pretraining learning rate.
    model.encoder.set_melody_trainable(True)
    options = encoder_configuration(model.encoder)["encoder_config"]
    datasets = {split: MelodyPretrainDataset(
        data["prepared_dir"], split,
        **{k: options[k] for k in ("encoder_lines_per_window", "max_window_notes",
                                  "max_song_notes", "max_song_lines")},
        limit=8 if smoke_test else None,
    ) for split in ("train", "valid")}
    loaders = {split: DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=split == "train",
        generator=torch.Generator().manual_seed(settings["seed"]),
        collate_fn=bridge_collate, num_workers=0 if smoke_test else settings.get("num_workers", 0),
    ) for split, dataset in datasets.items()}
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        betas=tuple(settings.get("adam_betas", (0.9, 0.98))),
        eps=settings.get("adam_epsilon", 1e-5),
        weight_decay=settings.get("weight_decay", 0.01),
    )
    epochs, max_batches = (1, 2) if smoke_test else (settings["epochs"], None)
    accumulation = settings.get("gradient_accumulation_steps", 1)
    steps = math.ceil(min(len(loaders["train"]), max_batches or math.inf) / accumulation)
    scheduler = build_scheduler(optimizer, settings, epochs * steps)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"smoke-{stamp}" if smoke_test else stamp
    output = Path(output_dir) if output_dir else Path(settings["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    masking = {"probability": probability, "replacement": "zero_both_views",
               "anchors": "keep_first_note_of_each_window", "successor_intervals": "hidden"}
    hashes = {k: manifest["sha256"][k] for k in ("train.jsonl", "valid.jsonl")}
    run = {
        "objective": OBJECTIVE, "config": config, "smoke_test": smoke_test,
        "precision": precision, "masking": masking, "data_sha256": hashes,
        "checkpoint_selection_metric": "valid.loss", "total_steps": epochs * steps,
        "melody_provenance": provenance, "pretraining_split_check": split_check,
        "examples": {s: len(ds) for s, ds in datasets.items()},
        "encoder_windows": {s: ds.encoder_windows for s, ds in datasets.items()},
        "skipped": {s: ds.skipped for s, ds in datasets.items()},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    logger.info("Masked-note pretraining: all encoders trainable; %s %s; output=%s",
                device, precision, output)
    best, stale = math.inf, 0
    training_masks = torch.Generator().manual_seed(settings["seed"])
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in tqdm(range(epochs), desc="Masked-note epochs"):
            training = run_pretrain_epoch(
                model, tqdm(loaders["train"], desc="Masked train", leave=False), device,
                mask_probability=probability, mask_generator=training_masks,
                optimizer=optimizer, scheduler=scheduler, accumulation_steps=accumulation,
                precision=precision, gradient_clip=clip, max_batches=max_batches,
            )
            validation = run_pretrain_epoch(
                model, tqdm(loaders["valid"], desc="Masked valid", leave=False), device,
                mask_probability=probability,
                # Identical held-out notes every epoch; training masks keep changing.
                mask_generator=torch.Generator().manual_seed(settings["seed"] + 1),
                precision=precision, max_batches=max_batches,
            )
            handle.write(json.dumps({"epoch": epoch + 1, "train": training, "valid": validation,
                                     "learning_rates": scheduler.get_last_lr()}) + "\n")
            handle.flush()
            logger.info("Epoch %d train CE %.4f valid CE %.4f; masked-note accuracy: %s",
                        epoch + 1, training["loss"], validation["loss"], validation["accuracy"])
            logger.info("Validation majority-class accuracy: %s", validation["majority_accuracy"])
            if validation["loss"] < best:
                best, stale = validation["loss"], 0
                model.save(output / "best", data_sha256=hashes, masking=masking)
                (output / "best" / "run.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
    logger.info("Best masked-note checkpoint: %s", output / "best")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="New run directory (must not exist)")
    parser.add_argument("--smoke-test", action="store_true", help="Two train/valid batches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    train_pretrain(load_config(args.config), output_dir=args.output_dir, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
