"""Load a contrastive melody trunk without instantiating the audio tower."""

import hashlib
from pathlib import Path, PosixPath, WindowsPath

import torch

from .melody_encoder.encoding import MELODY_REPRESENTATION
from .melody_encoder.modeling import MelodyTransformerEncoder
from .runtime import log_stage

UPSTREAM_REVISION = "ff6247f2613fd5e1d1f06c44b73f9dadfa6cb257"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contrastive_melody(path):
    """Strictly import the current two-pool melody weights; never instantiate HuBERT.

    Upstream saves argparse Path values. Allow only those extra types, retaining
    weights_only loading rather than executing arbitrary checkpoint pickle code.
    """
    with (
        log_stage("Reading contrastive checkpoint metadata: %s", path),
        torch.serialization.safe_globals([Path, PosixPath, WindowsPath]),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("melody_representation") != MELODY_REPRESENTATION:
        raise ValueError("Expected a current note-based two-pool contrastive checkpoint")
    args = checkpoint["args"]
    if args.get("audio_pooling") != "note":
        raise ValueError("Expected two-pool checkpoint metadata audio_pooling='note'")
    names = ("d_model", "num_layers", "num_heads", "dim_feedforward")
    if any(f"melody_{name}" not in args for name in names):
        raise ValueError("Checkpoint is missing melody architecture arguments")
    config = {name: args[f"melody_{name}"] for name in names}
    config.update(
        projection_dim=args["projection_dim"],
        dropout=args["dropout"],
        pooling=args.get("melody_pooling", "cls"),
        max_length=args.get("melody_max_length", 4096),
    )
    # Existing upstream runs omit pooling/max_length and use cls/4096 defaults.
    with log_stage("Loading melody tower weights"):
        tower = MelodyTransformerEncoder(**config)
        state = {
            key.removeprefix("melody_encoder."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("melody_encoder.")
        }
        tower.load_state_dict(state, strict=True)
    del state, checkpoint
    # The projection operates AFTER pooling; applying it to each note is not the
    # representation trained by contrastive learning. Keep only the trained trunk.
    tower.projection = None
    tower.projection_dim = None
    config["projection_dim"] = None
    with log_stage("Hashing complete contrastive checkpoint: %s", path):
        checksum = file_sha256(path)
    provenance = {
        "checkpoint": str(Path(path).resolve()),
        "sha256": checksum,
        "upstream_revision": UPSTREAM_REVISION,
        "audio_pooling": args["audio_pooling"],
        "pretraining_manifest": str(args.get("manifest", "")),
        "pretraining_train_split": args.get("train_split", "train"),
        "pretraining_valid_split": args.get("val_split", "val"),
    }
    return tower, config, provenance
