import copy
import csv
from pathlib import Path

import pytest
import torch

from prosodia_lyricist.melody_checkpoint import file_sha256, load_contrastive_melody
from prosodia_lyricist.melody_data import audit_pretraining_splits, preserve_pretraining_splits
from prosodia_lyricist.melody_encoder.encoding import MELODY_REPRESENTATION
from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder
from prosodia_lyricist.midi import midi_melody_record


def test_learned_report_shows_all_midi_notes_without_aligning_slots():
    from prosodia_lyricist.infer import template_report
    from prosodia_lyricist.midi import midi_records
    from prosodia_lyricist.report import markdown_report

    path = Path(__file__).resolve().parents[1] / "examples" / "imagine.mid"
    lines = midi_melody_record(path)["lines"]
    for line in lines:
        line["syllables"] = [{"stress": "strong", "length": "long"}]
    original = copy.deepcopy(lines)
    report = template_report(path, lines, title="Imagine", track=0, stress_source="learned")
    assert lines == original
    assert report["note_comparison"]["lines"] == midi_records(path, title="Imagine")
    markdown = markdown_report(report)
    assert markdown.count("### MIDI notes and beat-based comparison") == 17
    # Seven raw notes must remain visible even though the first phrase has just one slot.
    phrase = markdown.split("## Phrase 1\n")[1].split("## Phrase 2\n")[0]
    assert "| 7 | 69 | 1800–1920 | 1:4.75 | <weak,short> |" in phrase
    assert "| Slot | Input | Word / IPA | Output |" in phrase
    assert "| 1 | <strong,long> | — | — |" in phrase
    assert "Comparison only" in phrase


def test_bridge_config_paths_without_bart(tmp_path):
    from prosodia_lyricist.config import load_config

    path = tmp_path / "bridge.yaml"
    path.write_text(
        "data:\n  prepared_dir: ./data\n  pretraining_manifest: ./up.csv\n"
        "model:\n  conditioning: bridge\n  melody_checkpoint: ./tower.pt\n"
        "  contrastive_fusion:\n    checkpoint: ./contrastive.pt\n"
        "training:\n  output_dir: ./runs\n"
    )
    config = load_config(path)
    assert config["data"]["pretraining_manifest"] == str(tmp_path / "up.csv")
    assert config["model"]["melody_checkpoint"] == str(tmp_path / "tower.pt")
    assert config["model"]["contrastive_fusion"]["checkpoint"] == str(tmp_path / "contrastive.pt")
    assert "pretrained" not in config["model"]


def save_contrastive(path, *, pooling="cls"):
    config = dict(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
        max_length=32,
        projection_dim=4,
        pooling=pooling,
    )
    tower = MelodyTransformerEncoder(**config)
    args = {f"melody_{k}": v for k, v in config.items() if k not in ("dropout", "projection_dim")}
    args.update(audio_pooling="note", dropout=0, projection_dim=4, manifest=path.parent / "up.csv")
    torch.save(
        {
            "melody_representation": MELODY_REPRESENTATION,
            "args": args,
            "model_state_dict": {f"melody_encoder.{k}": v for k, v in tower.state_dict().items()},
        },
        path,
    )
    return tower


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_contrastive_import_is_strict_and_preserves_note_vectors(tmp_path, pooling):
    path = tmp_path / "contrastive.pt"
    original = save_contrastive(path, pooling=pooling).eval()
    tower, cfg, provenance = load_contrastive_melody(path)
    assert cfg["pooling"] == pooling and tower.projection is None
    assert provenance["sha256"] == file_sha256(path)
    tower.eval()
    features = torch.randn(1, 4, 177)
    assert torch.equal(
        original.encode(features).note_embeddings,
        tower.encode(features, project=False).note_embeddings,
    )
    with torch.serialization.safe_globals([type(path)]):
        checkpoint = torch.load(path, weights_only=True)
    checkpoint["args"]["melody_pretrained_checkpoint"] = path.parent / "masked-notes.pt"
    torch.save(checkpoint, path)
    _, _, provenance = load_contrastive_melody(path)
    assert provenance["upstream_melody_pretrained_checkpoint"] == str(
        path.parent / "masked-notes.pt"
    )
    checkpoint["melody_representation"] = "legacy_frame"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="note-based"):
        load_contrastive_melody(path)


def write_upstream(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "dali_id", "split", "artist", "title", "line_count"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_split_reuse_and_duplicate_leakage_rejection(tmp_path):
    path = tmp_path / "up.csv"
    write_upstream(
        path,
        [
            dict(
                sample_id="x",
                dali_id="up",
                split="test",
                artist="Artist",
                title="Song",
                line_count=2,
            )
        ],
    )
    songs = {"down": {"artist": "artist", "title": "Song!"}}
    assert preserve_pretraining_splits(songs, {"down": "train"}, path) == {"down": "train"}
    assert not audit_pretraining_splits(
        {"down": {**songs["down"], "split": "train"}},
        path,
    )["conflicts"]
    path.write_text(path.read_text().replace(",test,", ",train,"))
    assert preserve_pretraining_splits(songs, {"down": "test"}, path) == {"down": "train"}
    with pytest.raises(ValueError, match="pretraining_train_in_downstream_heldout"):
        audit_pretraining_splits({"down": {**songs["down"], "split": "test"}}, path)
