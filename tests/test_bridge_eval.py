"""Bridge evaluation denominators, generation semantics, and held-out CLI."""

import json
from types import SimpleNamespace

import pytest
import torch

from prosodia_lyricist.bridge_data import (
    FIRST_LINE,
    PAIR_OFFSET,
    bridge_collate,
    encode_bridge_source,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_eval import evaluate_bridge, main
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import evaluate_templates, run_bridge_epoch
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


def line(pairs, notes=None, start=0):
    n = len(pairs) if notes is None else notes
    return {
        "syllables": [{"stress": s, "length": v} for s, v in pairs],
        "melody": {
            "midi_pitches": [60] * n,
            "onset_seconds": list(range(start, start + n)),
            "note_duration_seconds": [0.5] * n,
        },
    }


def example(lines):
    return {**encode_bridge_source(lines), "labels": encode_bridge_targets(lines, 8)}


@pytest.fixture
def bridge():
    model = ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=32),
        max_syllables=8,
        lines_per_window=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
    )
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([10.0, -10.0, -10.0, -10.0]))
    return model


def test_teacher_forced_accuracies_ignore_prefixes_padding_and_weight_by_slots():
    class FixedLogits(torch.nn.Module):
        def forward(self, **batch):
            labels = batch["labels"]
            logits = torch.zeros(*labels.shape, 4)
            predicted = [0, 0, 3, 3, 2, 0] if labels.shape[-1] == 6 else [0, 3]
            for index, pair in enumerate(predicted):
                logits[0, index, pair] = 10
            loss = torch.tensor(2.0 if len(predicted) == 6 else 4.0)
            return SimpleNamespace(logits=logits, loss=loss)

    batches = [
        {
            "labels": torch.tensor([[FIRST_LINE, 3, 4, 5, 6, -100]]),
            "syllable_counts": torch.tensor([[4]]),
            "line_counts": torch.tensor([1]),
        },
        {
            "labels": torch.tensor([[FIRST_LINE, PAIR_OFFSET + 3]]),
            "syllable_counts": torch.tensor([[1]]),
            "line_counts": torch.tensor([1]),
        },
    ]
    scores = run_bridge_epoch(FixedLogits(), batches, torch.device("cpu"))
    assert scores["loss"] == pytest.approx(2.4)
    assert scores["teacher_forced_strength_accuracy"] == pytest.approx(4 / 5)
    assert scores["teacher_forced_length_accuracy"] == pytest.approx(3 / 5)
    assert scores["teacher_forced_pair_accuracy"] == pytest.approx(2 / 5)
    assert scores["tokens"] == 5


def test_generation_bleu_and_accuracy_include_missing_extra_slots(bridge, monkeypatch):
    lines = [
        line([("strong", "long")] * 4),
        line([("strong", "long")] * 3, start=10),
        line([("strong", "short"), ("weak", "long"), ("weak", "short")], notes=2),
        line([("weak", "short")], notes=2, start=10),
    ]
    batch = bridge_collate([example(lines[:2]), example(lines[2:])])
    original = bridge.generate

    def generate(**source):
        assert "labels" not in source and "syllable_counts" not in source
        return original(**source)

    monkeypatch.setattr(bridge, "generate", generate)
    scores = evaluate_templates(bridge, [batch], torch.device("cpu"))
    assert "labels" in batch  # Evaluation must not consume the caller's batch.
    assert scores["compared_slots"] == 12
    assert scores["strength_accuracy"] == pytest.approx(8 / 12)
    assert scores["length_accuracy"] == pytest.approx(8 / 12)
    assert scores["pair_accuracy"] == pytest.approx(7 / 12)
    assert scores["exact_template_accuracy"] == 0.5
    assert scores["prosody_bleu"] == 0.25  # Exactly the unsmoothed phrase-mean BLEU-4.
    assert scores["bleu_short_phrases"] == 3
    assert scores["note_ipa_count_match_rate"] == 0.5
    assert scores["note_ipa_count_mae"] == 0.5
    assert scores["count_diagnostic_phrases"] == 4


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    pairs = [("strong", "long")] * 4
    songs = {
        "train": [{"song_id": "train-song", "lines": [line([("weak", "short")])]}],
        "valid": [{"song_id": "valid-song", "lines": [line(pairs[:2])]}],
        "test": [
            {"song_id": "test-song", "lines": [
                line(pairs), line(pairs[:3], start=10), line(pairs[:2], start=20),
            ]},
            {"song_id": "overlong-song", "lines": [line(pairs, notes=33)]},
            {"song_id": "slot-limit-song", "lines": [line(pairs, notes=9)]},
        ],
    }
    for split, records in songs.items():
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": {"include_melody": True, "stress_source": "ipa", "max_syllables": 8},
        "sha256": {f"{s}.jsonl": file_sha256(directory / f"{s}.jsonl") for s in songs},
        "songs": {
            row["song_id"]: {"split": split, "group": row["song_id"]}
            for split, records in songs.items()
            for row in records
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_cli_defaults_to_test_and_preserves_checkpoint_and_data_provenance(
    tmp_path, prepared, bridge, capsys
):
    checkpoint = tmp_path / "bridge"
    bridge.save(checkpoint)
    output = tmp_path / "out" / "eval.json"
    report = main([
        "--checkpoint", str(checkpoint), "--prepared-dir", str(prepared),
        "--device", "cpu", "--batch-size", "2", "--output", str(output), "--no-progress",
    ])
    assert json.loads(output.read_text()) == report == json.loads(capsys.readouterr().out)
    assert report["data"]["split"] == "test"
    assert report["data"]["windows"] == 3
    assert report["data"]["songs"] == 2
    assert report["data"]["skipped_note_limit_windows"] == 1
    assert report["teacher_forced"]["slots"] == 13
    assert report["teacher_forced"]["accuracy"] == dict(strength=1.0, length=1.0, combined=1.0)
    assert report["teacher_forced"]["loss"] < 0.001
    assert report["generation"]["phrases"] == 3
    assert report["generation"]["skipped_slot_limit_windows"] == 1
    assert report["generation"]["skipped_slot_limit_phrases"] == 1
    assert report["generation"]["prosody_bleu"] == pytest.approx(1 / 3)
    assert report["generation"]["accuracy"]["combined"] == 1.0
    assert report["generation"]["count_diagnostic_phrases"] == 4
    assert report["checkpoint_weights_sha256"] == file_sha256(checkpoint / "bridge_weights.pt")
    valid = evaluate_bridge(checkpoint, prepared, split="valid", device="cpu", limit=1)
    assert valid["teacher_forced"]["slots"] == 2
    assert valid["data"]["limit_windows"] == 1
    assert valid["generation"]["prosody_bleu"] == 0.0
    (prepared / "test.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        evaluate_bridge(checkpoint, prepared, device="cpu")


@pytest.mark.parametrize("kwargs", [
    {"split": "train"}, {"limit": 0}, {"batch_size": 0}, {"num_workers": -1},
    {"precision": "bf16", "device": "cpu"},
])
def test_invalid_settings_fail_before_loading_checkpoint(kwargs):
    with pytest.raises(ValueError):
        evaluate_bridge("unused", "unused", **kwargs)
