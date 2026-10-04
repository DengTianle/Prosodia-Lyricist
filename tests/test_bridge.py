"""Learned templates: no lyric leakage, variable syllable counts, reusable lyric decoder."""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from prosodia_lyricist.bridge_data import (
    FIRST_LINE,
    PAIR_OFFSET,
    BridgeDataset,
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_source,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import evaluate_templates, run_bridge_epoch
from prosodia_lyricist.features import encode_source
from prosodia_lyricist.melody_checkpoint import file_sha256, load_contrastive_melody
from prosodia_lyricist.melody_data import audit_pretraining_splits, preserve_pretraining_splits
from prosodia_lyricist.melody_encoder.encoding import MELODY_REPRESENTATION
from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder
from prosodia_lyricist.midi import midi_melody_record
from prosodia_lyricist.model import ProsodyBart


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


@pytest.fixture
def lines(record):
    first = copy.deepcopy(record)
    first["melody"] = {
        "midi_pitches": [60, 62, 64, 64],
        "onset_seconds": [0, 0.5, 1.5, 2],
        "note_duration_seconds": [0.5, 0.5, 0.5, 2],
    }
    second = {
        "syllables": [{"stress": "weak", "length": "long"}],
        "melody": {"midi_pitches": [67], "onset_seconds": [4], "note_duration_seconds": [1]},
    }
    return [first, second]


@pytest.fixture
def bridge():
    return ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=32),
        max_syllables=8,
        lines_per_window=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
    )


def example(lines):
    return {**encode_bridge_source(lines), "labels": encode_bridge_targets(lines, 8)}


def test_source_target_separation_and_template_decoder_compatibility(lines, tokenizer):
    source = encode_bridge_source(lines)
    changed = copy.deepcopy(lines)
    for line in changed:
        line.update(text="secret", words=[], syllables=[])
    other = encode_bridge_source(changed)
    np.testing.assert_array_equal(source["melody_features"], other["melody_features"])
    assert source["note_line_ids"] == other["note_line_ids"] == [1, 1, 1, 1, 2]
    labels = encode_bridge_targets(lines, 8)
    assert labels == [FIRST_LINE, 6, 3, 4, FIRST_LINE + 1, 5]
    reconstructed = decode_bridge_tokens(labels)
    batch = bridge_collate([example(lines), example(lines[:1])])
    assert batch["syllable_counts"].tolist() == [[3, 1], [3, 0]]
    assert encode_source({"lines": reconstructed}, tokenizer, 8) == encode_source(
        {"lines": lines}, tokenizer, 8
    )


def test_gradients_freezing_causality_padding_and_self_contained_reload(tmp_path, lines, bridge):
    batch = bridge_collate([example(lines), example(lines[:1])])
    bridge.set_melody_trainable(False)
    bridge.train()
    assert not bridge.melody_encoder.training
    bridge(**batch).loss.backward()
    assert bridge.adapter[1].weight.grad.abs().sum() > 0
    assert bridge.strength_head.weight.grad.abs().sum() > 0
    assert bridge.length_head.weight.grad.abs().sum() > 0
    assert not hasattr(bridge, "count_head")
    assert all(p.grad is None for p in bridge.melody_encoder.parameters())
    bridge.zero_grad(set_to_none=True)
    bridge.set_melody_trainable(True)
    bridge(**batch).loss.backward()
    assert bridge.melody_encoder.input_projection[0].weight.grad.abs().sum() > 0
    bridge.eval()
    output = bridge(**batch)
    future = {**batch, "labels": batch["labels"].clone()}
    future["labels"][0, 1] = 3
    after = bridge(**future)
    assert set(output.loss_components) == {"strength", "length"}
    targets = (batch["labels"] - PAIR_OFFSET).masked_fill(batch["labels"].ge(FIRST_LINE), -100)
    targets = targets.masked_fill(batch["labels"].eq(-100), -100)
    slots = targets.ne(-100)
    strength_targets = (targets // 2).masked_fill(~slots, -100)
    length_targets = (targets % 2).masked_fill(~slots, -100)
    expected_strength = torch.nn.functional.cross_entropy(
        output.strength_logits.flatten(0, 1), strength_targets.flatten()
    )
    expected_length = torch.nn.functional.cross_entropy(
        output.length_logits.flatten(0, 1), length_targets.flatten()
    )
    assert torch.equal(output.loss_components["strength"], expected_strength)
    assert torch.equal(output.loss_components["length"], expected_length)
    assert torch.equal(output.loss, expected_strength + expected_length)
    assert torch.equal(output.logits[:, :2], after.logits[:, :2])
    assert not torch.equal(output.logits[:, 2:], after.logits[:, 2:])
    padded = {
        key: torch.nn.functional.pad(value, (0, 0, 0, 3), value=99)
        if key == "melody_features"
        else value
        if key in ("line_counts", "syllable_counts")
        else torch.nn.functional.pad(value, (0, 3), value=-100 if key == "labels" else 0)
        for key, value in batch.items()
    }
    assert torch.allclose(output.loss, bridge(**padded).loss, atol=1e-6)
    bridge.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    assert torch.equal(output.logits, restored(**batch).logits)
    # Saved model contains the complete melody trunk; no upstream path is required.
    assert any(key.startswith("melody_encoder.") for key in restored.state_dict())


def test_note_counts_fix_slots_and_line_ids_in_padded_batches(lines, bridge):
    batch = bridge_collate([encode_bridge_source(lines), encode_bridge_source(lines[:1])])
    bridge.eval()
    with torch.no_grad():
        for head in (bridge.strength_head, bridge.length_head):
            head.weight.zero_()
            head.bias.copy_(torch.tensor([1000.0, -1000.0]))
    result = bridge.generate(**batch)
    assert result.syllable_counts.tolist() == [[4, 1], [4, 0]]
    assert len(decode_bridge_tokens(result.sequences[1].tolist())) == 1
    assert not hasattr(result, "predicted_syllable_counts")
    first = result.sequences[0].tolist()
    assert first == [FIRST_LINE] + [PAIR_OFFSET] * 4 + [FIRST_LINE + 1, PAIR_OFFSET]
    assert bridge.strength_head.out_features == bridge.length_head.out_features == 2
    assert not hasattr(bridge, "output")
    assert "line_end" not in bridge.vocabulary and "eos" not in bridge.vocabulary


@pytest.mark.parametrize("strength,length", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("use_ipa_counts", [False, True])
def test_two_heads_feed_back_both_labels_and_encode_template_input(
    strength, length, use_ipa_counts, bridge, lines, tokenizer, monkeypatch
):
    with torch.no_grad():
        for head, selected in ((bridge.strength_head, strength), (bridge.length_head, length)):
            head.weight.zero_()
            head.bias.fill_(-10)
            head.bias[selected] = 10
    prefixes = []
    original_decode = bridge.decode

    def decode(tokens, *args):
        prefixes.append(tokens.clone())
        return original_decode(tokens, *args)

    monkeypatch.setattr(bridge, "decode", decode)
    source = bridge_collate([encode_bridge_source(lines)])
    if use_ipa_counts:
        source["syllable_counts"] = torch.tensor([[len(line["syllables"]) for line in lines]])
    result = bridge.eval().generate(**source)
    predicted = decode_bridge_tokens(result.sequences[0].tolist())
    pair_token = PAIR_OFFSET + 2 * strength + length
    assert prefixes[1][0, -1] == pair_token  # Both predictions enter the next step.
    expected_pair = {
        "stress": ("strong", "weak")[strength], "length": ("long", "short")[length]
    }
    assert all(s == expected_pair for line in predicted for s in line["syllables"])
    encoded = encode_source({"lines": predicted}, tokenizer, 8)
    active_lengths = [value for value in encoded["length_ids"] if value]
    slots = 4 if use_ipa_counts else 5
    assert active_lengths == [length + 1] * slots
    stress_token = tokenizer.convert_tokens_to_ids(f"<{expected_pair['stress']}>")
    assert encoded["input_ids"].count(stress_token) == slots


def test_training_and_validation_use_ipa_skeleton(
    monkeypatch, lines, bridge
):
    batch = bridge_collate([example(lines)])
    broken = {**batch, "syllable_counts": torch.tensor([[2, 2]])}
    with pytest.raises(ValueError, match="skeleton"):
        bridge(**broken)
    original = bridge.generate
    called = []

    def generate(**kwargs):
        assert "labels" not in kwargs
        assert torch.equal(kwargs["syllable_counts"], batch["syllable_counts"])
        called.append(True)
        return original(**kwargs)

    monkeypatch.setattr(bridge, "generate", generate)
    scores = evaluate_templates(bridge, [batch], torch.device("cpu"))
    assert called and scores["note_ipa_count_match_rate"] == 0.5
    assert scores["note_ipa_count_mae"] == 0.5
    assert scores["slot_count_source"] == "ipa"
    assert scores["skipped_slot_limit_windows"] == 0


def test_token_weighted_accumulation_and_partial_group(lines, bridge):
    joint = copy.deepcopy(bridge)
    a, b = example(lines), example(lines[:1])
    results = []
    for model, batches, accumulation in (
        (bridge, [bridge_collate([a]), bridge_collate([b])], 3),
        (joint, [bridge_collate([a, b])], 1),
    ):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        results.append(
            run_bridge_epoch(
                model,
                batches,
                torch.device("cpu"),
                optimizer=optimizer,
                scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                accumulation_steps=accumulation,
                gradient_clip=None,
            )
        )
    assert results[0]["optimizer_steps"] == 1
    for a, b in zip(bridge.parameters(), joint.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_saved_limits_and_checkpoint_validation(tmp_path, lines, bridge):
    bridge.max_notes = 4
    bridge.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    assert restored.max_notes == 4
    with pytest.raises(ValueError, match="note limit"):
        restored.generate(**bridge_collate([encode_bridge_source(lines)]))
    metadata_path = tmp_path / "bridge.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["prosody_rules"] = "incompatible"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature rules"):
        ProsodyBridge.load(tmp_path)
    metadata["format_version"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="retrain"):
        ProsodyBridge.load(tmp_path)


def test_v2_checkpoint_reuses_prosody_weights_without_count_head(tmp_path, lines, bridge):
    bridge = ProsodyBridge(
        bridge.melody_config, **bridge.decoder_config, max_syllables=8,
        lines_per_window=2, output_design="joint_pair",
    )
    bridge.eval().save(tmp_path)
    batch = bridge_collate([encode_bridge_source(lines)])
    expected = bridge.generate(**batch)
    metadata_path = tmp_path / "bridge.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(format_version=2, target_scheme="line_skeleton_v2")
    metadata.pop("inference_count_source")
    metadata_path.write_text(json.dumps(metadata))
    state = bridge.state_dict()
    # Recreate the exact v2-only parameters; a biased count head cannot affect v3 slots.
    head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 8))
    state.update({f"count_head.{key}": value for key, value in head.state_dict().items()})
    torch.save(state, tmp_path / "bridge_weights.pt")
    restored = ProsodyBridge.load(tmp_path).eval()
    assert not hasattr(restored, "count_head")
    assert torch.equal(restored.generate(**batch).sequences, expected.sequences)
    for key, value in bridge.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    restored.save(tmp_path / "converted")
    assert json.loads((tmp_path / "converted" / "bridge.json").read_text())["format_version"] == 3
    state.pop("output.weight")
    torch.save(state, tmp_path / "bridge_weights.pt")
    with pytest.raises(RuntimeError, match="output.weight"):
        ProsodyBridge.load(tmp_path)


def test_v3_checkpoint_preserves_joint_distribution_and_note_lengths(tmp_path, lines, bridge):
    legacy = ProsodyBridge(
        bridge.melody_config, **bridge.decoder_config, max_syllables=8,
        lines_per_window=2, output_design="joint_pair",
    ).eval()
    # The joint argmax differs from the product of marginal argmaxes.
    with torch.no_grad():
        legacy.output.weight.zero_()
        legacy.output.bias.copy_(torch.tensor([0.0, 1.0, 0.9, -10.0]))
    legacy.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    assert restored.output_design == "joint_pair"
    batch = bridge_collate([example(lines)])
    assert torch.equal(legacy(**batch).logits, restored(**batch).logits)
    assert set(restored(**batch).loss_components) == {"prosody"}
    source = bridge_collate([encode_bridge_source(lines)])
    predicted = restored.generate(**source)
    assert predicted.syllable_counts.tolist() == [[4, 1]]
    pairs = decode_bridge_tokens(predicted.sequences[0].tolist())
    assert pairs[0]["syllables"] == [{"stress": "strong", "length": "short"}] * 4


def test_v4_checkpoint_rejects_wrong_heads_and_label_order(tmp_path, bridge):
    bridge.save(tmp_path)
    path = tmp_path / "bridge.json"
    metadata = json.loads(path.read_text())
    assert metadata["format_version"] == 4
    metadata["output_labels"]["strength"].reverse()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature rules"):
        ProsodyBridge.load(tmp_path)
    bridge.save(tmp_path)
    state = bridge.state_dict()
    state.pop("length_head.weight")
    torch.save(state, tmp_path / "bridge_weights.pt")
    with pytest.raises(RuntimeError, match="length_head.weight"):
        ProsodyBridge.load(tmp_path)


def test_inference_limits_notes_and_validation_limits_ipa_counts(lines, bridge):
    bridge.max_syllables = 3
    batch = bridge_collate([example(lines), example(lines[1:])])
    with pytest.raises(ValueError, match="slot limit 3; no truncation"):
        bridge.eval().generate(**bridge_collate([encode_bridge_source(lines)]))
    scores = evaluate_templates(bridge, [batch], torch.device("cpu"))
    assert scores["phrases"] == 3 and scores["skipped_slot_limit_windows"] == 0
    bridge.max_syllables = 2
    scores = evaluate_templates(bridge, [batch], torch.device("cpu"))
    assert scores["phrases"] == 1
    assert scores["skipped_slot_limit_windows"] == 1
    assert scores["skipped_slot_limit_phrases"] == 2
    assert scores["note_ipa_count_match_rate"] == pytest.approx(2 / 3)
    skipped = evaluate_templates(bridge, [bridge_collate([example(lines)])], torch.device("cpu"))
    assert skipped["phrases"] == 0 and skipped["exact_template_accuracy"] is None


def test_removed_skeleton_option_is_rejected(monkeypatch):
    from prosodia_lyricist.infer import infer, main

    with pytest.raises(TypeError, match="bridge_skeleton"):
        infer("unused", "unused.mid", bridge_skeleton={"lines": []})
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--midi",
            "unused.mid",
            "--template-only",
            "--bridge-skeleton",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_note_skeleton_spans_encoder_windows_and_tail(tmp_path, bridge):
    from prosodia_lyricist.bridge_infer import predict_templates

    midi = pytest.importorskip("miditoolkit")
    song = midi.MidiFile(ticks_per_beat=480)
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60 + i, i * 960, i * 960 + 240) for i in range(3)]
    song.instruments.append(instrument)
    song.markers = [midi.Marker("end 0", 480), midi.Marker("end 1", 1440)]
    path = tmp_path / "song.mid"
    song.dump(str(path))
    bridge.save(tmp_path / "bridge")
    records, metadata = predict_templates(tmp_path / "bridge", path)
    assert metadata["encoder_windows"] == 2
    assert metadata["syllable_counts"] == metadata["note_counts"] == [1, 1, 1]
    assert metadata["count_sources"] == ["notes"] * 3
    assert "predicted_syllable_counts" not in metadata
    assert [line["line_id"] for line in records] == [0, 1, 2]
    assert records[-1]["unmarked_tail"]


def test_bridge_config_paths_without_bart(tmp_path):
    from prosodia_lyricist.config import load_config

    path = tmp_path / "bridge.yaml"
    path.write_text(
        "data:\n  prepared_dir: ./data\n  pretraining_manifest: ./up.csv\n"
        "model:\n  conditioning: bridge\n  melody_checkpoint: ./tower.pt\n"
        "training:\n  output_dir: ./runs\n"
    )
    config = load_config(path)
    assert config["data"]["pretraining_manifest"] == str(tmp_path / "up.csv")
    assert config["model"]["melody_checkpoint"] == str(tmp_path / "tower.pt")
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
    assert preserve_pretraining_splits(songs, {"down": "train"}, path) == {"down": "test"}
    with pytest.raises(ValueError, match="crosses splits"):
        audit_pretraining_splits({"down": {**songs["down"], "split": "train"}}, path)


def test_window_tail_note_limits_and_checksums(tmp_path, lines):
    path = tmp_path / "train.jsonl"
    third = copy.deepcopy(lines[1])
    third["melody"]["onset_seconds"] = [6]
    path.write_text(json.dumps({"song_id": "s", "lines": [*lines, third]}) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": {"include_melody": True, "stress_source": "ipa", "max_syllables": 8},
                "songs": {"s": {"split": "train", "group": "s"}},
                "sha256": {"train.jsonl": file_sha256(path)},
            }
        )
    )
    ds = BridgeDataset(tmp_path, "train", lines_per_window=2, max_notes=5)
    assert len(ds) == 2 and [e["line_count"] for e in ds] == [2, 1]
    limited = BridgeDataset(tmp_path, "train", lines_per_window=2, max_notes=4)
    assert len(limited) == 1 and limited.skipped == 1
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest"):
        BridgeDataset(tmp_path, "train", lines_per_window=2, max_notes=5)


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("history_mask_probability", [0.0, 0.5])
def test_prepare_train_and_reload_bridge(
    tmp_path, monkeypatch, annotation, pretrained, history_mask_probability
):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.prepare import prepare
    from prosodia_lyricist.train import train

    monkeypatch.setattr(ipa, "backend", lambda: None)
    monkeypatch.setattr(
        ipa,
        "parse_words",
        lambda text: [
            {"text": word, "syllables": [ipa.syllable_features("ˈeɪ")]} for word in text.split()
        ],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index in range(20):
        song = copy.deepcopy(annotation)
        song["info"].update(id=f"s-{index}", title=f"Song {index}")
        for note in song["annotations"]["annot"]["notes"]:
            note["freq"] = [440.0]
        (raw / f"{index}.json").write_text(json.dumps(song))
        rows.append(
            dict(
                sample_id=f"sample-{index}",
                dali_id=f"s-{index}",
                split="train" if index < 10 else "val",
                artist="Artist",
                title=f"Song {index}",
                line_count=2,
            )
        )
    upstream = tmp_path / "up.csv"
    write_upstream(upstream, rows)
    checkpoint = tmp_path / "contrastive.pt"
    original = save_contrastive(checkpoint)
    config = {
        "data": dict(
            dali_dir=str(raw),
            prepared_dir=str(tmp_path / "data"),
            language="english",
            min_ncc=0,
            max_syllables=8,
            valid_fraction=0.2,
            test_fraction=0.2,
            seed=1234,
            stress_source="ipa",
            include_melody=True,
            lines_per_window=2,
            pretraining_manifest=str(upstream) if pretrained else None,
        ),
        "model": dict(
            conditioning="bridge",
            melody_checkpoint=str(checkpoint) if pretrained else None,
            max_notes=32,
        ),
        "training": dict(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            seed=1234,
            epochs=1,
            batch_size=2,
            patience=2,
            learning_rate=0.001,
            melody_learning_rate=0.0001,
            melody_unfreeze_epoch=None,
            history_mask_probability=history_mask_probability,
            warmup_steps=0,
            schedule="constant_after_warmup",
        ),
    }
    manifest = prepare(config)
    if pretrained:
        assert manifest["pretraining_split_audit"]["shared_songs"] == 20
        assert manifest["songs"]["s-19"]["split"] == "valid"
    output = train(config, smoke_test=True)
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert metrics["train"]["loss"] > 0
    assert metrics["train"]["history_mask_probability"] == history_mask_probability
    assert metrics["valid"]["history_mask_probability"] == 0
    run = json.loads((output / "run.json").read_text())
    assert run["config"]["training"]["history_mask_probability"] == history_mask_probability
    assert 0 <= metrics["valid"]["generation"]["note_ipa_count_match_rate"] <= 1
    assert set(metrics["train"]["components"]) == {"strength", "length"}
    assert metrics["train"]["loss"] == pytest.approx(sum(metrics["train"]["components"].values()))
    restored = ProsodyBridge.load(output / "best")
    ds = BridgeDataset(tmp_path / "data", "valid", lines_per_window=2, max_notes=32)
    assert len(ds[0]["note_line_ids"]) == 4  # Retains the DALI melisma note.
    assert len(decode_bridge_tokens(ds[0]["labels"])[0]["syllables"]) == 2
    if pretrained:
        for key, value in restored.melody_encoder.state_dict().items():
            assert torch.equal(value, original.state_dict()[key])
        upstream.write_text(upstream.read_text() + "\n")
        with pytest.raises(ValueError, match="unchanged"):
            train(config, smoke_test=True)
    else:
        with pytest.raises(ValueError, match="melody_checkpoint"):
            train(config)


def test_combined_inference_and_reports(tmp_path, monkeypatch, bridge, tiny_model, tokenizer):
    from prosodia_lyricist import generation, ipa
    from prosodia_lyricist.infer import infer, main

    midi = pytest.importorskip("miditoolkit")
    monkeypatch.setattr(
        ipa,
        "parse_words",
        lambda text: [
            {"text": word, "syllables": [ipa.syllable_features("ˈeɪ")]} for word in text.split()
        ],
    )
    model = ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0)
    checkpoint = tmp_path / "template"
    model.save(checkpoint, tokenizer)
    (checkpoint / "run.json").write_text(
        json.dumps(
            {
                "data_schema_version": 4,
                "config": {
                    "data": {"stress_source": "ipa"},
                    "model": {"max_source_length": 64, "max_target_length": 64},
                },
            }
        )
    )
    bridge.save(tmp_path / "bridge")
    original_decode = ProsodyBridge.decode

    def decode(self, tokens, memory, mask, skeleton):
        values = original_decode(self, tokens, memory, mask, skeleton)
        for logits in (values.strength_logits, values.length_logits):
            logits[:, -1] = -1000
            logits[:, -1, 0] = 1000
        return values

    monkeypatch.setattr(ProsodyBridge, "decode", decode)
    song = midi.MidiFile(ticks_per_beat=480)
    instrument = midi.Instrument(0)
    instrument.notes = [
        midi.Note(80, 60, 0, 240),
        midi.Note(80, 62, 240, 480),
        midi.Note(80, 64, 960, 1440),
    ]
    song.instruments.append(instrument)
    song.markers.append(midi.Marker("end", 480))
    song.tempo_changes = [midi.TempoChange(120, 0), midi.TempoChange(60, 960)]
    song.time_signature_changes = [midi.TimeSignature(3, 4, 0)]
    path = tmp_path / "song.mid"
    song.dump(str(path))
    raw = midi_melody_record(path)
    assert len(raw["lines"]) == 2 and raw["lines"][1]["unmarked_tail"]
    assert raw["lines"][1]["melody"]["note_duration_seconds"] == pytest.approx([1])
    weights_hash = file_sha256(checkpoint / "weights.pt")

    def script():
        sequence = iter(
            tokenizer.convert_tokens_to_ids(
                ["<s>", "hello", "<word_end>", ".", "world", "<word_end>", ".", "</s>"]
            )
        )
        monkeypatch.setattr(
            generation,
            "sample",
            lambda logits, **kwargs: next(sequence) if logits.numel() == len(tokenizer) else 2,
        )

    script()
    result = infer(
        checkpoint,
        path,
        bridge_checkpoint=tmp_path / "bridge",
        device="cpu",
        top_k=1,
        max_new_tokens=16,
        return_report=True,
        reference_lines=["hello", "world"],
    )
    assert result["lyrics"] == ["hello", "world"]
    assert result["stress_source"] == "learned"
    assert result["bridge"]["note_counts"] == [2, 1]
    assert result["bridge"]["syllable_counts"] == [2, 1]
    assert result["bridge"]["count_sources"] == ["notes", "notes"]
    assert result["note_comparison"]["method"] is None
    assert "only 4/4" in result["note_comparison"]["warning"]
    assert result["note_comparison"]["lines"][0]["syllables"][0]["note"]["pitch"] == 60
    assert result["encoded_source"] == encode_source(
        {"title": "", "lines": result["template"]}, tokenizer, 8
    )
    assert result["reference_perplexity"]["perplexity"] > 0
    assert result["decoder_explanation"]["prosody_correction"]
    script()
    other = infer(
        checkpoint,
        path,
        bridge_checkpoint=tmp_path / "bridge",
        device="cpu",
        top_k=1,
        max_new_tokens=16,
        return_report=True,
        reference_lines=["song", "light"],
    )
    assert other["encoded_source"] == result["encoded_source"]
    assert other["generated_token_ids"] == result["generated_token_ids"]
    assert file_sha256(checkpoint / "weights.pt") == weights_hash
    with pytest.raises(ValueError, match="heuristic option"):
        infer(checkpoint, path, bridge_checkpoint=tmp_path / "bridge", stress_source="supplement")
    prefix = tmp_path / "report"
    explanations = tmp_path / "explanations.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--checkpoint",
            str(checkpoint),
            "--bridge-checkpoint",
            str(tmp_path / "bridge"),
            "--midi",
            str(path),
            "--device",
            "cpu",
            "--report-prefix",
            str(prefix),
            "--explanations",
            str(explanations),
            "--max-new-tokens",
            "16",
        ],
    )
    script()
    main()
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["bridge"]["syllable_counts"] == [2, 1]
    assert report["bridge"]["count_sources"] == ["notes", "notes"]
    assert report["decoder_explanation"] == json.loads(explanations.read_text())
    assert "one prosody slot per note" in prefix.with_suffix(".md").read_text()
    assert "| 1 | 60 | 0–240 | — | — |" in prefix.with_suffix(".md").read_text()
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--bridge-checkpoint",
            str(tmp_path / "bridge"),
            "--midi",
            str(path),
            "--device",
            "cpu",
            "--template-only",
            "--report-prefix",
            str(tmp_path / "templates"),
        ],
    )
    main()
    assert json.loads((tmp_path / "templates.json").read_text())["bridge"]["syllable_counts"] == [
        2,
        1,
    ]
