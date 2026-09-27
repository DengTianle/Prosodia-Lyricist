"""Direct melody transfer preserves song context and explainable decoder behavior."""

import copy
import csv
import json

import numpy as np
import pytest
import torch

from prosodia_lyricist.features import TARGET_KEYS, WORD_END, encode_targets
from prosodia_lyricist.melody_checkpoint import load_contrastive_melody
from prosodia_lyricist.melody_data import (
    MelodyCollator,
    MelodySongDataset,
    encode_melody_source,
    preserve_pretraining_splits,
    song_units,
)
from prosodia_lyricist.melody_encoder.encoding import MELODY_REPRESENTATION, encode_note_sequence
from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder
from prosodia_lyricist.melody_infer import midi_melody_record
from prosodia_lyricist.melody_model import MelodyBart
from prosodia_lyricist.melody_train import run_melody_epoch
from prosodia_lyricist.model import LOSS_NAMES, ProsodyBart
from prosodia_lyricist.prepare import prepare


@pytest.fixture
def melody_song(record):
    first = copy.deepcopy(record)
    first.update(
        words=[{"text": "hello", "syllable_count": 2}, {"text": "world", "syllable_count": 1}],
        paragraph_id=0,
        melody={
            "midi_pitches": [60, 62, 64, 64],
            "onset_seconds": [0, 0.5, 1.5, 2],
            "note_duration_seconds": [0.5, 0.5, 0.5, 2],
        },
    )
    second = {
        "text": "light",
        "words": [{"text": "light", "syllable_count": 1}],
        "syllables": [{"stress": "strong", "length": "long"}],
        "paragraph_id": 1,
        "melody": {"midi_pitches": [67], "onset_seconds": [4], "note_duration_seconds": [1]},
    }
    return {"title": "a song", "song_id": "song-a", "lines": [first, second]}


@pytest.fixture
def direct_model(tiny_model):
    template = ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0)
    return MelodyBart.from_template(
        template,
        melody_config={
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dim_feedforward": 16,
            "dropout": 0,
            "pooling": "cls",
        },
    )


def example(song, tokenizer, **kwargs):
    return {**encode_melody_source(song, tokenizer, **kwargs), **encode_targets(song, tokenizer, 8)}


def test_source_window_packing_and_no_lyric_leakage(melody_song, tokenizer):
    source = encode_melody_source(melody_song, tokenizer)
    assert [len(w) for w in source["windows"]] == [4, 1]
    changed = copy.deepcopy(melody_song)
    for line in changed["lines"]:
        line.update(text="secret", words=[], syllables=[])
    other = encode_melody_source(changed, tokenizer)
    assert source["input_ids"] == other["input_ids"]
    assert source["positions"] == other["positions"]
    for a, b in zip(source["windows"], other["windows"], strict=True):
        np.testing.assert_array_equal(a, b)
    batch = MelodyCollator(tokenizer.pad_token_id)(
        [
            example(melody_song, tokenizer),
            example({**melody_song, "lines": melody_song["lines"][:1]}, tokenizer),
        ]
    )
    assert batch["input_ids"].shape[0] == 2  # Songs, not three independent windows.
    assert batch["melody_features"].shape[:2] == (3, 4)
    mask = batch["melody_attention_mask"].bool()
    positions = batch["note_positions"][mask]
    assert positions.unique().numel() == 9
    assert batch["attention_mask"].reshape(-1)[positions].all()
    assert batch["labels"][0].eq(tokenizer.bos_token_id).sum() == 1
    assert batch["labels"][1, -1] == -100
    grouped = encode_melody_source(melody_song, tokenizer, lines_per_window=2)
    assert len(grouped["windows"]) == 1
    np.testing.assert_array_equal(
        grouped["windows"][0],
        encode_note_sequence(
            np.array([60, 62, 64, 64, 67]),
            np.array([0, 0.5, 1.5, 2, 4]),
            np.array([0.5, 0.5, 0.5, 2, 1]),
        ),
    )
    # The short final window is retained rather than dropping the tail.
    assert len(encode_melody_source(melody_song, tokenizer, lines_per_window=3)["windows"]) == 1


def test_four_stream_gradients_freezing_padding_and_reload(
    tmp_path,
    melody_song,
    tokenizer,
    direct_model,
):
    batch = MelodyCollator(tokenizer.pad_token_id)([example(melody_song, tokenizer)])
    direct_model.set_trainable(melody=False, decoder=False)
    direct_model.train()
    assert not direct_model.bart.training and not direct_model.melody_encoder.training
    before = {k: p.detach().clone() for k, p in direct_model.named_parameters()}
    output = direct_model(**batch)
    assert tuple(output.loss_components) == LOSS_NAMES
    assert torch.allclose(output.loss, sum(output.loss_components.values()))
    output.loss.backward()
    assert direct_model.adapter[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in direct_model.bart.parameters())
    assert all(p.grad is None for p in direct_model.melody_encoder.parameters())
    optimizer = torch.optim.AdamW(direct_model.parameters(), lr=1e-3)
    optimizer.step()
    for k, p in direct_model.named_parameters():
        if not k.startswith("adapter."):
            assert torch.equal(before[k], p)
    direct_model.zero_grad(set_to_none=True)
    direct_model.set_trainable(melody=True, decoder=True)
    direct_model(**batch).loss.backward()
    for head in direct_model.prosody_heads.values():
        assert head.weight.grad.abs().sum() > 0
    assert direct_model.decoder_projection.weight.grad.abs().sum() > 0
    assert direct_model.melody_encoder.input_projection[0].weight.grad.abs().sum() > 0
    direct_model.eval()
    output = direct_model(**batch)
    padded = dict(batch)
    padded["melody_features"] = torch.nn.functional.pad(
        batch["melody_features"], (0, 0, 0, 3), value=99
    )
    padded["melody_attention_mask"] = torch.nn.functional.pad(
        batch["melody_attention_mask"], (0, 3)
    )
    padded["note_positions"] = torch.nn.functional.pad(batch["note_positions"], (0, 3), value=-1)
    for key in TARGET_KEYS:
        padded[key] = torch.nn.functional.pad(batch[key], (0, 3), value=-100)
    assert torch.allclose(output.loss, direct_model(**padded).loss, atol=1e-6)
    direct_model.save(tmp_path, tokenizer)
    restored = ProsodyBart.load(tmp_path).eval()
    assert isinstance(restored, MelodyBart)
    restored_output = restored(**batch)
    # Frozen eval can select a different fused Transformer kernel than grad-enabled eval.
    assert torch.allclose(output.logits, restored_output.logits, atol=1e-6)
    for key, value in direct_model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert restored.source_config == direct_model.source_config
    # Training targets remain causally shifted with direct melody conditioning.
    future = dict(batch, stress_labels=batch["stress_labels"].clone())
    future["stress_labels"][0, 2] = 2
    later = restored(**future).logits
    assert torch.equal(restored_output.logits[:, :3], later[:, :3])
    assert not torch.equal(restored_output.logits[:, 3:], later[:, 3:])


def test_template_initialization_preserves_all_weights(tiny_model):
    template = ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0)
    model = MelodyBart.from_template(
        template,
        melody_config={
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dim_feedforward": 16,
        },
    )
    for key, tensor in template.state_dict().items():
        assert torch.equal(model.state_dict()[key], tensor)
    with pytest.raises(ValueError, match="four-stream"):
        MelodyBart.from_template(tiny_model)


def test_real_cached_generation_and_correction(monkeypatch, direct_model, melody_song, tokenizer):
    source = MelodyCollator(tokenizer.pad_token_id)([encode_melody_source(melody_song, tokenizer)])
    direct_model.eval()
    sequence = iter(
        [
            tokenizer.bos_token_id,
            tokenizer.convert_tokens_to_ids("hello"),
            tokenizer.convert_tokens_to_ids(WORD_END),
            tokenizer.eos_token_id,
        ]
    )
    original = direct_model.bart.forward
    feedback = []
    embed = direct_model.embed_target

    def capture(*args):
        feedback.append([int(x[0, 0]) for x in args])
        return embed(*args)

    def scripted(**kwargs):
        result = original(**kwargs)  # Exercise actual encoder and decoder KV cache.
        result.logits.fill_(-1000)
        result.logits[:, -1, next(sequence)] = 1000
        return result

    monkeypatch.setattr(direct_model, "embed_target", capture)
    monkeypatch.setattr(direct_model.bart, "forward", scripted)
    syllables = melody_song["lines"][0]["syllables"][:2]
    result = direct_model.generate(
        **source,
        tokenizer=tokenizer,
        max_new_tokens=8,
        pronunciation=lambda _: [{"syllables": syllables}],
    )
    assert result.explanations[0]["text"] == "hello"
    assert result.explanations[0]["completed"]
    assert feedback[-1][1:] == [2, 1, 1]
    assert result.syllable_ids[0, 3] == 2


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_import_current_checkpoint_and_pooling(tmp_path, pooling):
    tower = MelodyTransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        projection_dim=4,
        dropout=0,
        pooling=pooling,
    )
    checkpoint = {
        "melody_representation": MELODY_REPRESENTATION,
        "args": {
            "audio_pooling": "note",
            "melody_d_model": 8,
            "melody_num_layers": 1,
            "melody_num_heads": 2,
            "melody_dim_feedforward": 16,
            "projection_dim": 4,
            "dropout": 0,
            "melody_pooling": pooling,
        },
        "model_state_dict": {f"melody_encoder.{k}": v for k, v in tower.state_dict().items()},
    }
    path = tmp_path / "encoder.pt"
    torch.save(checkpoint, path)
    restored, config, _ = load_contrastive_melody(path)
    assert config["pooling"] == pooling and restored.projection is None
    tower.eval()
    restored.eval()
    inputs = torch.randn(1, 4, 177)
    assert torch.equal(
        tower.encode(inputs).note_embeddings, restored.encode(inputs, project=False).note_embeddings
    )
    # Changing pooling on identical weights changes only the pooled summary.
    other = copy.deepcopy(restored)
    other.pooling = "mean" if pooling == "cls" else "cls"
    assert torch.equal(
        restored.encode(inputs, project=False).note_embeddings,
        other.encode(inputs, project=False).note_embeddings,
    )
    checkpoint["melody_representation"] = "legacy_frames"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="note-based"):
        load_contrastive_melody(path)


def test_accumulation_matches_joint_batch_and_partial_group(direct_model, melody_song, tokenizer):
    a = example(melody_song, tokenizer)
    b = example({**melody_song, "lines": melody_song["lines"][:1]}, tokenizer)
    collator = MelodyCollator(tokenizer.pad_token_id)
    joint = copy.deepcopy(direct_model)
    # Both models have zero dropout. Token weighting accounts for unequal lengths.
    opt = torch.optim.SGD(direct_model.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(joint.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1)
    result = run_melody_epoch(
        direct_model,
        [collator([a]), collator([b])],
        torch.device("cpu"),
        optimizer=opt,
        scheduler=scheduler,
        accumulation_steps=3,
        gradient_clip=None,
    )
    run_melody_epoch(
        joint,
        [collator([a, b])],
        torch.device("cpu"),
        optimizer=opt2,
        scheduler=scheduler2,
        gradient_clip=None,
    )
    assert result["optimizer_steps"] == 1 and result["batches"] == 2
    for left, right in zip(direct_model.parameters(), joint.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6)


def test_paragraphs_and_lengths(tmp_path, tokenizer, melody_song):
    assert [len(s["lines"]) for s in song_units(melody_song, "paragraph")] == [1, 1]
    row = json.dumps(melody_song) + "\n"
    path = tmp_path / "train.jsonl"
    path.write_text(row)
    import hashlib

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "config": {"include_melody": True, "stress_source": "ipa", "max_syllables": 8},
                "songs": {"song-a": {"split": "train", "group": "song-a"}},
                "sha256": {"train.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()},
            }
        )
    )
    size = len(encode_melody_source(melody_song, tokenizer)["input_ids"])
    ds = MelodySongDataset(
        tmp_path, "train", tokenizer, max_source_length=size, max_target_length=64
    )
    assert len(ds) == 1 and ds.skipped == 0
    with pytest.raises(ValueError, match="No usable"):
        MelodySongDataset(
            tmp_path, "train", tokenizer, max_source_length=size - 1, max_target_length=64
        )
    ds = MelodySongDataset(
        tmp_path,
        "train",
        tokenizer,
        unit="paragraph",
        max_source_length=size - 1,
        max_target_length=64,
    )
    assert len(ds) == 2


def test_split_reuse_including_duplicate_identity(tmp_path):
    path = tmp_path / "pretraining.csv"
    with path.open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "dali_id", "split", "artist", "title"]
        )
        writer.writeheader()
        writer.writerow(
            dict(sample_id="x", dali_id="up", split="test", artist="Artist", title="Song")
        )
    songs = {"down": {"artist": "artist", "title": "Song!"}}
    assert preserve_pretraining_splits(songs, {"down": "train"}, path) == {"down": "test"}


@pytest.mark.parametrize("warmstart", [False, True])
def test_prepare_train_reload_and_midi(
    tmp_path,
    annotation,
    tokenizer,
    tiny_model,
    monkeypatch,
    warmstart,
):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.infer import infer
    from prosodia_lyricist.train import train

    midi = pytest.importorskip("miditoolkit")
    monkeypatch.setattr(ipa, "backend", lambda: None)
    monkeypatch.setattr(
        ipa,
        "parse_words",
        lambda text: [
            {"text": w, "syllables": [{"stress": "strong", "length": "short"}]}
            for w in text.split()
        ],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(20):
        row = copy.deepcopy(annotation)
        row["info"].update(id=f"song-{i}", title=f"song {i}")
        for note in row["annotations"]["annot"]["notes"]:
            note["freq"] = [261.6256]
        (raw / f"{i}.json").write_text(json.dumps(row))
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    config = {
        "data": {
            "dali_dir": str(raw),
            "prepared_dir": str(tmp_path / "data"),
            "language": "english",
            "min_ncc": 0,
            "max_syllables": 8,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 1234,
            "stress_source": "ipa",
            "include_melody": True,
            "unit": "song",
        },
        "model": {
            "conditioning": "melody",
            "pretrained": str(tmp_path / "tokenizer"),
            "dropout": 0,
            "max_source_length": 64,
            "max_target_length": 64,
        },
        "training": {
            "output_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "seed": 1234,
            "epochs": 1,
            "batch_size": 2,
            "patience": 2,
            "learning_rate": 0.001,
            "adapter_learning_rate": 0.001,
            "melody_learning_rate": 0.0001,
            "warmup_steps": 0,
            "schedule": "constant_after_warmup",
            "freeze_decoder_epochs": 0,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 2,
        },
    }
    manifest = prepare(config)
    if warmstart:
        template_path = tmp_path / "template"
        ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0).save(template_path, tokenizer)
        pretraining = tmp_path / "pretraining.csv"
        with pretraining.open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "dali_id",
                    "split",
                    "artist",
                    "title",
                    "line_count",
                ],
            )
            writer.writeheader()
            for i in range(20):
                writer.writerow(
                    {
                        "sample_id": str(i),
                        "dali_id": f"song-{i}",
                        "split": manifest["songs"][f"song-{i}"]["split"],
                        "artist": "Artist",
                        "title": f"song {i}",
                        "line_count": 1,
                    }
                )
        tower = MelodyTransformerEncoder(
            d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, projection_dim=4, dropout=0
        )
        checkpoint_path = tmp_path / "melody.pt"
        torch.save(
            {
                "melody_representation": MELODY_REPRESENTATION,
                "args": {
                    "audio_pooling": "note",
                    "melody_d_model": 8,
                    "melody_num_layers": 1,
                    "melody_num_heads": 2,
                    "melody_dim_feedforward": 16,
                    "projection_dim": 4,
                    "dropout": 0,
                    "manifest": pretraining,
                },
                "model_state_dict": {
                    f"melody_encoder.{k}": v for k, v in tower.state_dict().items()
                },
            },
            checkpoint_path,
        )
        config["data"]["pretraining_manifest"] = str(pretraining)
        config["model"].update(
            template_checkpoint=str(template_path), melody_checkpoint=str(checkpoint_path)
        )
        config["training"].update(
            freeze_decoder_epochs=1, loss_weights={"lyrics": 1, "syllables": 0.5}
        )
        manifest = prepare(config)
    split = manifest["songs"]["song-0"]["split"]
    record = json.loads((tmp_path / "data" / f"{split}.jsonl").read_text().splitlines()[0])
    assert len(record["lines"][0]["melody"]["midi_pitches"]) == 4  # Keep melisma note.
    output = train(config, smoke_test=not warmstart, local_files_only=True)
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert set(metrics["train"]["components"]) == set(LOSS_NAMES)
    assert metrics["train"]["optimizer_steps"] == (metrics["train"]["batches"] + 1) // 2
    if warmstart:
        assert MelodyBart.load(output / "best").loss_weights["syllables"] == 0.5
        saved = json.loads((output / "run.json").read_text())
        assert saved["pretraining_split_audit"]["shared_songs"] == 20
    song = midi.MidiFile()
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60, 0, 240), midi.Note(80, 62, 960, 1440)]
    song.instruments.append(instrument)
    song.markers.append(midi.Marker("Phrase_0", 480))
    song.tempo_changes = [midi.TempoChange(120, 0), midi.TempoChange(60, 480)]
    song.dump(str(tmp_path / "song.mid"))
    source = midi_melody_record(tmp_path / "song.mid")
    assert len(source["lines"]) == 2  # Retain trailing unmarked phrase.
    assert source["lines"][1]["melody"]["note_duration_seconds"] == [1.0]
    result = infer(
        output / "best",
        tmp_path / "song.mid",
        max_new_tokens=3,
        device="cpu",
        top_k=1,
        return_explanations=True,
        prosody_correction=False,
    )
    assert result["conditioning"] == "melody"
    assert result["input_phrases"] == result["encoder_windows"] == 2
    assert len(result["streams"]) == 5
