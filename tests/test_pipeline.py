import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from prosodia_lyricist.data import LyricDataset, ProsodyCollator, read_manifest
from prosodia_lyricist.legacy_features import encode_example, encode_source
from prosodia_lyricist.legacy_model import ProsodyBart
from prosodia_lyricist.prepare import prepare, song_groups, split_for
from prosodia_lyricist.train import run_epoch


def test_padding_and_remainder_zero(tokenizer, record):
    a = encode_example(record, tokenizer, 8)
    b = encode_example(
        {**record, "text": "light", "syllables": record["syllables"][:1]}, tokenizer, 8
    )
    batch = ProsodyCollator(tokenizer.pad_token_id)([a, b])
    assert batch["input_ids"][1, -1] == tokenizer.pad_token_id
    assert batch["attention_mask"][1, -1] == 0
    assert batch["labels"][1, -1] == -100
    assert batch["length_ids"][1, -1] == batch["remainder_ids"][1, -1] == 0
    assert [n for n in a["remainder_ids"] if n] == [3, 2, 1]


def test_source_does_not_contain_target_text(tokenizer, record):
    a = encode_source(record, tokenizer, 8)
    assert a == encode_source({**record, "text": "different secret lyrics"}, tokenizer, 8)
    with pytest.raises(ValueError, match="syllables"):
        encode_source({**record, "syllables": []}, tokenizer, 8)


def test_forward_shifts_labels_and_ignores_padding(tokenizer, tiny_model, record):
    example = encode_example(record, tokenizer, 8)
    batch = ProsodyCollator(tokenizer.pad_token_id)([example])
    tiny_model.eval()
    captured = {}

    def capture(module, args, kwargs):
        captured["decoder_input_ids"] = kwargs["decoder_input_ids"].clone()

    hook = tiny_model.bart.model.register_forward_pre_hook(capture, with_kwargs=True)
    output = tiny_model(**batch)
    hook.remove()
    expected = tiny_model.bart.prepare_decoder_input_ids_from_labels(batch["labels"])
    assert torch.equal(captured["decoder_input_ids"], expected)
    assert captured["decoder_input_ids"][0, 0] == 2
    assert captured["decoder_input_ids"][0, 1] == 0  # shift BOS, not current target
    padded = {
        key: torch.nn.functional.pad(
            value, (0, 2), value=(-100 if key == "labels" else 1 if key == "input_ids" else 0)
        )
        for key, value in batch.items()
    }
    assert torch.allclose(output.loss, tiny_model(**padded).loss, atol=1e-6)
    output.loss.backward()
    assert tiny_model.projection.weight.grad.abs().sum() > 0


def test_optimizer_and_checkpoint_round_trip(tmp_path, tokenizer, tiny_model, record):
    batch = ProsodyCollator(tokenizer.pad_token_id)([encode_example(record, tokenizer, 8)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = tiny_model.projection.weight.detach().clone()
    metrics = run_epoch(tiny_model, [batch], "cpu", optimizer=optimizer, scheduler=scheduler)
    assert metrics["batches"] == 1 and metrics["tokens"] == len(batch["labels"][0])
    assert not torch.equal(before, tiny_model.projection.weight)
    tiny_model.bart.generation_config.num_beams = 2
    tiny_model.eval().save(tmp_path, tokenizer)
    restored = ProsodyBart.load(tmp_path).eval()
    assert restored.bart.generation_config.num_beams == 2
    restored_tokenizer = AutoTokenizer.from_pretrained(
        tmp_path / "tokenizer", local_files_only=True
    )
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert torch.equal(tiny_model(**batch).logits, restored(**batch).logits)
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    result = restored.generate(**inputs, max_new_tokens=3, do_sample=False)
    assert result.ndim == 2 and result.shape[1] <= 4


def test_validation_loss_weighted_by_tokens():
    class Model:
        def train(self, mode):
            assert mode is False

        def __call__(self, labels):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(loss=torch.tensor(float(labels[0, 0])))

    metrics = run_epoch(
        Model(),
        [{"labels": torch.tensor([[2, -100]])}, {"labels": torch.tensor([[4, 4, 4]])}],
        "cpu",
    )
    assert metrics["loss"] == 3.5


def test_duplicate_song_grouping_is_transitive():
    songs = {
        "a": {"artist": "Artist", "title": "Song!", "audio": {"url": "1"}},
        "b": {"artist": "artist", "title": "Song", "audio": {"url": "2"}},
        "c": {"artist": "Other", "title": "Title", "audio": {"url": "2"}},
        "d": {"artist": "Different", "title": "Title", "audio": {"url": "3"}},
    }
    groups = song_groups(songs)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
    assert groups == song_groups(dict(reversed(list(songs.items()))))


def test_prepare_reproducible_splits_and_loader(tmp_path, annotation, tokenizer):
    source = tmp_path / "raw"
    source.mkdir()
    for i in range(30):
        row = copy.deepcopy(annotation)
        row["info"].update(id=f"song-{i}", title=f"Song {i}")
        (source / f"{i:02}.json").write_text(json.dumps(row))
    config = {
        "data": {
            "dali_dir": str(source),
            "prepared_dir": str(tmp_path / "prepared"),
            "max_syllables": 8,
            "language": "english",
            "min_ncc": 0,
            "seed": 1234,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
            "stress_source": "lexical",
        }
    }
    first = prepare(config)
    second = prepare(config)
    assert first == second
    assert first["counts"]["train_lines"] > 0
    assert first["counts"]["valid_lines"] > 0
    assert first["counts"]["test_lines"] > 0
    directory = config["data"]["prepared_dir"]
    dataset = LyricDataset(
        directory,
        "train",
        tokenizer,
        decoder_mode="lyrics",
        max_source_length=64,
        max_target_length=64,
    )
    assert len(dataset) == first["counts"]["train_lines"]
    assert read_manifest(directory) == first
    from pathlib import Path

    with (Path(directory) / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="differs"):
        LyricDataset(
            directory,
            "train",
            tokenizer,
            decoder_mode="lyrics",
            max_source_length=64,
            max_target_length=64,
        )


def test_split_is_stable():
    assert split_for("song", 1234, 0.1, 0.1) == split_for("song", 1234, 0.1, 0.1)
    assert {split_for(str(i), 1234, 0.1, 0.1) for i in range(100)} == {"train", "valid", "test"}


@pytest.mark.parametrize("auxiliary", [False, True])
def test_offline_training_and_midi_inference(tmp_path, annotation, tokenizer, auxiliary):
    from prosodia_lyricist.infer import infer
    from prosodia_lyricist.train import train

    midi = pytest.importorskip("miditoolkit")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(30):
        row = copy.deepcopy(annotation)
        row["info"].update(id=f"song-{i}", title=f"Song {i}")
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
            "stress_source": "unknown",
        },
        "model": {
            "decoder_mode": "lyrics",
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
            "num_workers": 0,
            "patience": 2,
            "warmup_ratio": 0,
            "gradient_clip": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "fixed_batch_order": True,
            "loss_weights": {"syllable": 0.5, "remainder": 0.5} if auxiliary else {},
        },
    }
    prepare(config)
    output = train(config, smoke_test=True, local_files_only=True)
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert metrics["train"]["batches"] == metrics["valid"]["batches"] == 2
    assert metrics["learning_rate"] == 0
    assert ("syllable" in metrics["train"]["components"]) == auxiliary
    melody = midi.MidiFile()
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60, 0, 480)]
    melody.instruments.append(instrument)
    melody.markers.append(midi.Marker("Phrase_0", 480))
    melody.dump(str(tmp_path / "test.mid"))
    lyrics = infer(
        output / "best",
        tmp_path / "test.mid",
        title="a song",
        device="cpu",
        top_k=1,
        max_new_tokens=3,
    )
    assert len(lyrics) == 1
