"""Scratch bridge towers train without checkpoint I/O and keep initialization provenance."""

import json
from pathlib import Path

import pytest
import torch

from prosodia_lyricist import bridge_train
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture
def config(tmp_path):
    data = dict(prepared_dir=str(tmp_path / "data"), include_melody=True,
                stress_source="ipa", max_syllables=8, lines_per_window=2,
                pretraining_manifest=None)
    directory = Path(data["prepared_dir"])
    directory.mkdir()
    songs, hashes = {}, {}
    for split, count in (("train", 10), ("valid", 2)):
        records = []
        for index in range(count):
            name = f"{split}-{index}"
            songs[name] = {"split": split, "group": name}
            records.append({"song_id": name, "lines": [{
                "syllables": [{"stress": "strong", "length": "long"},
                              {"stress": "weak", "length": "short"}],
                "melody": {"midi_pitches": [60, 62, 64], "onset_seconds": [0, 1, 2],
                           "note_duration_seconds": [0.5, 0.25, 0.5]},
            }]})
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        hashes[path.name] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "config": data, "songs": songs,
        "sha256": hashes, "pronunciation": {"rules": "test"},
    }))
    return {
        "data": data,
        "model": dict(
            conditioning="bridge", melody_initialization="random", melody_checkpoint=None,
            random_melody_encoder=dict(d_model=8, num_layers=2, num_heads=2,
                                       dim_feedforward=16, dropout=0, pooling="cls", max_length=32),
            bridge_scope="song", encoder_lines_per_window=2, max_window_notes=16,
            max_song_notes=32, max_song_lines=4, max_target_length=32, song_encoder_layers=1,
            bridge_decoder=dict(d_model=8, num_layers=1, num_heads=2,
                                dim_feedforward=16, dropout=0),
        ),
        "training": dict(
            output_dir=str(tmp_path / "runs"), device="cpu", seed=1234, epochs=1,
            batch_size=2, patience=2, learning_rate=1e-3, melody_learning_rate=1e-3,
            melody_unfreeze_epoch=0, warmup_steps=0, schedule="constant_after_warmup",
            history_mask_probability=0.3,
        ),
    }


@pytest.mark.parametrize("smoke_test", [False, True])
@pytest.mark.parametrize("checkpoint_present", [False, True])
def test_random_tower_never_loads_weights_and_trains_full_architecture(
    config, tmp_path, monkeypatch, smoke_test, checkpoint_present
):
    ignored = tmp_path / "ignored.pt"
    if checkpoint_present:
        ignored.write_text("This is deliberately not a PyTorch checkpoint.")
        config["model"]["melody_checkpoint"] = str(ignored)
    initial = []
    original_init = ProsodyBridge.__init__

    def capture_initial(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        initial.append(self.melody_encoder.input_projection[0].weight.detach().clone())

    def forbidden(*args, **kwargs):
        pytest.fail("Random initialization must not load pretrained weights")

    def checked_hash(path):
        assert Path(path) != ignored
        return file_sha256(path)

    with monkeypatch.context() as context:
        context.setattr(ProsodyBridge, "__init__", capture_initial)
        context.setattr(torch, "load", forbidden)
        context.setattr(MelodyTransformerEncoder, "load_state_dict", forbidden)
        context.setattr(bridge_train, "load_contrastive_melody", forbidden)
        context.setattr(bridge_train, "file_sha256", checked_hash)
        output = bridge_train.train_bridge(config, smoke_test=smoke_test)

    run = json.loads((output / "run.json").read_text())
    metrics = json.loads((output / "metrics.jsonl").read_text())
    restored = ProsodyBridge.load(output / "best")
    assert restored.melody_config == {
        **config["model"]["random_melody_encoder"], "projection_dim": None,
    }
    assert restored.melody_encoder.projection is None
    assert run["smoke_test"] == smoke_test
    assert run["examples"]["train"] == (8 if smoke_test else 10)
    assert run["melody_initialization"] == "random"
    assert run["melody_provenance"] == restored.provenance == {
        "initialization": "random", "weights_loaded": False,
        "checkpoint": None, "sha256": None,
        "architecture_source": "model.random_melody_encoder",
    }
    assert not metrics["melody_frozen"]
    assert metrics["train"]["loss"] > 0
    assert not torch.equal(initial[0], restored.melody_encoder.input_projection[0].weight)


def test_random_initialization_is_reproducible_and_seeded(config, tmp_path, monkeypatch):
    original, initial = bridge_train.run_bridge_epoch, []

    def capture_epoch(model, *args, **kwargs):
        if kwargs.get("optimizer") is not None:
            initial.append(model.melody_encoder.input_projection[0].weight.detach().clone())
        return original(model, *args, **kwargs)

    monkeypatch.setattr(bridge_train, "run_bridge_epoch", capture_epoch)
    trained = []
    for index, seed in enumerate((17, 17, 18)):
        config["training"]["seed"] = seed
        output = bridge_train.train_bridge(config, output_dir=tmp_path / f"seed-{index}")
        trained.append(ProsodyBridge.load(output / "best").state_dict())
    assert torch.equal(initial[0], initial[1])
    assert not torch.equal(initial[0], initial[2])
    assert all(torch.equal(value, trained[1][key]) for key, value in trained[0].items())


@pytest.mark.parametrize("unfreeze,expected", [(None, [True, True]), (1, [True, False])])
def test_random_initialization_respects_freeze_schedule(config, unfreeze, expected):
    config["training"].update(epochs=2, melody_unfreeze_epoch=unfreeze)
    output = bridge_train.train_bridge(config)
    metrics = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [epoch["melody_frozen"] for epoch in metrics] == expected


def test_random_initialization_preserves_preparation_manifest_audit(config, tmp_path):
    upstream = tmp_path / "upstream.csv"
    upstream.write_text("sample_id,dali_id,split,line_count\nup-1,up-song,train,2\n")
    config["data"]["pretraining_manifest"] = str(upstream)
    path = Path(config["data"]["prepared_dir"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"] = config["data"]
    manifest["pretraining_split_audit"] = {"sha256": file_sha256(upstream)}
    path.write_text(json.dumps(manifest))
    output = bridge_train.train_bridge(config)
    run = json.loads((output / "run.json").read_text())
    assert run["pretraining_split_audit"] == manifest["pretraining_split_audit"]
    upstream.write_text(upstream.read_text() + "\n")
    with pytest.raises(ValueError, match="unchanged melody pretraining manifest"):
        bridge_train.train_bridge(config)


@pytest.mark.parametrize("initialization", [None, "scratch", 1])
def test_invalid_initialization_fails_before_loading_data(initialization):
    with pytest.raises(ValueError, match="melody_initialization"):
        bridge_train.train_bridge({
            "data": {}, "model": {"melody_initialization": initialization}, "training": {},
        })


@pytest.mark.parametrize("override", [
    None, {}, {"d_model": 0}, {"num_heads": 3}, {"max_length": 0},
    {"dropout": float("nan")}, {"pooling": "unsupported"}, {"projection_dim": 4},
])
def test_invalid_random_architecture_fails_before_loading_data(override):
    architecture = override
    if override:
        architecture = dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16)
        architecture.update(override)
    with pytest.raises(ValueError, match="random_melody_encoder"):
        bridge_train.train_bridge({
            "data": {}, "training": {},
            "model": {"melody_initialization": "random", "random_melody_encoder": architecture},
        })
