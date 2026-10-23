"""Musical reconstruction, leakage guards, and encoder transfer to IPA training."""

import copy
import json
from pathlib import Path

import pytest
import torch

from prosodia_lyricist import bridge_pretrain, bridge_train
from prosodia_lyricist.bridge_alignment import ALIGNMENT_SCHEME
from prosodia_lyricist.bridge_data import bridge_collate
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_pretrain import (
    ATTRIBUTES,
    MaskedMelodyModel,
    MelodyPretrainDataset,
    corrupt_notes,
    eligible_notes,
    encoder_configuration,
    initialize_from_masked,
    read_pretraining_metadata,
    run_pretrain_epoch,
    sample_note_mask,
    train_pretrain,
)
from prosodia_lyricist.ipa import REPO_RULES, REPO_STRESSES
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.melody_encoder.encoding import ONSET_SHIFT_SLICE
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture
def config(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    data = dict(prepared_dir=str(directory), include_melody=True, lines_per_window=2,
                max_syllables=8, stress_source="ipa", prosody_mode="actual")
    songs, hashes = {}, {}
    for split, count in (("train", 3), ("valid", 2)):
        rows = []
        for index in range(count):
            name = f"{split}-{index}"
            songs[name] = {"group": name, "split": split}
            lines = []
            for phrase in range(3 - index % 2):
                n = 4 + index % 2
                lines.append(dict(
                    melody=dict(midi_pitches=[60, 62, 58, 65, 61][:n],
                                onset_seconds=[phrase * 10 + t for t in (0, 1, 2.5, 4, 5.5)][:n],
                                note_duration_seconds=[0.25, 0.5, 0.75, 1, 0.25][:n]),
                    syllables=[dict(stress=REPO_STRESSES[i % 3],
                                    length=("long", "short")[i % 2]) for i in range(n)],
                    note_alignment=dict(scheme=ALIGNMENT_SCHEME, note_count=n,
                                        words=[dict(note_starts=list(range(n)),
                                                    syllable_indices=list(range(n)))]),
                ))
            rows.append(dict(song_id=name, title="unused", lines=lines))
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes[path.name] = file_sha256(path)
    # Deliberately no test.jsonl or raw DALI files: neither stage may open them.
    (directory / "manifest.json").write_text(json.dumps(dict(
        schema_version=SCHEMA_VERSION, config=data, songs=songs, sha256=hashes,
        pronunciation=dict(rules=REPO_RULES),
    )))
    return {
        "data": data,
        "model": dict(
            conditioning="bridge", melody_initialization="random", melody_checkpoint=None,
            random_melody_encoder=dict(d_model=8, num_heads=2, num_layers=1,
                                       dim_feedforward=16, dropout=0, max_length=32),
            bridge_scope="song", query_mode="note", encoder_lines_per_window=2,
            max_window_notes=16, max_song_notes=32, max_song_lines=4, max_target_length=32,
            song_encoder_layers=1,
            song_encoder=dict(d_model=8, num_heads=2, dim_feedforward=16, dropout=0),
        ),
        "training": dict(device="cpu", precision="fp32", seed=123, epochs=2, batch_size=2,
                         output_dir=str(tmp_path / "runs"), patience=2, learning_rate=1e-3,
                         melody_learning_rate=1e-3, melody_unfreeze_epoch=0,
                         gradient_accumulation_steps=2, mask_probability=0.3,
                         gradient_clip=1, warmup_steps=0, schedule="constant_after_warmup"),
    }


def dataset(config, split="train"):
    cfg = config["model"]
    return MelodyPretrainDataset(config["data"]["prepared_dir"], split, **{
        k: cfg[k] for k in ("encoder_lines_per_window", "max_window_notes",
                            "max_song_notes", "max_song_lines")
    })


def model_for(config):
    cfg = config["model"]
    return MaskedMelodyModel(cfg["random_melody_encoder"], {
        **cfg["song_encoder"], **{k: cfg[k] for k in (
            "encoder_lines_per_window", "max_window_notes", "max_song_notes",
            "max_song_lines", "song_encoder_layers",
        )},
    })


def test_source_only_pretraining_ignores_ipa_and_alignment(config):
    before = dataset(config)
    directory = Path(config["data"]["prepared_dir"])
    path = directory / "train.jsonl"
    records = [json.loads(row) for row in path.read_text().splitlines()]
    for song in records:
        del song["title"]
        for line in song["lines"]:
            del line["syllables"], line["note_alignment"]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["pronunciation"]
    manifest["config"]["stress_source"] = "not_ipa"
    manifest["sha256"][path.name] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    after = dataset(config)
    old, new = bridge_collate(before.examples), bridge_collate(after.examples)
    assert all(torch.equal(old[k], new[k]) for k in old)
    assert not any(k in new for k in ("labels", "target_note_indices", "syllable_counts"))


def test_masking_keeps_window_references_and_padding(config):
    batch = bridge_collate(dataset(config).examples)
    eligible = eligible_notes(batch, 2)
    assert not eligible[:, 0].any() and not eligible[0, 8]
    assert not eligible[~batch["melody_attention_mask"].bool()].any()
    full = sample_note_mask(batch, 1, 2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(full, eligible)
    small = sample_note_mask(batch, 1e-10, 2, generator=torch.Generator().manual_seed(5))
    assert small.sum(1).tolist() == [1, 1, 1]
    repeated = sample_note_mask(batch, 1e-10, 2, generator=torch.Generator().manual_seed(5))
    assert torch.equal(small, repeated)


@pytest.mark.parametrize("probability", [0, -0.1, 1.1, None, True, float("nan")])
def test_invalid_mask_probability(config, probability):
    with pytest.raises(ValueError, match="mask_probability"):
        sample_note_mask(bridge_collate(dataset(config).examples), probability, 2,
                         generator=torch.Generator())


def test_both_views_and_successor_intervals_are_masked_before_local_tower(config, monkeypatch):
    model = model_for(config)
    batch = bridge_collate(dataset(config).examples[:1])
    selected = torch.zeros_like(batch["melody_attention_mask"], dtype=torch.bool)
    selected[0, [2, 7, 10]] = True  # Include the last note before a local-window boundary.
    original = {k: v.clone() for k, v in batch.items()}
    masked = corrupt_notes(batch, selected)
    assert masked["melody_features"][selected].eq(0).all()
    assert masked["song_features"][selected].eq(0).all()
    assert masked["melody_features"][0, [3, 8, 11], ONSET_SHIFT_SLICE].eq(0).all()
    assert masked["song_features"][0, [3, 8, 11]][:, [1, 3, 5]].eq(0).all()
    assert torch.equal(masked["melody_features"][0, 1], original["melody_features"][0, 1])
    assert torch.equal(masked["song_features"][0, 3, 7:], original["song_features"][0, 3, 7:])
    assert all(torch.equal(batch[k], original[k]) for k in batch)
    captured, encode = [], model.encoder.melody_encoder.encode

    def capture(features, *args, **kwargs):
        captured.append(features.detach().clone())
        return encode(features, *args, **kwargs)

    monkeypatch.setattr(model.encoder.melody_encoder, "encode", capture)
    output = model(**batch, selected=selected)
    assert captured[0][0, 2].eq(0).all() and captured[0][1, 2].eq(0).all()
    assert output.targets["pitch"].tolist() == [2, 5, 2]
    assert output.targets["pitch_sign"].tolist() == [0, 1, 0]
    assert output.count == 3


def test_gradients_reach_both_encoders_and_fusion_without_ipa_heads(config):
    model = model_for(config)
    batch = bridge_collate(dataset(config).examples)
    selected = sample_note_mask(batch, 0.5, 2, generator=torch.Generator())
    model(**batch, selected=selected).loss.backward()
    for prefix in ("encoder.melody_encoder.transformer.", "encoder.adapter.",
                   "encoder.song_features.", "encoder.song_encoder.", "heads."):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(prefix))
    assert not any("strength_head" in n or "length_head" in n for n, _ in model.named_parameters())


def test_fixed_mask_reconstruction_can_learn(config):
    torch.manual_seed(123)
    model = model_for(config)
    batch = bridge_collate(dataset(config).examples[:1])
    selected = sample_note_mask(batch, 0.3, 2, generator=torch.Generator().manual_seed(17))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(**batch, selected=selected).loss.item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        model(**batch, selected=selected).loss.backward()
        optimizer.step()
    assert model(**batch, selected=selected).loss.item() < initial * 0.4


def test_deterministic_validation_and_masked_token_weighted_accumulation(config, monkeypatch):
    examples = dataset(config).examples
    model = model_for(config)
    batches = [bridge_collate(examples[:1]), bridge_collate(examples[1:])]
    first = run_pretrain_epoch(
        model, batches, "cpu", mask_generator=torch.Generator().manual_seed(8)
    )
    second = run_pretrain_epoch(
        model, batches, "cpu", mask_generator=torch.Generator().manual_seed(8)
    )
    assert first == second

    # Different numbers of masked tokens in each microbatch must give the same
    # update as their concatenated batch, including a partial accumulation group.
    monkeypatch.setattr(bridge_pretrain, "sample_note_mask", lambda batch, *a, **kw:
                        eligible_notes(batch, 2))
    combined, accumulated = copy.deepcopy(model), copy.deepcopy(model)
    opt_combined = torch.optim.SGD(combined.parameters(), lr=0.01)
    opt_accumulated = torch.optim.SGD(accumulated.parameters(), lr=0.01)
    run_pretrain_epoch(combined, [bridge_collate(examples)], "cpu", optimizer=opt_combined,
                       mask_generator=torch.Generator(), gradient_clip=None)
    result = run_pretrain_epoch(accumulated, batches, "cpu", optimizer=opt_accumulated,
                                accumulation_steps=3, mask_generator=torch.Generator(),
                                gradient_clip=None)
    assert result["updates"] == 1 and result["batches"] == 2
    for a, b in zip(combined.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-5)


def test_checkpoint_roundtrip_and_full_encoder_transfer_with_fresh_prosody_heads(config, tmp_path):
    model = model_for(config).eval()
    batch = bridge_collate(dataset(config).examples)
    selected = sample_note_mask(batch, 0.3, 2, generator=torch.Generator())
    directory = tmp_path / "pretrained"
    manifest_path = Path(config["data"]["prepared_dir"]) / "manifest.json"
    hashes = json.loads(manifest_path.read_text())["sha256"]
    model.save(directory, data_sha256=hashes, masking={"probability": 0.3})
    restored = MaskedMelodyModel.load(directory).eval()
    for name, logits in model(**batch, selected=selected).logits.items():
        torch.testing.assert_close(logits, restored(**batch, selected=selected).logits[name])
    spec = encoder_configuration(model.encoder)
    bridge = ProsodyBridge(spec["melody_config"], **spec["encoder_config"],
                            bridge_scope="song", query_mode="note",
                            strength_labels=REPO_STRESSES).eval()
    heads_before = bridge.strength_head.weight.detach().clone()
    meta = read_pretraining_metadata(directory)
    initialize_from_masked(bridge, directory, meta)
    assert torch.equal(heads_before, bridge.strength_head.weight)
    torch.testing.assert_close(model.encoder.encode(**batch), bridge.encode(**batch))
    # Inference retains one prediction per note, without reconstruction heads.
    assert bridge.predict_notes(**batch).length_logits.shape[:2] == batch["note_line_ids"].shape
    with pytest.raises(ValueError, match="architecture differs"):
        initialize_from_masked(bridge, directory, {**meta, "encoder_config": {}})
    with pytest.raises(ValueError, match="splits differ"):
        read_pretraining_metadata(directory, manifest={"sha256": {}})
    meta["song_feature_scheme"] = "wrong"
    (directory / "pretrain.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature representation"):
        MaskedMelodyModel.load(directory)


@pytest.mark.parametrize("smoke_test", [False, True])
def test_end_to_end_self_supervision_then_ipa_finetune(config, tmp_path, monkeypatch, smoke_test):
    torch.manual_seed(config["training"]["seed"])
    initial = model_for(config).encoder.melody_encoder.input_projection[0].weight.detach().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Random pretraining and masked fine-tuning must not import contrastive weights")

    monkeypatch.setattr(bridge_pretrain, "load_contrastive_melody", forbidden)
    monkeypatch.setattr(bridge_train, "load_contrastive_melody", forbidden)
    output = train_pretrain(config, smoke_test=smoke_test)
    pretrained = MaskedMelodyModel.load(output / "best")
    assert not torch.equal(initial, pretrained.encoder.melody_encoder.input_projection[0].weight)
    metrics = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == (1 if smoke_test else 2)
    assert metrics[0]["train"]["updates"] == 1
    assert all(0 <= v <= 1 for v in metrics[0]["valid"]["accuracy"].values())
    assert set(metrics[0]["train"]["components"]) == set(ATTRIBUTES)
    for name in ATTRIBUTES:
        scores = metrics[0]["valid"]
        assert sum(scores["target_counts"][name]) == scores["masked_notes"]
        assert sum(scores["prediction_counts"][name]) == scores["masked_notes"]
    config["model"].update(melody_initialization="masked", melody_checkpoint=str(output / "best"))
    # Scratch masked pretraining has no dependency on an old contrastive CSV,
    # even if the prepared manifest retains its historical path.
    config["data"]["pretraining_manifest"] = str(tmp_path / "missing-contrastive.csv")
    manifest_path = Path(config["data"]["prepared_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = config["data"]
    manifest_path.write_text(json.dumps(manifest))
    captured, original = [], bridge_train.run_bridge_epoch

    def capture(model, *args, **kwargs):
        if kwargs.get("optimizer") is not None:
            captured.append(copy.deepcopy(model.state_dict()))
        return original(model, *args, **kwargs)

    monkeypatch.setattr(bridge_train, "run_bridge_epoch", capture)
    config["training"]["epochs"] = 1
    fine_output = bridge_train.train_bridge(config, output_dir=tmp_path / "finetune",
                                            smoke_test=smoke_test)
    for name, value in pretrained.encoder.state_dict().items():
        assert torch.equal(value, captured[0][name])
    bridge = ProsodyBridge.load(fine_output / "best")
    run = json.loads((fine_output / "run.json").read_text())
    assert run["pretraining_split_check"]["basis"] == "masked_pretraining_train_valid_checksums"
    assert bridge.provenance["initialization"] == "masked"
    assert bridge.provenance["loaded_scope"] == "full_encoder"
    assert bridge.strength_labels == REPO_STRESSES
    assert bridge.predict_notes(**bridge_collate(dataset(config).examples)).length_logits.shape == (
        3, 12, 2,
    )


def test_source_checksums_and_melody_limits(config):
    config["model"]["max_song_notes"] = 11
    ds = dataset(config)
    assert len(ds) == 1 and ds.skipped["song_notes"] == 2
    path = Path(config["data"]["prepared_dir"]) / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        dataset(config)
    with pytest.raises(ValueError, match="only train and valid"):
        dataset(config, "test")
