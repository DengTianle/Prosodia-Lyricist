"""Gradient flow, source/target separation, and pretrained decoder compatibility."""

import copy
import json

import pytest
import torch

from prosodia_lyricist.bridge_data import decode_bridge_tokens
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.end_to_end_data import EndToEndCollator, EndToEndDataset, encode_song
from prosodia_lyricist.end_to_end_model import EndToEndProsody, condition_inputs
from prosodia_lyricist.end_to_end_train import build_model, run_epoch, train_end_to_end
from prosodia_lyricist.features import TARGET_KEYS, encode_source, token_id
from prosodia_lyricist.ipa import HISTORICAL_POLICY, REPO_RULES, REPO_STRESSES
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.model import ProsodyBart
from prosodia_lyricist.prepare import SCHEMA_VERSION
from prosodia_lyricist.xai_model import XAILyricist


@pytest.fixture
def joint_song():
    return {
        "song_id": "a",
        "title": "a song",
        "lines": [
            {
                "text": "hello world",
                "keyword": "light",
                "words": [
                    {"text": "hello", "syllable_count": 2},
                    {"text": "world", "syllable_count": 1},
                ],
                "syllables": [
                    {"stress": "weak", "length": "short"},
                    {"stress": "strong", "length": "long"},
                    {"stress": "strong", "length": "short"},
                ],
                "melody": {
                    "midi_pitches": [60, 62, 65, 63],
                    "onset_seconds": [0, 0.5, 1, 1.5],
                    "note_duration_seconds": [0.25, 0.4, 0.25, 0.5],
                },
            },
            {
                "text": "light",
                "words": [{"text": "light", "syllable_count": 1}],
                "syllables": [{"stress": "strong", "length": "long"}],
                "melody": {
                    "midi_pitches": [67],
                    "onset_seconds": [3],
                    "note_duration_seconds": [1],
                },
            },
        ],
    }


@pytest.fixture
def joint_model(tiny_model, tokenizer):
    bridge = ProsodyBridge(
        dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0, max_length=32),
        query_mode="parallel",
        bridge_scope="song",
        max_syllables=8,
        encoder_lines_per_window=2,
        max_window_notes=16,
        max_song_notes=32,
        max_song_lines=8,
        max_target_length=64,
        song_encoder_layers=1,
        d_model=16,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
        dropout=0,
    )
    template = ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0)
    return EndToEndProsody(
        bridge,
        template,
        stress_token_ids=[token_id(tokenizer, f"<{s}>") for s in ("strong", "weak")],
        loss_weights=dict(lyrics=1, syllables=0, stresses=0, lengths=0),
    )


def encoded(song, tokenizer, **kwargs):
    return encode_song(song, tokenizer, max_syllables=8, encoder_lines_per_window=2, **kwargs)


def batch(song, tokenizer):
    return EndToEndCollator(tokenizer.pad_token_id)([encoded(song, tokenizer)])


def as_xai(model, tokenizer):
    return EndToEndProsody(
        model.bridge,
        XAILyricist(model.template_model.bart, max_syllables=8),
        stress_token_ids=[token_id(tokenizer, f"<{s}>") for s in ("strong", "weak")],
    )


def as_repo_xai(model, tokenizer):
    bridge = ProsodyBridge(
        model.bridge.melody_config,
        **model.bridge.decoder_config,
        query_mode="parallel",
        bridge_scope="song",
        strength_labels=REPO_STRESSES,
        max_syllables=8,
        encoder_lines_per_window=2,
        max_window_notes=16,
        max_song_notes=32,
        max_song_lines=8,
        max_target_length=64,
        song_encoder_layers=1,
    )
    return EndToEndProsody(
        bridge,
        XAILyricist(model.template_model.bart, max_syllables=8),
        stress_token_ids=[token_id(tokenizer, f"<{s}>") for s in REPO_STRESSES],
        anchor_weight=0.2,
    )


def test_repo_three_class_bridge_retains_secondary_stress_and_exact_hard_source(
    joint_model, joint_song, tokenizer, tmp_path, monkeypatch
):
    model = as_repo_xai(joint_model, tokenizer).eval()
    song = copy.deepcopy(joint_song)
    song["lines"][0]["syllables"][1]["stress"] = "substrong"
    example = encoded(song, tokenizer, decoder_mode="xai", strength_labels=REPO_STRESSES)
    inputs = EndToEndCollator(tokenizer.pad_token_id)([example])
    source = model.condition_source(**condition_inputs(inputs), template_mode="hard")
    assert source.stress_probabilities.shape[-1] == 3
    assert model.bridge.strength_head.out_features == 3
    assert inputs["bridge_labels"][0, 2] == 2  # substrong/long, with structural prefix first
    preview = decode_bridge_tokens(source.sequences[0].tolist(), pairs=model.bridge.pairs)
    from prosodia_lyricist.xai_features import encode_source as xai_source

    discrete = xai_source({**song, "lines": preview}, tokenizer, 8)
    tensors = {k: torch.tensor([v]) for k, v in discrete.items()}
    torch.testing.assert_close(
        source.inputs_embeds,
        model.template_model.embed_source(
            tensors["input_ids"], tensors["length_ids"], tensors["remainder_ids"]
        ),
    )
    output = model(**inputs)
    active = inputs["bridge_labels"].ne(-100)
    truth = inputs["bridge_labels"][active]
    expected = torch.nn.functional.cross_entropy(
        output.bridge_output.strength_logits[active], truth // 2
    ) + torch.nn.functional.cross_entropy(output.bridge_output.length_logits[active], truth % 2)
    torch.testing.assert_close(output.loss_components["bridge_anchor"], expected)
    output.loss.backward()
    assert model.bridge.strength_head.weight.grad[1].abs().sum() > 0
    assert model.bridge.length_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.template_model.parameters())
    model.save(tmp_path, tokenizer)
    assert json.loads((tmp_path / "bridge/bridge.json").read_text())["format_version"] == 7
    restored = EndToEndProsody.load(tmp_path).eval()
    assert restored.prosody_mode == "actual"
    assert restored.bridge.strength_labels == REPO_STRESSES
    assert torch.equal(restored(**inputs).logits, output.logits)
    generated = restored.generate(
        **condition_inputs(inputs), tokenizer=tokenizer, max_new_tokens=6, do_sample=False
    )
    assert generated.explanations[0]["melody_conditioning"]["stress_labels"] == list(REPO_STRESSES)
    from prosodia_lyricist.bridge_infer import predict_templates

    with torch.no_grad():
        restored.bridge.strength_head.bias.copy_(torch.tensor([-1e6, 1e6, -1e6]))
    restored.bridge.save(tmp_path / "bridge")
    monkeypatch.setattr(
        "prosodia_lyricist.bridge_infer.midi_melody_record",
        lambda *args, **kwargs: {**song, "unmarked_notes_omitted": 1},
    )
    predicted, metadata = predict_templates(tmp_path / "bridge", "unused.mid")
    assert {s["stress"] for r in predicted for s in r["syllables"]} == {"substrong"}
    assert metadata["prosody_mode"] == "actual"
    assert predicted[-1]["unmarked_notes_omitted"] == 1


def test_repo_training_requires_reprepared_data_and_evaluates_all_three_classes(
    joint_config, joint_model, tokenizer, tmp_path, monkeypatch
):
    model = as_repo_xai(joint_model, tokenizer)
    with pytest.raises(ValueError, match="Prepared IPA rules"):
        EndToEndDataset(
            joint_config["data"]["prepared_dir"],
            "train",
            tokenizer,
            **model.dataset_options,
            max_source_length=64,
            max_target_length=64,
        )
    template = tmp_path / "repo-template"
    model.template_model.save(template, tokenizer)
    config = copy.deepcopy(joint_config)
    config["data"]["prosody_mode"] = "actual"
    config["model"].update(template_checkpoint=str(template), decoder_mode="xai")
    config["training"].update(loss_weights={"lyrics": 1}, bridge_anchor_weight=0.2)
    directory = tmp_path / "data"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["config"] = config["data"]
    manifest["pronunciation"] = dict(rules=REPO_RULES, pronunciation_policy=HISTORICAL_POLICY)
    for split in ("train", "valid", "test"):
        path = directory / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["lines"][0]["syllables"][1]["stress"] = "substrong"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["sha256"][path.name] = file_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    output = train_end_to_end(config, smoke_test=True)
    trained = EndToEndProsody.load(output / "best").eval()
    assert trained.bridge.strength_head.out_features == 3
    from prosodia_lyricist import ipa
    from prosodia_lyricist.end_to_end_eval import evaluate_end_to_end

    def parse(text, **kwargs):
        assert kwargs["prosody_mode"] == "actual"
        return [{"text": w, "syllables": [{"ipa": "`ə"}]} for w in text.split()]

    monkeypatch.setattr(ipa, "parse_words", parse)
    report = evaluate_end_to_end(
        output / "best",
        directory,
        device="cpu",
        generation_limit=1,
        max_new_tokens=5,
        compare_hard=True,
    )
    assert report["prosody_mode"] == "actual"
    assert report["teacher_forced"]["bridge_compared_slots"] > 0
    assert report["hard_template_ablation"]["generation"]["songs"] == 1
    assert report["duration_baselines"]["all_ipa_slots"]["compared_slots"] > 0
    json.dumps(report, allow_nan=False)


def test_xai_lyric_loss_trains_bridge_and_hard_projection_matches_reference(
    joint_model, joint_song, tokenizer
):
    model = as_xai(joint_model, tokenizer).eval()
    example = encoded(joint_song, tokenizer, decoder_mode="xai")
    assert set(example["targets"]) == {"labels"}
    assert [v for v in example["source"]["remainder_ids"] if v] == [3, 2, 1, 1]
    inputs = EndToEndCollator(tokenizer.pad_token_id)([example])
    hard = model.condition_source(**condition_inputs(inputs), template_mode="hard")
    from prosodia_lyricist.xai_features import encode_source as xai_source

    preview = decode_bridge_tokens(hard.sequences[0].tolist())
    record = {
        **joint_song,
        "lines": [
            {**line, **predicted}
            for line, predicted in zip(joint_song["lines"], preview, strict=True)
        ],
    }
    discrete = {k: torch.tensor([v]) for k, v in xai_source(record, tokenizer, 8).items()}
    torch.testing.assert_close(
        hard.inputs_embeds,
        model.template_model.embed_source(
            discrete["input_ids"], discrete["length_ids"], discrete["remainder_ids"]
        ),
    )
    changed = copy.deepcopy(joint_song)
    for line in changed["lines"]:
        line["text"], line["words"] = "secret", []
        for syllable in line["syllables"]:
            syllable.update(stress="weak", length="short")
    assert (
        example["source"]
        == encoded(changed, tokenizer, decoder_mode="xai", with_targets=False)["source"]
    )
    output = model(**inputs)
    assert set(output.loss_components) == {"lyrics", "bridge_anchor"}
    assert "prosody_logits" not in output
    output.loss.backward()
    for module in (
        model.bridge.strength_head,
        model.bridge.length_head,
        model.bridge.song_features[0],
    ):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.template_model.parameters())
    model.set_template_trainable("encoder")
    assert model.template_model.remainder_embedding.weight.requires_grad
    assert model.template_model.length_embedding.weight.requires_grad
    assert not model.template_model.bart.get_decoder().embed_tokens.weight.requires_grad


def test_xai_joint_training_save_reload_and_generation(
    joint_config, joint_model, tokenizer, tmp_path
):
    model = as_xai(joint_model, tokenizer)
    template = tmp_path / "xai-template"
    model.template_model.save(template, tokenizer)
    original = model.template_model.bart.get_decoder().layers[0].fc1.weight.detach().clone()
    config = copy.deepcopy(joint_config)
    config["model"].update(template_checkpoint=str(template), decoder_mode="xai")
    config["training"].update(
        template_unfreeze_epoch=0, template_finetune="all", loss_weights={"lyrics": 1}
    )
    output = train_end_to_end(config, smoke_test=True)
    restored = EndToEndProsody.load(output / "best").eval()
    assert isinstance(restored.template_model, XAILyricist)
    assert not torch.equal(
        original, restored.template_model.bart.get_decoder().layers[0].fc1.weight
    )
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert set(metrics["train"]["components"]) == {"lyrics", "bridge_anchor"}
    assert "teacher_forced_word_event_accuracy" not in metrics["valid"]
    ds = EndToEndDataset(
        config["data"]["prepared_dir"],
        "valid",
        tokenizer,
        **restored.dataset_options,
        max_source_length=64,
        max_target_length=64,
    )
    inputs = EndToEndCollator(tokenizer.pad_token_id)([ds[0]])
    generated = restored.generate(
        **condition_inputs(inputs), tokenizer=tokenizer, max_new_tokens=6, do_sample=False
    )
    assert generated.explanations[0]["decoder_mode"] == "xai_single_stream"
    assert generated.explanations[0]["melody_conditioning"]["mode"] == "soft"
    assert not hasattr(generated, "stress_ids")
    from prosodia_lyricist.evaluation import conditional_perplexity

    source = restored.condition_source(**condition_inputs(inputs))
    ordinary = {
        k: inputs[k] for k in ("input_ids", "length_ids", "remainder_ids", "attention_mask")
    }
    score = conditional_perplexity(
        restored.template_model,
        {**ordinary, "inputs_embeds": source.inputs_embeds},
        inputs["labels"],
        tokenizer,
    )
    direct = restored.template_model(
        **ordinary, inputs_embeds=source.inputs_embeds, labels=inputs["labels"]
    )
    assert score["mean_nll"] == pytest.approx(direct.loss.item(), abs=1e-6)


def test_lyrics_alone_train_bridge_through_frozen_pretrained_decoder(
    joint_model,
    joint_song,
    tokenizer,
):
    model = joint_model.train()
    assert not model.template_model.training and not model.bridge.melody_encoder.training
    assert model.bridge.decoder.training
    assert not hasattr(model.bridge, "token_embedding")
    before = {k: v.clone() for k, v in model.template_model.state_dict().items()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    output = model(**batch(joint_song, tokenizer))
    output.loss.backward()
    for module in (
        model.bridge.strength_head,
        model.bridge.length_head,
        model.bridge.song_features[0],
    ):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.template_model.parameters())
    assert all(p.grad is None for p in model.bridge.melody_encoder.parameters())
    optimizer.step()
    assert all(torch.equal(v, model.template_model.state_dict()[k]) for k, v in before.items())


def test_source_has_no_reference_prosody_or_lyric_leak(joint_song, tokenizer, joint_model):
    changed = copy.deepcopy(joint_song)
    for line in changed["lines"]:
        line["text"] = "secret"
        line["words"] = []
        for syllable in line["syllables"]:
            syllable.update(stress="weak", length="short")
    first = encoded(joint_song, tokenizer, with_targets=False)
    second = encoded(changed, tokenizer, with_targets=False)
    assert first["source"] == second["source"]
    assert set(first) == set(second)
    model = joint_model.eval()
    inputs = batch(joint_song, tokenizer)
    before = model(**inputs)
    altered = {**inputs, "bridge_labels": inputs["bridge_labels"].clone()}
    active = altered["bridge_labels"].ne(-100)
    altered["bridge_labels"][active] = 3 - altered["bridge_labels"][active]
    after = model(**altered)
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.bridge_output.logits, after.bridge_output.logits)
    assert torch.equal(before.loss, after.loss)  # Anchor weight defaults to zero.


def test_hard_embeddings_match_existing_source_path(joint_model, joint_song, tokenizer):
    model = joint_model.eval()
    inputs = batch(joint_song, tokenizer)
    source = model.condition_source(**condition_inputs(inputs), template_mode="hard")
    predictions = decode_bridge_tokens(source.sequences[0].tolist())
    record = {
        **joint_song,
        "lines": [
            {**line, **prediction}
            for line, prediction in zip(joint_song["lines"], predictions, strict=True)
        ],
    }
    hard = {k: torch.tensor([v]) for k, v in encode_source(record, tokenizer, 8).items()}
    torch.testing.assert_close(
        source.inputs_embeds,
        model.template_model.embed_source(hard["input_ids"], hard["length_ids"]),
        rtol=0,
        atol=1e-7,
    )
    original = model.template_model(**hard, **{k: inputs[k] for k in TARGET_KEYS})
    joint = model(**inputs, template_mode="hard")
    torch.testing.assert_close(joint.logits, original.logits, rtol=0, atol=1e-7)
    for name in original.loss_components:
        torch.testing.assert_close(joint.loss_components[name], original.loss_components[name])


def test_decoder_still_receives_past_words_and_all_prosody_streams(
    joint_model,
    joint_song,
    tokenizer,
    monkeypatch,
):
    model = joint_model.eval()
    inputs = batch(joint_song, tokenizer)
    seen = []
    original = model.template_model.embed_target

    def capture(*args):
        seen.append(args)
        return original(*args)

    monkeypatch.setattr(model.template_model, "embed_target", capture)
    before = model(**inputs)
    for expected, actual in zip(
        (model.template_model.shift_features(inputs[k]) for k in TARGET_KEYS[1:]),
        seen[0][1:],
        strict=True,
    ):
        assert torch.equal(expected, actual)
    inputs["length_labels"][0, 2] = 2
    after = model(**inputs)
    assert torch.equal(before.bridge_output.logits, after.bridge_output.logits)
    assert torch.equal(before.logits[:, :3], after.logits[:, :3])
    assert not torch.equal(before.logits[:, 3:], after.logits[:, 3:])


def test_padding_is_inert_and_melody_changes_reach_slots(joint_model, joint_song, tokenizer):
    model = joint_model.eval()
    short = {**joint_song, "lines": joint_song["lines"][:1]}
    alone = model(**batch(short, tokenizer))
    together = EndToEndCollator(tokenizer.pad_token_id)(
        [
            encoded(short, tokenizer),
            encoded(joint_song, tokenizer),
        ]
    )
    output = model(**together)
    torch.testing.assert_close(
        alone.logits[0], output.logits[0, : alone.logits.shape[1]], rtol=1e-5, atol=1e-6
    )
    altered = copy.deepcopy(short)
    altered["lines"][0]["melody"]["note_duration_seconds"] = [1.5, 0.1, 2, 0.01]
    changed = model(**batch(altered, tokenizer))
    assert not torch.allclose(alone.bridge_output.logits, changed.bridge_output.logits)


def test_freeze_schedule_protects_shared_decoder_embeddings(joint_model):
    model = joint_model
    for mode in ("encoder", "all", "frozen"):
        model.set_template_trainable(mode)
        model.train()
        assert model.template_model.bart.get_encoder().layers[0].fc1.weight.requires_grad == (
            mode != "frozen"
        )
        for parameter in (
            model.template_model.bart.get_decoder().layers[0].fc1.weight,
            model.template_model.bart.get_decoder().embed_tokens.weight,
            model.template_model.length_embedding.weight,
            model.template_model.stress_embedding.weight,
        ):
            assert parameter.requires_grad == (mode == "all")
    model.bridge.set_melody_trainable(True)
    model.train()
    assert model.bridge.melody_encoder.training
    model.eval()
    assert not model.bridge.melody_encoder.training


def test_save_reload_and_generation_keep_soft_interface(
    joint_model,
    joint_song,
    tokenizer,
    tmp_path,
):
    model = joint_model.eval()
    inputs = batch(joint_song, tokenizer)
    before = model(**inputs)
    model.save(tmp_path, tokenizer)
    restored = EndToEndProsody.load(tmp_path).eval()
    after = restored(**inputs)
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.bridge_output.logits, after.bridge_output.logits)
    options = dict(tokenizer=tokenizer, max_new_tokens=9, prosody_correction=False)
    generated = restored.generate(**condition_inputs(inputs), **options)
    source = restored.condition_source(**condition_inputs(inputs))
    ordinary = restored.template_model.generate(
        **{k: inputs[k] for k in ("input_ids", "length_ids", "attention_mask")},
        inputs_embeds=source.inputs_embeds,
        **options,
    )
    assert torch.equal(generated.sequences, ordinary.sequences)
    assert torch.equal(generated.length_ids, ordinary.length_ids)
    assert generated.explanations[0]["melody_conditioning"]["mode"] == "soft"


def test_gradient_accumulation_uses_event_and_anchor_denominators(
    joint_model,
    joint_song,
    tokenizer,
):
    joint_model.anchor_weight = 0.2
    first = encoded(joint_song, tokenizer)
    second = encoded({**joint_song, "lines": joint_song["lines"][:1]}, tokenizer)
    collate = EndToEndCollator(tokenizer.pad_token_id)
    one, two = copy.deepcopy(joint_model), copy.deepcopy(joint_model)
    a = run_epoch(
        one,
        [collate([first, second])],
        torch.device("cpu"),
        optimizer=torch.optim.SGD(one.parameters(), lr=0.01),
    )
    b = run_epoch(
        two,
        [collate([first]), collate([second])],
        torch.device("cpu"),
        optimizer=torch.optim.SGD(two.parameters(), lr=0.01),
        accumulation_steps=2,
    )
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)
    assert a["optimizer_steps"] == b["optimizer_steps"] == 1
    for name, value in one.state_dict().items():
        torch.testing.assert_close(value, two.state_dict()[name], rtol=1e-5, atol=1e-7)


@pytest.fixture
def joint_config(tmp_path, joint_model, joint_song, tokenizer):
    template = tmp_path / "template"
    joint_model.template_model.save(template, tokenizer)
    directory = tmp_path / "data"
    directory.mkdir()
    data = dict(
        prepared_dir=str(directory),
        include_melody=True,
        stress_source="ipa",
        max_syllables=8,
        lines_per_window=2,
        pretraining_manifest=None,
    )
    songs, hashes = {}, {}
    for split in ("train", "valid", "test"):
        records = []
        for index in range(3):
            song = {**joint_song, "song_id": f"{split}-{index}"}
            if index == 1:
                song = {**song, "lines": song["lines"][:1]}
            songs[song["song_id"]] = dict(split=split, group=song["song_id"])
            records.append(song)
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        hashes[path.name] = file_sha256(path)
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=SCHEMA_VERSION,
                config=data,
                songs=songs,
                sha256=hashes,
                pronunciation={"rules": "test"},
            )
        )
    )
    return dict(
        data=data,
        model=dict(
            conditioning="end_to_end",
            template_checkpoint=str(template),
            melody_initialization="random",
            random_melody_encoder=dict(
                d_model=16,
                num_layers=1,
                num_heads=2,
                dim_feedforward=32,
                dropout=0,
                max_length=32,
            ),
            encoder_lines_per_window=2,
            max_window_notes=16,
            max_song_notes=32,
            max_song_lines=8,
            max_bridge_length=64,
            max_source_length=64,
            max_target_length=64,
            song_encoder_layers=1,
            slot_encoder=dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0),
        ),
        training=dict(
            output_dir=str(tmp_path / "runs"),
            epochs=1,
            batch_size=2,
            patience=2,
            gradient_accumulation_steps=2,
            learning_rate=0.01,
            melody_learning_rate=0.001,
            template_learning_rate=0.001,
            device="cpu",
            precision="fp32",
            seed=17,
            warmup_steps=0,
            schedule="constant_after_warmup",
        ),
    )


def test_joined_dataset_checks_all_limits_and_note_count_alignment(joint_config, tokenizer):
    config = joint_config
    options = dict(
        max_syllables=8,
        encoder_lines_per_window=2,
        max_window_notes=16,
        max_song_notes=32,
        max_song_lines=8,
        max_bridge_length=64,
        max_source_length=64,
        max_target_length=64,
    )
    dataset = EndToEndDataset(config["data"]["prepared_dir"], "train", tokenizer, **options)
    assert len(dataset) == 3
    assert dataset[0]["song_id"] == "train-0"
    assert dataset[0]["syllable_counts"] == [3, 1]
    notes = EndToEndDataset(
        config["data"]["prepared_dir"], "train", tokenizer, **options, count_source="notes"
    )
    assert notes[0]["syllable_counts"] == [4, 1]
    assert notes[0]["bridge_labels"][:5] == [-100] * 5
    reduced = EndToEndDataset(
        config["data"]["prepared_dir"], "train", tokenizer, **{**options, "max_song_lines": 1}
    )
    assert len(reduced) == 1 and reduced.skipped_limits["song_lines"] == 2


def test_full_training_saves_pretrained_decoder_unchanged(joint_config):
    output = train_end_to_end(joint_config, smoke_test=True)
    original = ProsodyBart.load(joint_config["model"]["template_checkpoint"])
    trained = EndToEndProsody.load(output / "best")
    assert all(
        torch.equal(v, trained.template_model.state_dict()[k])
        for k, v in original.state_dict().items()
    )
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert metrics["train"]["optimizer_steps"] == 1
    assert metrics["train"]["components"]["lyrics"] > 0
    assert metrics["template_trainable"] == "frozen"


@pytest.mark.parametrize("overrides", [None, {"stresses": 0.5}])
def test_training_defaults_to_equal_losses_despite_pretrained_weights(joint_config, overrides):
    original = ProsodyBart.load(joint_config["model"]["template_checkpoint"])
    assert original.loss_weights == dict(lyrics=1, syllables=0, stresses=0, lengths=0)
    if overrides is not None:
        joint_config["training"]["loss_weights"] = overrides
    model, _ = build_model(joint_config)
    assert model.template_model.loss_weights == {
        **dict(lyrics=1, syllables=1, stresses=1, lengths=1),
        **(overrides or {}),
    }


def test_checkpoint_selection_early_stopping_and_display_use_combined_loss(
    joint_config, monkeypatch, caplog
):
    joint_config["training"].update(epochs=5, patience=1)
    # Lyric CE worsens at epoch 2 and improves at epoch 3; combined loss does the opposite.
    validation = iter(
        [
            dict(lyrics=1, syllables=1, stresses=1, lengths=1),
            dict(lyrics=2, syllables=0.25, stresses=0.25, lengths=0.5),
            dict(lyrics=0.5, syllables=1, stresses=1, lengths=1),
        ]
    )
    validated, saved_epochs = [], []
    original_save = EndToEndProsody.save

    def run(model, loader, device, *, optimizer=None, **kwargs):
        components = (
            dict(lyrics=0.5, syllables=3, stresses=1.5, lengths=2)
            if optimizer is not None
            else next(validation)
        )
        metrics = dict(loss=sum(components.values()), components=components)
        if optimizer is None:
            validated.append(metrics)
        return metrics

    def save(model, directory, tokenizer):
        saved_epochs.append(len(validated))
        original_save(model, directory, tokenizer)

    monkeypatch.setattr("prosodia_lyricist.end_to_end_train.run_epoch", run)
    monkeypatch.setattr(EndToEndProsody, "save", save)
    with caplog.at_level("INFO", logger="prosodia_lyricist.end_to_end_train"):
        output = train_end_to_end(joint_config)

    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [m["valid"]["loss"] for m in metrics] == [4, 3, 3.5]
    assert saved_epochs == [1, 2]
    assert "Epoch 2 combined loss: train=7.0000 valid=3.0000" in caplog.text
    run = json.loads((output / "best" / "run.json").read_text())
    assert run["checkpoint_selection"] == (
        "combined validation loss (weighted decoder losses plus bridge anchor)"
    )


@pytest.mark.parametrize("decoder_mode", ["explainable", "xai"])
def test_evaluator_scores_lyrics_and_controls_same_songs(
    joint_model,
    joint_config,
    tokenizer,
    tmp_path,
    monkeypatch,
    decoder_mode,
):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.end_to_end_eval import evaluate_end_to_end

    def pronounce(text):
        return [
            {"text": word, "syllables": [{"ipa": "ˈaɪ", "stress": "strong", "length": "long"}]}
            for word in text.strip().split()
            if word.strip(".")
        ]

    monkeypatch.setattr(ipa, "parse_words", pronounce)
    if decoder_mode == "xai":
        joint_model = as_xai(joint_model, tokenizer)
    checkpoint = tmp_path / "joint"
    joint_model.eval().save(checkpoint, tokenizer)
    report = evaluate_end_to_end(
        checkpoint,
        joint_config["data"]["prepared_dir"],
        device="cpu",
        max_new_tokens=9,
        generation_limit=2,
        compare_hard=True,
        melody_controls=True,
    )
    assert report["data"]["songs"] == 3
    assert report["decoder_mode"] == decoder_mode
    assert report["generation"]["songs"] == 2
    assert report["hard_template_ablation"]["generation"]["songs"] == 2
    assert report["generation"]["compared_slots"] > 0
    assert report["duration_baselines"]["all_ipa_slots"]
    control = report["melody_controls"]
    assert control["paired_songs"] == 2 and control["unpaired_songs"] == 1
    assert [s["song_id"] for s in control["correct_generation"]["samples"]] == [
        s["song_id"] for s in control["wrong_generation"]["samples"]
    ]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("decoder_mode", ["explainable", "xai"])
def test_midi_inference_uses_soft_embeddings_for_generation_and_perplexity(
    joint_model,
    joint_config,
    joint_song,
    tokenizer,
    tmp_path,
    monkeypatch,
    decoder_mode,
):
    from prosodia_lyricist import infer as inference
    from prosodia_lyricist import midi

    if decoder_mode == "xai":
        joint_model = as_xai(joint_model, tokenizer)
    checkpoint = tmp_path / "joint-midi"
    joint_model.eval().save(checkpoint, tokenizer)
    (checkpoint / "run.json").write_text(
        json.dumps(
            dict(
                config=joint_config,
                data_schema_version=SCHEMA_VERSION,
            )
        )
    )
    source = {
        **joint_song,
        "lines": [
            {"melody": line["melody"], "unmarked_tail": False} for line in joint_song["lines"]
        ],
    }
    monkeypatch.setattr(
        "prosodia_lyricist.end_to_end_infer.midi_melody_record", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(midi, "midi_note_comparison", lambda *args, **kwargs: {"warning": None})
    monkeypatch.setattr(
        inference, "midi_note_comparison", lambda *args, **kwargs: {"warning": None}
    )
    seen = []
    model_class = XAILyricist if decoder_mode == "xai" else ProsodyBart
    original = model_class.generate

    def capture(self, *args, **kwargs):
        seen.append(kwargs["inputs_embeds"].clone())
        return original(self, *args, **kwargs)

    def score(model, inputs, *args, **kwargs):
        assert torch.equal(inputs["inputs_embeds"], seen[0])
        return dict(perplexity=2, token_count=1)

    monkeypatch.setattr(model_class, "generate", capture)
    monkeypatch.setattr(inference, "conditional_perplexity", score)
    monkeypatch.setattr(inference, "evaluate_prosody", lambda *args: {})
    path = tmp_path / "example.mid"
    path.write_bytes(b"MIDI stub; the note reader is replaced above")
    report = inference.infer(
        checkpoint,
        path,
        device="cpu",
        top_k=1,
        max_new_tokens=9,
        prosody_correction=False,
        return_report=True,
    )
    assert seen and report["template_role"] == "argmax_preview"
    assert report["bridge"]["count_sources"] == ["notes", "notes"]
    assert report["bridge"]["syllable_counts"] == [4, 1]
    assert report["decoder_mode"] == (
        "xai_single_stream" if decoder_mode == "xai" else "explainable"
    )


def test_literal_stress_tokens_in_prompts_do_not_become_slots(joint_song, tokenizer, joint_model):
    song = copy.deepcopy(joint_song)
    song["title"] = "<weak>"
    song["lines"][0]["keyword"] = "<strong>"
    inputs = batch(song, tokenizer)
    assert inputs["slot_indices"].ge(0).sum() == 4
    assert joint_model.eval()(**inputs).logits.isfinite().all()


def test_bad_slot_mapping_is_rejected(joint_model, joint_song, tokenizer):
    inputs = batch(joint_song, tokenizer)
    positions = inputs["slot_indices"][0].ge(0).nonzero().flatten()
    inputs["slot_indices"][0, positions[1]] = inputs["slot_indices"][0, positions[0]]
    with pytest.raises(ValueError, match="ordered bridge slots"):
        joint_model(**inputs)


def test_perplexity_uses_soft_inputs_instead_of_placeholder_ids(joint_model, joint_song, tokenizer):
    from prosodia_lyricist.evaluation import conditional_perplexity

    model = joint_model.eval()
    inputs = batch(joint_song, tokenizer)
    source = model.condition_source(**condition_inputs(inputs))
    encoded_source = {
        key: inputs[key].clone() for key in ("input_ids", "length_ids", "attention_mask")
    }
    encoded_source["inputs_embeds"] = source.inputs_embeds

    def score():
        return conditional_perplexity(
            model.template_model,
            encoded_source,
            inputs["labels"],
            tokenizer,
            prosody_labels=[inputs[key] for key in TARGET_KEYS[1:]],
        )

    original = score()
    encoded_source["input_ids"].fill_(tokenizer.unk_token_id)
    encoded_source["length_ids"].fill_(2)
    assert score() == original


def test_generation_requires_the_entire_conditioner_in_eval(joint_model, joint_song, tokenizer):
    assert not joint_model.template_model.training
    with pytest.raises(ValueError, match="model.eval"):
        joint_model.generate(**condition_inputs(batch(joint_song, tokenizer)), tokenizer=tokenizer)


def test_pretrained_tower_loads_exactly_without_reinitializing_template(joint_config, tmp_path):
    from prosodia_lyricist.melody_encoder.encoding import MELODY_REPRESENTATION
    from prosodia_lyricist.melody_encoder.modeling import MelodyTransformerEncoder

    architecture = joint_config["model"]["random_melody_encoder"]
    tower = MelodyTransformerEncoder(**architecture, projection_dim=8)
    checkpoint = tmp_path / "contrastive.pt"
    args = {
        f"melody_{key}": architecture[key]
        for key in ("d_model", "num_layers", "num_heads", "dim_feedforward")
    }
    args.update(audio_pooling="note", dropout=0, projection_dim=8, melody_max_length=32)
    torch.save(
        dict(
            melody_representation=MELODY_REPRESENTATION,
            args=args,
            model_state_dict={f"melody_encoder.{k}": v for k, v in tower.state_dict().items()},
        ),
        checkpoint,
    )
    joint_config["model"].update(
        melody_initialization="pretrained", melody_checkpoint=str(checkpoint)
    )
    model, _ = build_model(joint_config)
    for key, value in model.bridge.melody_encoder.state_dict().items():
        assert torch.equal(value, tower.state_dict()[key])
    assert model.bridge.melody_encoder.projection is None
    assert model.bridge.provenance["sha256"] == file_sha256(checkpoint)
    template = ProsodyBart.load(joint_config["model"]["template_checkpoint"])
    assert all(
        torch.equal(v, model.template_model.state_dict()[k])
        for k, v in template.state_dict().items()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 hardware unavailable",
)
def test_cuda_bf16_backward_through_frozen_decoder(joint_model, joint_song, tokenizer):
    model = joint_model.to("cuda").train()
    inputs = {k: v.to("cuda") for k, v in batch(joint_song, tokenizer).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**inputs)
    output.loss.backward()
    assert output.loss.isfinite()
    for head in (model.bridge.strength_head, model.bridge.length_head):
        assert head.weight.grad.isfinite().all() and head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.template_model.parameters())
