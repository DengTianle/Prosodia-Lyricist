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
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.model import ProsodyBart
from prosodia_lyricist.prepare import SCHEMA_VERSION


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


def test_evaluator_scores_lyrics_and_controls_same_songs(
    joint_model,
    joint_config,
    tokenizer,
    tmp_path,
    monkeypatch,
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


def test_midi_inference_uses_soft_embeddings_for_generation_and_perplexity(
    joint_model,
    joint_config,
    joint_song,
    tokenizer,
    tmp_path,
    monkeypatch,
):
    from prosodia_lyricist import infer as inference
    from prosodia_lyricist import midi

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
    original = ProsodyBart.generate

    def capture(self, *args, **kwargs):
        seen.append(kwargs["inputs_embeds"].clone())
        return original(self, *args, **kwargs)

    def score(model, inputs, *args, **kwargs):
        assert torch.equal(inputs["inputs_embeds"], seen[0])
        return dict(perplexity=2, token_count=1)

    monkeypatch.setattr(ProsodyBart, "generate", capture)
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
