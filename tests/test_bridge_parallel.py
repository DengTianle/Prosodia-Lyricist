"""Parallel supervised queries, XAI vocabularies, and unchanged lyric decoding."""

import copy
import json
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prosodia_lyricist.bridge_data import (
    PAIR_OFFSET,
    SLOT,
    BridgeDataset,
    bridge_collate,
    encode_bridge_song,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_eval import evaluate_bridge
from prosodia_lyricist.bridge_infer import predict_templates
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_readout import ParallelReadout
from prosodia_lyricist.bridge_train import evaluate_templates, run_bridge_epoch, train_bridge
from prosodia_lyricist.ipa import PAPER_RULES, REPO_RULES, REPO_STRESSES
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture(params=[("strong", "weak"), REPO_STRESSES])
def model(request):
    torch.manual_seed(17)
    return ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=16),
        bridge_scope="song",
        query_mode="parallel",
        query_attention="cross",
        strength_labels=request.param,
        encoder_lines_per_window=2,
        max_window_notes=16,
        max_song_notes=32,
        max_song_lines=4,
        max_target_length=32,
        max_syllables=8,
        song_encoder_layers=1,
        d_model=8,
        num_layers=2,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
    )


def song_lines(model):
    return [
        {
            "syllables": [
                dict(zip(("stress", "length"), model.pairs[(index * 2 + k) % len(model.pairs)]))
                for k in range(slots)
            ],
            "melody": {
                "midi_pitches": list(range(60, 60 + notes)),
                "onset_seconds": [index * 10 + k * 0.5 for k in range(notes)],
                "note_duration_seconds": [0.25 + k * 0.1 for k in range(notes)],
            },
        }
        for index, (notes, slots) in enumerate(((3, 2), (2, 1), (1, 2)))
    ]


def example(model, lines):
    return {
        **encode_bridge_song(lines),
        "labels": encode_bridge_targets(lines, 8, pairs=model.pairs),
    }


def batch_for(model):
    lines = song_lines(model)
    return bridge_collate([example(model, lines), example(model, lines[:1])], pairs=model.pairs)


def test_labels_only_supervise_loss_and_generation_is_the_same_parallel_pass(model):
    model.eval()
    batch = batch_for(model)
    output = model(**batch)
    slots = output.skeleton["tokens"].eq(SLOT)
    changed = {**batch, "labels": batch["labels"].clone()}
    changed["labels"][slots] = (batch["labels"][slots] - PAIR_OFFSET + 1) % len(
        model.pairs
    ) + PAIR_OFFSET
    after = model(**changed)
    torch.testing.assert_close(after.logits, output.logits, rtol=0, atol=0)
    assert output.loss != after.loss
    targets = batch["labels"][slots] - PAIR_OFFSET
    expected = F.cross_entropy(output.strength_logits[slots], targets // 2) + F.cross_entropy(
        output.length_logits[slots], targets % 2
    )
    torch.testing.assert_close(output.loss, expected)
    assert model.strength_head.out_features == len(model.strength_labels)
    assert model.length_head.out_features == 2
    assert isinstance(model.decoder, ParallelReadout)
    assert not hasattr(model, "token_embedding")
    assert not any("self_attn" in name for name, _ in model.decoder.named_modules())
    calls = []
    hook = model.decoder.register_forward_hook(lambda *args: calls.append(True))
    result = model.generate(**{k: v for k, v in batch.items() if k != "labels"})
    hook.remove()
    assert len(calls) == 1
    assert torch.equal(result.sequences, model.template_sequences(output))
    assert result.syllable_counts.tolist() == [[2, 1, 2], [2, 0, 0]]
    assert model.note_counts(batch["note_line_ids"], batch["line_counts"])[0].tolist() == [3, 2, 1]
    with pytest.raises(ValueError, match="no label history"):
        model(**batch, history_mask_probability=0.5)


def test_losses_train_both_heads_queries_and_encoder_across_windows(model):
    batch = batch_for(model)
    batch["song_features"].requires_grad_()
    output = model(**batch)
    gradient = torch.autograd.grad(
        output.strength_logits[0, 1, 0], batch["song_features"], retain_graph=True
    )[0]
    assert gradient[0, 5].abs().sum() > 0  # First syllable reads the final encoder window.
    output.loss.backward()
    for parameter in (
        model.melody_encoder.input_projection[0].weight,
        model.adapter[1].weight,
        model.song_features[0].weight,
        model.song_encoder.layers[0].self_attn.in_proj_weight,
        model.decoder.layers[0].cross_attention.in_proj_weight,
        model.query_type_embedding.weight,
        model.slot_embedding.weight,
        model.strength_head.weight,
        model.length_head.weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.set_melody_trainable(False)
    model.train()
    model(**batch).loss.backward()
    assert not model.melody_encoder.training
    assert all(p.grad is None for p in model.melody_encoder.parameters())
    assert model.adapter[1].weight.grad.abs().sum() > 0


def test_source_and_target_padding_are_ignored_and_checkpoint_is_self_contained(model, tmp_path):
    model.eval()
    batch = batch_for(model)
    output = model(**batch)
    padded = {}
    for key, value in batch.items():
        if key in ("line_counts", "syllable_counts"):
            padded[key] = value
        elif key in ("melody_features", "song_features"):
            padded[key] = F.pad(value, (0, 0, 0, 3), value=999)
        else:
            padded[key] = F.pad(value, (0, 3), value=-100 if key == "labels" else 0)
    after = model(**padded)
    slots = output.skeleton["tokens"].eq(SLOT)
    torch.testing.assert_close(
        after.logits[:, : output.logits.size(1)][slots], output.logits[slots]
    )
    torch.testing.assert_close(after.loss, output.loss)
    model.save(tmp_path)
    metadata = json.loads((tmp_path / "bridge.json").read_text())
    assert metadata["format_version"] == 8
    assert metadata["query_attention"] == "cross"
    assert "readout_config" in metadata and "decoder_config" not in metadata
    assert metadata["prosody_rules"] == (
        REPO_RULES if len(model.strength_labels) == 3 else PAPER_RULES
    )
    restored = ProsodyBridge.load(tmp_path).eval()
    torch.testing.assert_close(restored(**batch).logits, output.logits, rtol=0, atol=0)
    metadata["query_attention"] = "self_cross"
    (tmp_path / "bridge.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature rules"):
        ProsodyBridge.load(tmp_path)


def test_parallel_accumulation_weights_each_supervised_slot(model):
    combined = copy.deepcopy(model)
    lines = song_lines(model)
    a, b = example(model, lines), example(model, lines[:1])
    for net, batches, accumulation in (
        (
            model,
            [bridge_collate([a], pairs=model.pairs), bridge_collate([b], pairs=model.pairs)],
            3,
        ),
        (combined, [bridge_collate([a, b], pairs=model.pairs)], 1),
    ):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        run_bridge_epoch(
            net,
            batches,
            "cpu",
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            accumulation_steps=accumulation,
            gradient_clip=None,
        )
    for a, b in zip(model.parameters(), combined.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def prepare_fixture(model, directory):
    directory.mkdir()
    data = dict(
        prepared_dir=str(directory),
        include_melody=True,
        stress_source="ipa",
        prosody_mode="actual" if len(model.strength_labels) == 3 else "paper",
        max_syllables=8,
        lines_per_window=2,
        pretraining_manifest=None,
    )
    songs, hashes = {}, {}
    for split in ("train", "valid", "test"):
        records = [
            {"song_id": f"{split}-{i}", "lines": song_lines(model)[:width]}
            for i, width in enumerate((3, 1))
        ]
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        hashes[path.name] = file_sha256(path)
        songs.update({r["song_id"]: dict(split=split, group=r["song_id"]) for r in records})
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=SCHEMA_VERSION,
                config=data,
                songs=songs,
                sha256=hashes,
                pronunciation=dict(rules=model.prosody_rules),
            )
        )
    )
    return data


def test_standalone_training_and_single_pass_evaluation(model, tmp_path, monkeypatch):
    data = prepare_fixture(model, tmp_path / "data")
    config = dict(
        data=data,
        model=dict(
            conditioning="bridge",
            melody_initialization="random",
            random_melody_encoder=model.melody_config | {"projection_dim": None},
            encoder_lines_per_window=2,
            max_window_notes=16,
            max_song_notes=32,
            max_song_lines=4,
            max_target_length=32,
            song_encoder_layers=1,
            bridge_readout=model.decoder_config,
        ),
        training=dict(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            seed=17,
            epochs=1,
            batch_size=2,
            patience=2,
            learning_rate=1e-3,
            melody_learning_rate=1e-3,
            melody_unfreeze_epoch=0,
            warmup_steps=0,
            schedule="constant_after_warmup",
        ),
    )
    config["model"]["random_melody_encoder"].pop("projection_dim")
    output = train_bridge(config)
    run = json.loads((output / "run.json").read_text())
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert run["query_mode"] == "parallel" and run["query_attention"] == "cross"
    assert "history_mask_probability" not in metrics["train"]
    assert "teacher_forced_pair_accuracy" not in metrics["valid"]
    assert metrics["valid"]["templates"]["compared_slots"] == 7
    loaded = ProsodyBridge.load(output / "best")
    assert loaded.strength_labels == model.strength_labels
    calls, original = [], ProsodyBridge.forward

    def capture(self, **batch):
        calls.append(True)
        return original(self, **batch)

    monkeypatch.setattr(ProsodyBridge, "forward", capture)
    report = evaluate_bridge(output / "best", data["prepared_dir"], device="cpu", batch_size=2)
    assert len(calls) == 1
    assert report["schema_version"] == 4
    assert "teacher_forced" not in report and "generation" not in report
    assert report["prediction"]["slots"] == report["prediction"]["compared_slots"] == 7
    dataset = BridgeDataset(data["prepared_dir"], "test", **loaded.dataset_options)
    loader = DataLoader(
        dataset, batch_size=2, collate_fn=partial(bridge_collate, pairs=model.pairs)
    )
    generated = evaluate_templates(loaded, loader, "cpu")
    assert generated["pair_accuracy"] == report["prediction"]["accuracy"]["combined"]
    assert generated["prosody_bleu"] == report["prediction"]["prosody_bleu"]
    assert report["duration_baselines"]["all_ipa_slots"]["compared_slots"] == 7
    json.dumps(report, allow_nan=False)
    incompatible = ("strong", "weak") if len(model.strength_labels) == 3 else REPO_STRESSES
    with pytest.raises(ValueError, match="Prepared IPA rules"):
        BridgeDataset(
            data["prepared_dir"],
            "test",
            **{
                **loaded.dataset_options,
                "strength_labels": incompatible,
            },
        )


def test_midi_queries_preserve_secondary_stress_and_xai_decoding(
    tmp_path, tiny_model, tokenizer, monkeypatch
):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.infer import infer, main
    from prosodia_lyricist.xai_features import encode_source
    from prosodia_lyricist.xai_model import XAILyricist

    midi = pytest.importorskip("miditoolkit")
    song = midi.MidiFile(ticks_per_beat=480)
    track = midi.Instrument(0)
    track.notes = [
        midi.Note(80, 60 + i, start, start + 240) for i, start in enumerate((0, 480, 960, 1920))
    ]
    song.instruments.append(track)
    song.markers = [midi.Marker("end", 720), midi.Marker("end", 1680)]
    song.tempo_changes = [midi.TempoChange(90, 0), midi.TempoChange(120, 900)]
    song.time_signature_changes = [midi.TimeSignature(3, 4, 0)]
    path = tmp_path / "song.mid"
    song.dump(str(path))
    bridge = ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=16),
        bridge_scope="song",
        query_mode="parallel",
        query_attention="cross",
        strength_labels=REPO_STRESSES,
        encoder_lines_per_window=2,
        max_song_notes=32,
        max_song_lines=4,
        max_target_length=32,
        max_syllables=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
    )
    with torch.no_grad():
        bridge.strength_head.weight.zero_()
        bridge.strength_head.bias.copy_(torch.tensor([-1000.0, 1000.0, -1000.0]))
        bridge.length_head.weight.zero_()
        bridge.length_head.bias.copy_(torch.tensor([-1000.0, 1000.0]))
    bridge.save(tmp_path / "bridge")
    records, metadata = predict_templates(tmp_path / "bridge", path)
    assert metadata["decoding"] == "parallel_queries"
    assert metadata["prosody_mode"] == "actual"
    assert metadata["encoder_windows"] == 2
    assert metadata["syllable_counts"] == metadata["note_counts"] == [2, 1, 1]
    assert records[-1]["unmarked_tail"]
    assert all(
        s == dict(stress="substrong", length="short") for r in records for s in r["syllables"]
    )
    checkpoint = tmp_path / "xai"
    template = XAILyricist(tiny_model.bart, max_syllables=8).eval()
    with torch.no_grad():
        template.bart.final_logits_bias.fill_(-1000)
        template.bart.final_logits_bias[0, tokenizer.convert_tokens_to_ids("hello")] = 1000
    template.save(checkpoint, tokenizer)
    (checkpoint / "run.json").write_text(
        json.dumps(
            dict(
                data_schema_version=SCHEMA_VERSION,
                config=dict(
                    data=dict(stress_source="ipa", prosody_mode="actual"),
                    model=dict(max_source_length=64, max_target_length=64),
                ),
            )
        )
    )
    weights_hash = file_sha256(checkpoint / "weights.pt")

    def parse(text, **kwargs):
        assert kwargs.get("prosody_mode") == "actual"
        return [
            dict(text=w, syllables=[dict(ipa="ˌə", stress="substrong", length="short")])
            for w in text.split()
        ]

    monkeypatch.setattr(ipa, "parse_words", parse)
    options = dict(
        bridge_checkpoint=tmp_path / "bridge",
        device="cpu",
        title="a song",
        top_k=1,
        max_new_tokens=4,
        return_report=True,
    )
    report = infer(checkpoint, path, **options)
    assert report["decoder_mode"] == "xai_single_stream"
    assert report["encoded_source"] == encode_source(
        dict(title="a song", lines=records), tokenizer, 8
    )
    assert [n for n in report["encoded_source"]["remainder_ids"] if n] == [2, 1, 1, 1]
    assert (
        tokenizer.convert_tokens_to_ids("<keywords>") not in report["encoded_source"]["input_ids"]
    )
    other = infer(checkpoint, path, reference_lines=["hello", "world", "light"], **options)
    assert other["encoded_source"] == report["encoded_source"]
    assert other["generated_token_ids"] == report["generated_token_ids"]
    assert file_sha256(checkpoint / "weights.pt") == weights_hash
    prefix = tmp_path / "preview"
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--bridge-checkpoint",
            str(tmp_path / "bridge"),
            "--midi",
            str(path),
            "--template-only",
            "--device",
            "cpu",
            "--report-prefix",
            str(prefix),
        ],
    )
    main()
    assert (
        json.loads(prefix.with_suffix(".json").read_text())["bridge"]["decoding"]
        == "parallel_queries"
    )


def test_old_history_training_configuration_is_rejected():
    with pytest.raises(ValueError, match="Remove training.history_mask_probability"):
        train_bridge(
            dict(
                data={},
                model={},
                training=dict(
                    batch_size=1,
                    learning_rate=1e-4,
                    melody_learning_rate=2e-6,
                    history_mask_probability=0.3,
                ),
            )
        )
