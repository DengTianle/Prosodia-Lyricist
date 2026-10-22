"""Observed word alignment, direct note heads, masked CE and complete inference."""

import copy
import json
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prosodia_lyricist.bridge_alignment import (
    ALIGNMENT_SCHEME,
    aligned_note_indices,
    prepare_note_alignment,
    recover_note_alignment,
)
from prosodia_lyricist.bridge_data import (
    PAIR_OFFSET,
    BridgeDataset,
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_song,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_eval import evaluate_bridge
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import run_bridge_epoch, train_bridge
from prosodia_lyricist.ipa import REPO_STRESSES
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture
def model():
    torch.manual_seed(17)
    return ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=16),
        bridge_scope="song",
        query_mode="note",
        strength_labels=REPO_STRESSES,
        encoder_lines_per_window=1,
        max_window_notes=16,
        max_song_notes=32,
        max_song_lines=4,
        max_target_length=32,
        max_syllables=8,
        song_encoder_layers=1,
        d_model=8,
        num_heads=2,
        dim_feedforward=16,
        dropout=0,
    )


def lines():
    result = []
    for index, (word_info, note_texts) in enumerate(
        (
            ((("hello", 2, 3), ("every", 3, 2)), ("hel", "~", "lo", "e", "very")),
            ((("light", 1, 2), ("you", 1, 1)), ("light", "~", "you")),
        )
    ):
        syllables = [
            dict(stress=REPO_STRESSES[i % 3], length=("long", "short")[i % 2])
            for i in range(sum(info[1] for info in word_info))
        ]
        raw_notes = [
            {"text": text, "time": [index * 10 + i, index * 10 + i + 0.5]}
            for i, text in enumerate(note_texts)
        ]
        raw_words, position = [], 0
        for text, _, count in word_info:
            raw_words.append((text, raw_notes[position : position + count]))
            position += count
        line = {
            "line_id": index,
            "words": [dict(text=text, syllable_count=count) for text, count, _ in word_info],
            "syllables": syllables,
            "melody": dict(
                midi_pitches=[60 + i for i in range(len(raw_notes))],
                onset_seconds=[n["time"][0] for n in raw_notes],
                note_duration_seconds=[0.5] * len(raw_notes),
            ),
        }
        line["note_alignment"] = prepare_note_alignment(line, raw_words)
        result.append(line)
    return result


def example(source, policy="strict"):
    indices, _ = aligned_note_indices(source, policy)
    return {
        **encode_bridge_song(source, 1),
        "labels": encode_bridge_targets(
            source, 8, pairs=tuple((s, n) for s in REPO_STRESSES for n in ("long", "short"))
        ),
        "target_note_indices": indices,
    }


def batch_for(model):
    source = lines()
    return bridge_collate([example(source), example(source[:1])], pairs=model.pairs)


def test_word_local_strict_and_prefix_mapping_never_shift_following_words():
    strict, counts = aligned_note_indices(lines())
    prefix, _ = aligned_note_indices(lines(), "prefix")
    assert strict == [-1, 0, 2, -1, -1, -1, -1, 5, 7]
    assert prefix == [-1, 0, 2, 3, 4, -1, -1, 5, 7]
    assert counts["mismatched_words"] == 1
    assert counts["supervised_slots"] == 4 and counts["masked_slots"] == 3
    source = lines()
    source[0]["words"][1]["syllable_count"] = 1
    source[0]["syllables"] = source[0]["syllables"][:3]
    source[0]["note_alignment"]["words"][1]["syllable_indices"] = [2]
    assert aligned_note_indices(source, "strict")[0] == [-1, 0, 2, -1, -1, 5, 7]
    assert aligned_note_indices(source, "prefix")[0] == [-1, 0, 2, 3, -1, 5, 7]


def test_hyphen_tokenization_uses_raw_word_boundaries_and_unmatched_text_is_masked():
    source = {
        "words": [dict(text="sun", syllable_count=1), dict(text="light", syllable_count=1)],
        "syllables": [{}, {}],
    }
    notes = [{"text": text, "time": [i, i + 0.5]} for i, text in enumerate(("sun", "light"))]
    aligned = prepare_note_alignment(source, [("sun-light", notes)])
    assert aligned["words"] == [dict(note_starts=[0, 1], syllable_indices=[0, 1])]
    assert "unmatched_reason" not in aligned
    assert prepare_note_alignment(source, [("moon-light", notes)])["words"] == []


def test_direct_heads_gather_first_notes_and_gold_labels_only_affect_loss(model):
    model.eval()
    batch = batch_for(model)
    output = model(**batch)
    source = {
        k: v
        for k, v in batch.items()
        if k not in ("labels", "syllable_counts", "target_note_indices")
    }
    notes = model.predict_notes(**source)
    assert notes.strength_logits.shape == (2, 8, 3)
    assert not any(
        hasattr(model, key)
        for key in (
            "decoder",
            "query_type_embedding",
            "slot_embedding",
            "count_embedding",
            "token_embedding",
        )
    )
    active = output.target_mask
    assert int(active.sum()) == 6
    for row, position in active.nonzero().tolist():
        torch.testing.assert_close(
            output.logits[row, position],
            notes.logits[row, batch["target_note_indices"][row, position]],
        )
    targets = batch["labels"][active] - PAIR_OFFSET
    torch.testing.assert_close(
        output.loss,
        F.cross_entropy(output.strength_logits[active], targets // 2)
        + F.cross_entropy(output.length_logits[active], targets % 2),
    )
    changed = {**batch, "labels": batch["labels"].clone()}
    pair_slots = changed["labels"].ge(PAIR_OFFSET) & changed["labels"].lt(model.first_line)
    changed["labels"][pair_slots & ~active] = PAIR_OFFSET
    torch.testing.assert_close(model(**changed).loss, output.loss, rtol=0, atol=0)
    changed["labels"][active] = PAIR_OFFSET + (targets + 1) % len(model.pairs)
    after = model(**changed)
    torch.testing.assert_close(after.logits, output.logits, rtol=0, atol=0)
    assert after.loss != output.loss
    generated = model.generate(**source)
    assert generated.syllable_counts.tolist() == [[5, 3], [5, 0]]
    assert [
        len(line["syllables"])
        for line in decode_bridge_tokens(generated.sequences[0].tolist(), pairs=model.pairs)
    ] == [5, 3]
    with pytest.raises(ValueError, match="requires observed"):
        model.generate(**source, syllable_counts=batch["syllable_counts"])


def test_gradients_reach_both_heads_and_global_context_and_padding_is_ignored(model, tmp_path):
    model.eval()
    batch = batch_for(model)
    batch["song_features"].requires_grad_()
    output = model(**batch)
    gradient = torch.autograd.grad(
        output.strength_logits[0, 1, 0], batch["song_features"], retain_graph=True
    )[0]
    assert gradient[0, 7].abs().sum() > 0
    output.loss.backward()
    for parameter in (
        model.strength_head.weight,
        model.length_head.weight,
        model.song_encoder.layers[0].self_attn.in_proj_weight,
        model.melody_encoder.input_projection[0].weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    padded = dict(batch)
    for key in ("melody_features", "song_features"):
        padded[key] = F.pad(batch[key], (0, 0, 0, 2), value=99)
    for key in ("melody_attention_mask", "note_line_ids"):
        padded[key] = F.pad(batch[key], (0, 2), value=0)
    padded["labels"] = F.pad(batch["labels"], (0, 2), value=-100)
    padded["target_note_indices"] = F.pad(batch["target_note_indices"], (0, 2), value=-1)
    torch.testing.assert_close(model(**padded).loss, output.loss, atol=1e-6, rtol=1e-5)
    model.save(tmp_path)
    metadata = json.loads((tmp_path / "bridge.json").read_text())
    assert metadata["format_version"] == 9 and metadata["query_attention"] == "none"
    assert "decoder_config" not in metadata and "readout_config" not in metadata
    restored = ProsodyBridge.load(tmp_path).eval()
    torch.testing.assert_close(restored(**batch).logits, output.logits, rtol=0, atol=0)
    metadata["alignment_scheme"] = "unknown"
    (tmp_path / "bridge.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature rules"):
        ProsodyBridge.load(tmp_path)


@pytest.mark.parametrize(
    "indices",
    [
        [-1, 0, 0, -1, -1, -1, -1, 5, 7],  # Duplicate note.
        [-1, 0, 5, -1, -1, -1, -1, 5, 7],  # Wrong phrase.
        [0, 0, 2, -1, -1, -1, -1, 5, 7],  # Prefix is not a note target.
        [-1, 0, 8, -1, -1, -1, -1, 5, 7],  # Outside source.
    ],
)
def test_invalid_alignment_is_rejected(model, indices):
    batch = bridge_collate([example(lines())], pairs=model.pairs)
    batch["target_note_indices"] = torch.tensor([indices])
    with pytest.raises(ValueError, match="[Nn]ote|prefix"):
        model(**batch)


def test_accumulation_uses_supervised_slots_and_masked_phrase_metrics(model):
    combined = copy.deepcopy(model)
    a, b = example(lines()), example(lines()[:1])
    for net, batches, accumulation in (
        (
            model,
            [bridge_collate([a], pairs=model.pairs), bridge_collate([b], pairs=model.pairs)],
            3,
        ),
        (combined, [bridge_collate([a, b], pairs=model.pairs)], 1),
    ):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        metrics = run_bridge_epoch(
            net,
            batches,
            "cpu",
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            accumulation_steps=accumulation,
            gradient_clip=None,
            template_metrics=True,
        )
        assert metrics["tokens"] == metrics["templates"]["compared_slots"] == 6
        assert metrics["templates"]["fully_supervised_phrases"] == 1
        assert metrics["templates"]["masked_slots"] == 6
    for left, right in zip(model.parameters(), combined.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_training_reload_and_evaluation_use_identical_aligned_baseline_coverage(model, tmp_path):
    data = dict(
        prepared_dir=str(tmp_path),
        include_melody=True,
        stress_source="ipa",
        prosody_mode="actual",
        max_syllables=8,
        lines_per_window=1,
        pretraining_manifest=None,
    )
    songs, hashes = {}, {}
    for split in ("train", "valid"):
        song = dict(song_id=split, lines=lines())
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(song) + "\n")
        hashes[path.name] = file_sha256(path)
        songs[split] = dict(split=split, group=split)
    (tmp_path / "manifest.json").write_text(
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
    tower = {k: v for k, v in model.melody_config.items() if k != "projection_dim"}
    config = dict(
        data=data,
        model=dict(
            query_mode="note",
            melody_initialization="random",
            random_melody_encoder=tower,
            encoder_lines_per_window=1,
            max_window_notes=16,
            max_song_notes=32,
            max_song_lines=4,
            max_target_length=32,
            song_encoder_layers=1,
            song_encoder={k: v for k, v in model.decoder_config.items() if k != "num_layers"},
        ),
        training=dict(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            seed=17,
            epochs=1,
            batch_size=1,
            patience=1,
            learning_rate=1e-3,
            melody_learning_rate=1e-3,
            melody_unfreeze_epoch=0,
            warmup_steps=0,
            schedule="constant_after_warmup",
        ),
    )
    output = train_bridge(config)
    report = evaluate_bridge(output / "best", tmp_path, split="valid", device="cpu")
    assert report["schema_version"] == 5 and report["query_mode"] == "note"
    assert report["prediction"]["slots"] == report["prediction"]["compared_slots"] == 4
    assert report["prediction"]["fully_supervised_phrases"] == 1
    assert report["duration_baselines"]["all_ipa_slots"]["compared_slots"] == 4
    assert report["duration_baselines"]["masked_ipa_slots"] == 3
    assert report["data"]["alignment_counts"]["mismatched_words"] == 1
    assert "teacher_forced" not in report and "generation" not in report
    json.dumps(report, allow_nan=False)
    restored = ProsodyBridge.load(output / "best")
    ds = BridgeDataset(tmp_path, "valid", **restored.dataset_options)
    loader = DataLoader(ds, batch_size=1, collate_fn=partial(bridge_collate, pairs=model.pairs))
    metrics = run_bridge_epoch(restored, loader, "cpu", template_metrics=True)
    assert metrics["pair_accuracy"] == report["prediction"]["accuracy"]["combined"]


def test_old_prepared_data_recovers_links_without_pronunciation_or_rewriting(
    annotation,
    tmp_path,
    monkeypatch,
):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.dali import extract_lines
    from prosodia_lyricist.melody_data import attach_melody

    monkeypatch.setattr(
        ipa,
        "parse_words",
        lambda text: [
            dict(text="hello", syllables=[dict(stress="weak", length="short")] * 2),
            dict(text="world", syllables=[dict(stress="strong", length="long")]),
        ],
    )
    annot = annotation["annotations"]["annot"]
    for note in annot["notes"]:
        note["freq"] = [440.0]
    records, rejected = extract_lines(annotation["info"], annot, stress_source="ipa")
    assert not rejected
    attach_melody(records, annot)
    original = copy.deepcopy(records[0].pop("note_alignment"))
    path = tmp_path / "song-a.json"
    path.write_text(json.dumps(annotation))
    before = path.read_bytes()
    monkeypatch.setattr(ipa, "parse_words", lambda *a, **kw: pytest.fail("Must reuse prepared IPA"))
    song = dict(song_id="song-a", lines=records)
    assert recover_note_alignment(song, tmp_path) == file_sha256(path)
    assert records[0]["note_alignment"] == original
    assert original["scheme"] == ALIGNMENT_SCHEME
    assert path.read_bytes() == before
    records[0].pop("note_alignment")
    records[0]["melody"]["midi_pitches"][0] += 1
    with pytest.raises(ValueError, match="differ from the prepared"):
        recover_note_alignment(song, tmp_path)
