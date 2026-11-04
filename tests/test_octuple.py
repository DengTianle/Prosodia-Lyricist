"""Musical invariants, source-only tensors, and role-aware bridge behavior."""

import copy
import gzip
import json
from fractions import Fraction
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prosodia_lyricist.bridge_data import (
    BridgeDataset,
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_song,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_eval import evaluate_bridge
from prosodia_lyricist.bridge_infer import predict_midi_templates
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import run_bridge_epoch, train_bridge
from prosodia_lyricist.ipa import REPO_RULES
from prosodia_lyricist.lakh_octuple import prepare_canonical_lakh
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.melody_encoder.octuple import (
    GRID_POLICY,
    OFFSET,
    REPRESENTATION,
    START,
    MusicalClock,
    canonical_meter,
    codec_metadata,
    encode_octuple,
    time_centers,
)
from prosodia_lyricist.midi import midi_melody_record
from prosodia_lyricist.prepare import SCHEMA_VERSION
from prosodia_lyricist.split_integrity import content_fingerprints


def source(ppq=480):
    return dict(
        ticks_per_quarter=ppq,
        meter_phase=GRID_POLICY,
        time_signatures=[
            dict(tick=0, numerator=4, denominator=4),
            dict(tick=ppq, numerator=4, denominator=4),
        ],
        tempo_map=[dict(tick=0, tempo=500000), dict(tick=ppq // 2, tempo=250000)],
    )


def phrase(ppq=480, variant=0):
    clock = MusicalClock(source(ppq))
    onsets, durations = [0, ppq // 3, ppq], [ppq // 3, 2 * ppq // 3, ppq]
    return dict(
        melody=dict(
            midi_pitches=[60, 62 + variant, 64],
            onset_ticks=onsets,
            note_duration_ticks=durations,
            onset_seconds=[clock.seconds_at(Fraction(t, ppq)) for t in onsets],
            note_duration_seconds=[
                clock.seconds_at(Fraction(t + d, ppq)) - clock.seconds_at(Fraction(t, ppq))
                for t, d in zip(onsets, durations, strict=True)
            ],
        ),
        syllables=[dict(stress="strong", length="short"), dict(stress="weak", length="long")],
        note_roles=[0, 1, 0],
        note_alignment=dict(
            scheme="lakh_word_note_starts_v1",
            note_count=3,
            words=[dict(note_starts=[0, 2], syllable_indices=[0, 1])],
        ),
    )


def model(**options):
    return ProsodyBridge(
        dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0, max_length=32),
        max_song_notes=32,
        max_song_lines=4,
        max_target_length=32,
        representation=REPRESENTATION,
        **options,
    )


def example(variant=0):
    line = phrase(variant=variant)
    m = model()
    return {
        **encode_bridge_song([line], source=source(), representation=REPRESENTATION),
        "labels": encode_bridge_targets([line], 40, pairs=m.pairs),
        "target_note_indices": [-1, 0, 2],
        "note_roles": [0, 1, 0],
    }


def source_batch(batch):
    return {
        k: v
        for k, v in batch.items()
        if k not in ("labels", "syllable_counts", "target_note_indices", "note_roles")
    }


@pytest.mark.parametrize("resolution", [48, 96])
def test_ppq_triplets_continuous_meter_and_source_only_inputs(resolution):
    config = dict(resolution=resolution)
    a = encode_octuple([phrase(480)], source(480), config)
    b = encode_octuple([phrase(960)], source(960), config)
    np.testing.assert_array_equal(a["melody_features"], b["melody_features"])
    np.testing.assert_array_equal(a["timing_residuals"], b["timing_residuals"])
    assert a["melody_features"][:, 1].tolist() == [
        OFFSET,
        OFFSET + resolution // 3,
        OFFSET + resolution,
    ]
    assert a["melody_features"][0, 3] == START
    assert a["melody_features"][0, 5] == START
    assert a["melody_features"][2, 5] == OFFSET  # zero gap remains distinct from START
    changed = phrase()
    changed.update(text="unrelated", syllables=[], note_roles=[1, 1, 1], note_alignment={})
    c = encode_octuple([changed], source(), config)
    np.testing.assert_array_equal(a["melody_features"], c["melody_features"])
    np.testing.assert_array_equal(a["timing_residuals"], c["timing_residuals"])


def test_all_linear_time_bin_ties_round_up_exactly():
    from prosodia_lyricist.melody_encoder.octuple import _time_code

    for resolution in (48, 96):
        centers = time_centers(resolution)
        for j in range(4 * resolution):
            code, residual = _time_code(Fraction(2 * j + 1, 2 * resolution), centers, resolution)
            assert code - OFFSET == j + 1
            assert residual == -0.5


def test_bar_grid_changes_partial_bars_and_pickup_preserves_phase():
    s = source()
    s["time_signatures"].append(dict(tick=480 * 6, numerator=6, denominator=8))
    clock = MusicalClock(s)
    assert clock.meter(Fraction(5)) == (1, 1, 4, 4)
    assert clock.meter(Fraction(6)) == (2, 0, 6, 8)
    assert clock.meter(Fraction(9)) == (3, 0, 6, 8)
    assert clock.diagnostics == dict(redundant_signatures=1, off_bar_changes=1)
    s["bar_origin_tick"] = -480
    assert MusicalClock(s).meter(Fraction(0)) == (0, 1, 4, 4)
    p = phrase()
    p["melody"]["onset_ticks"] = [480 * 3, 480 * 1100, 480 * 1101]
    encoded = encode_octuple([p], source())
    assert encoded["melody_features"][0, 1] == OFFSET + 288
    assert encoded["melody_features"][1, 0] > OFFSET + 255


def test_tiny_notes_prebar_rounding_and_time_residuals():
    p = phrase()
    p["melody"].update(onset_ticks=[0, 479, 1919], note_duration_ticks=[1, 1, 1])
    encoded = encode_octuple([p], source())
    assert len(encoded["melody_features"]) == 3
    assert encoded["melody_features"][2, 1] == OFFSET + 383
    assert encoded["timing_residuals"][2, 0] == pytest.approx(0.8)
    centers = time_centers()
    for row in range(3):
        center = centers[encoded["melody_features"][row, 4] - OFFSET]
        restored = center + encoded["timing_residuals"][row, 1] / 96
        assert restored == pytest.approx(1 / 480, abs=1e-8)


def test_meter_identity_tempo_changes_missing_conflicting_metadata():
    clock = MusicalClock(source())
    assert clock.seconds_at(Fraction(1)) == 0.375
    assert clock.tempo(Fraction(1)) == 240
    s = source()
    s["time_signatures"] = [dict(tick=0, numerator=3, denominator=4)]
    triple = encode_octuple([phrase()], s)
    s["time_signatures"] = [dict(tick=0, numerator=6, denominator=8)]
    compound = encode_octuple([phrase()], s)
    assert triple["melody_features"][0, 6] != compound["melody_features"][0, 6]
    s["time_signatures"] = []
    with pytest.raises(ValueError, match="Missing explicit meter"):
        encode_octuple([phrase()], s)
    s = source()
    s["time_signatures"].append(dict(tick=0, numerator=3, denominator=4))
    with pytest.raises(ValueError, match="Conflicting"):
        encode_octuple([phrase()], s)


def test_padding_gradients_roundtrip_and_vocabulary_guard(tmp_path):
    m = model(predict_note_roles=True)
    e = example()
    batch = bridge_collate([e], pairs=m.pairs)
    assert batch["melody_features"].dtype == torch.long
    out = m(**batch)
    out.loss.backward()
    assert m.melody_encoder.input_projection.fields["pitch"].weight.grad.abs().sum() > 0
    assert m.role_head.weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        hidden = m.encode(**source_batch(batch))
        padded = {k: v.clone() for k, v in source_batch(batch).items()}
        for k, v in list(padded.items()):
            if v.ndim >= 2:
                padded[k] = torch.cat((v, torch.zeros_like(v[:, :2])), dim=1)
        torch.testing.assert_close(m.encode(**padded)[:, :3], hidden)
    m.save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    torch.testing.assert_close(
        m.generate(**source_batch(batch)).sequences,
        restored.generate(**source_batch(batch)).sequences,
    )
    metadata = json.loads((tmp_path / "bridge.json").read_text())
    metadata["octuple_codec"]["sha256"] = "invalid"
    (tmp_path / "bridge.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="quantizer"):
        ProsodyBridge.load(tmp_path)


def test_melisma_grouping_identical_syllables_and_phrase_first_constraint():
    m = model(predict_note_roles=True)
    strength = torch.tensor([[[8.0, 0.0, 0.0]] * 3])
    length = torch.tensor([[[8.0, 0.0]] * 3])
    notes = SimpleNamespace(
        strength_logits=strength,
        length_logits=length,
        logits=(strength.unsqueeze(-1) + length.unsqueeze(-2)).flatten(-2),
        role_logits=torch.tensor([[[0.0, 8.0], [0.0, 8.0], [8.0, 0.0]]]),
    )
    output = m.generate_from_notes(
        notes, torch.ones(1, 3), torch.ones(1, 3, dtype=torch.long), torch.tensor([1])
    )
    assert output.note_start_mask.tolist() == [[True, False, True]]
    assert output.syllable_counts.tolist() == [[2]]
    syllables = decode_bridge_tokens(output.sequences[0].tolist(), pairs=m.pairs)[0]["syllables"]
    assert len(syllables) == 2 and syllables[0] == syllables[1]


def test_role_loss_accumulation_has_independent_denominator():
    a = model(predict_note_roles=True, role_class_weights=[0.6, 3.0])
    b = copy.deepcopy(a)
    first, second = example(), example(1)
    second["target_note_indices"][-1] = -1
    second["note_roles"][-1] = -100
    collate = partial(bridge_collate, pairs=a.pairs)
    for m, batches, accumulation in (
        (a, [collate([first]), collate([second])], 2),
        (b, [collate([first, second])], 1),
    ):
        run_bridge_epoch(
            m,
            batches,
            "cpu",
            optimizer=torch.optim.SGD(m.parameters(), lr=0.01),
            accumulation_steps=accumulation,
            length_class_weights=[2, 1],
        )
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def write_midi(path, ppq=480, *, explicit_meter=True):
    import mido

    midi = mido.MidiFile(ticks_per_beat=ppq)
    meta = mido.MidiTrack()
    if explicit_meter:
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    meta.extend(
        [
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.MetaMessage("set_tempo", tempo=250000, time=ppq // 2),
            mido.MetaMessage("marker", text="phrase_end", time=ppq * 3 // 2),
        ]
    )
    notes = mido.MidiTrack()
    notes.extend(
        [
            mido.Message("note_on", note=60, velocity=80),
            mido.Message("note_off", note=60, time=ppq // 3),
            mido.Message("note_on", note=62, velocity=80),
            mido.Message("note_off", note=62, time=2 * ppq // 3),
            mido.Message("note_on", note=64, velocity=80),
            mido.Message("note_off", note=64, time=ppq),
        ]
    )
    midi.tracks.extend([meta, notes])
    midi.save(path)


def test_midi_preparation_tensor_parity_and_no_implicit_meter(tmp_path):
    path = tmp_path / "song.mid"
    write_midi(path)
    actual = midi_melody_record(path, canonical=True)
    encoded = encode_octuple(actual["lines"], actual["source"])
    expected = encode_octuple([phrase()], source())
    for key in ("melody_features", "timing_residuals"):
        np.testing.assert_array_equal(encoded[key], expected[key])
    np.testing.assert_allclose(
        actual["lines"][0]["melody"]["note_duration_seconds"],
        phrase()["melody"]["note_duration_seconds"],
    )
    records, meta = predict_midi_templates(model(predict_note_roles=True).eval(), path)
    assert meta["count_sources"] == ["predicted_syllable_starts"]
    assert len(records) == 1
    write_midi(path, explicit_meter=False)
    with pytest.raises(ValueError, match="Missing explicit meter"):
        midi_melody_record(path, canonical=True)


@pytest.fixture
def prepared_config(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    raw = tmp_path / "raw.jsonl.gz"
    songs, checksums, rows = {}, {}, []
    data = dict(
        source="lakh",
        lakh_file=str(raw),
        prepared_dir=str(base),
        include_melody=True,
        stress_source="ipa",
        prosody_mode="actual",
        max_syllables=40,
        seed=1234,
        valid_fraction=0.1,
        test_fraction=0.1,
        require_meter=True,
    )
    for i, split in enumerate(("train", "valid", "test")):
        md5 = f"{i:032x}"
        sid = "lakh:" + md5
        line = phrase(variant=i)
        line["melody"]["meter"] = canonical_meter([line], source())[0]
        song_source = source()
        song_source["meter_phase"] = "midi_signature_change_origin_v1"
        song = dict(song_id=sid, lines=[line], source=song_source)
        m = line["melody"]
        rows.append(
            dict(
                midi_md5=md5,
                notes=[
                    dict(pitch=p, onset_tick=t, duration_ticks=d, continuation=bool(role))
                    for p, t, d, role in zip(
                        m["midi_pitches"],
                        m["onset_ticks"],
                        m["note_duration_ticks"],
                        line["note_roles"],
                        strict=True,
                    )
                ],
            )
        )
        song["lines"][0].pop("note_roles")
        file = base / f"{split}.jsonl"
        file.write_text(json.dumps(song) + "\n")
        checksums[file.name] = file_sha256(file)
        songs[sid] = dict(group=sid, split=split)
    with gzip.open(raw, "wt") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in rows))
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        config=data,
        songs=songs,
        sha256=checksums,
        pronunciation=dict(rules=REPO_RULES),
        source=dict(sha256=file_sha256(raw)),
    )
    (base / "manifest.json").write_text(json.dumps(manifest))
    return dict(
        data={**data, "base_prepared_dir": str(base), "prepared_dir": str(tmp_path / "new")},
        model={
            **model(predict_note_roles=True).configuration(),
            "melody_encoder": dict(
                d_model=16, num_layers=1, num_heads=2, dim_feedforward=32, dropout=0, max_length=32
            ),
            "conditioning": "bridge",
            "melody_initialization": "random",
        },
        training=dict(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            seed=1234,
            epochs=1,
            batch_size=2,
            patience=1,
            learning_rate=1e-3,
            warmup_steps=0,
            schedule="constant_after_warmup",
        ),
    )


def test_canonical_migration_training_evaluation_and_split_preservation(prepared_config):
    cfg = prepared_config
    preflight = prepare_canonical_lakh(cfg, check_splits=True)
    assert preflight["split_counts"] == {"train": 1, "valid": 1, "test": 1}
    manifest = prepare_canonical_lakh(cfg)
    assert manifest["preparation_parent"]["split_changes"] == []
    assert manifest["note_role_counts"]["train_1"] == 1
    assert manifest["octuple_codec"] == codec_metadata()
    before = manifest["sha256"]
    assert prepare_canonical_lakh(cfg)["sha256"] == before
    ds = BridgeDataset(
        cfg["data"]["prepared_dir"], "valid", **model(predict_note_roles=True).dataset_options
    )
    assert ds[0]["note_roles"] == [0, 1, 0]
    output = train_bridge(cfg, smoke_test=True)
    report = evaluate_bridge(
        output / "best", cfg["data"]["prepared_dir"], split="valid", device="cpu"
    )
    assert report["note_roles"]["notes"] == 3
    assert report["free_prediction"]["slot_count_source"] == "predicted_syllable_starts"
    assert report["prediction"]["slot_count_source"] == "ipa"
    assert report["source_scheme"] == REPRESENTATION


def test_canonical_identity_ignores_phrase_boundaries_and_optional_meter():
    p = phrase()
    a = content_fingerprints([p], source())
    split = []
    for lo, hi in ((0, 1), (1, 3)):
        split.append({"melody": {k: v[lo:hi] for k, v in p["melody"].items()}})
    b = content_fingerprints(split, source())
    assert a["canonical_melody"] == b["canonical_melody"]
    assert a["octuple_input"] != b["octuple_input"]
