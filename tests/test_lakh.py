"""Calibrated Lakh preparation, shared actual IPA alignment, and metered training."""

import copy
import gzip
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prosodia_lyricist import ipa, lakh
from prosodia_lyricist.bridge_alignment import (
    ALIGNMENT_SCHEME,
    LAKH_ALIGNMENT_SCHEME,
    aligned_note_indices,
)
from prosodia_lyricist.bridge_data import BridgeDataset, encode_bridge_song
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import train_bridge
from prosodia_lyricist.config import load_config
from prosodia_lyricist.dali import AnnotationError
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import prepare


@pytest.fixture
def pronunciation(monkeypatch):
    # Exercise the real IPA rules with a deterministic dictionary/tokenizer.
    lexicon = {"hello": "'hɛ.loʊ", "you": "juː", "light": "'laɪt", "heaven": "'hɛ.vən"}
    monkeypatch.setattr(ipa, "_historical_lexicon", lambda: (lexicon, set(), {"you"}))
    monkeypatch.setattr(
        ipa,
        "backend",
        lambda: SimpleNamespace(
            Text=lambda text, **kw: SimpleNamespace(
                wordtokens=[SimpleNamespace(txt=t) for t in re.findall(r"\w+|[^\w\s]", text)]
            )
        ),
    )
    monkeypatch.setattr(lakh, "backend", lambda: None)
    monkeypatch.setattr(lakh, "historical_provenance", lambda: {"version": "1.5.0"})


def source_record(index=0):
    notes = []
    for i, (text, word, line, syllable, cont) in enumerate(
        [
            ("hel", 0, 0, 0, False),
            ("", 0, 0, 0, True),
            ("lo", 0, 0, 1, False),
            ("you", 1, 0, 2, False),
            ("light", 2, 1, 3, False),
        ]
    ):
        notes.append(
            dict(
                pitch=60 + i,
                onset_tick=i * 480,
                offset_tick=(i + 1) * 480,
                duration_ticks=480,
                onset_seconds=i / 4,
                offset_seconds=(i + 1) / 4,
                duration_seconds=0.25,
                onset_quarter=i / 2,
                duration_quarters=0.5,
                syllable_text=text,
                word_id=word,
                line_id=line,
                paragraph_id=0,
                syllable_id=syllable,
                continuation=cont,
                phonemes="Y UW1",
                stress=1,
            )
        )
    md5 = f"{index:032x}"
    return dict(
        source_file=f"A/B/C/TRABC/{md5}.mid",
        midi_md5=md5,
        msd_id=f"TRABC{index}",
        ticks_per_quarter=960,
        original_ticks_per_quarter=480,
        seconds_status="explicit_track_zero_tempo",
        timing_verification="verified",
        orthographic_alignment="count_consistent_index_mapping",
        notes=notes,
        deduplication=dict(is_representative=True, source_aliases=[f"alias/{md5}.mid"]),
        tempo_map=[dict(tempo=500000, original_tick=0, tick_960=0, quarter=0, track=0)],
        time_signatures=[
            dict(numerator=4, denominator=4, original_tick=0, tick_960=0, quarter=0, track=0),
            dict(
                numerator=6, denominator=8, original_tick=720, tick_960=1440, quarter=1.5, track=0
            ),
        ],
    )


def write_source(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def configuration(tmp_path, records):
    path = tmp_path / "source.jsonl.gz"
    write_source(path, records)
    return dict(
        data=dict(
            source="lakh",
            lakh_file=str(path),
            prepared_dir=str(tmp_path / "prepared"),
            stress_source="ipa",
            prosody_mode="actual",
            include_melody=True,
            require_meter=True,
            max_syllables=8,
            seed=1234,
            valid_fraction=0.2,
            test_fraction=0.2,
        ),
        model=dict(
            conditioning="bridge",
            melody_initialization="random",
            meter_conditioning=True,
            melody_encoder=dict(
                d_model=8, num_layers=1, num_heads=2, dim_feedforward=16, dropout=0, max_length=16
            ),
            max_song_notes=16,
            max_song_lines=4,
            max_target_length=32,
        ),
        training=dict(
            output_dir=str(tmp_path / "runs"),
            device="cpu",
            precision="fp32",
            seed=1234,
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


def test_processed_notes_actual_targets_and_melisma_anchors(pronunciation):
    raw = source_record()
    before = copy.deepcopy(raw)
    song = lakh.extract_song(raw)
    assert raw == before
    first = song["lines"][0]
    assert first["text"] == "hello you"
    assert first["melody"]["midi_pitches"] == [60, 61, 62, 63]
    assert first["melody"]["source_note_indices"] == [0, 1, 2, 3]
    assert first["sung_syllables"][0]["note_count"] == 2
    assert first["sung_syllables"][0]["duration"] == 0.5
    assert [(s["stress"], s["length"]) for s in first["syllables"]] == [
        ("strong", "short"),
        ("weak", "short"),
        ("weak", "long"),
    ]
    # IPA diphthongs stay short, independently of source ARPAbet/durations.
    assert song["lines"][1]["syllables"][0]["length"] == "short"
    assert first["note_alignment"]["scheme"] == LAKH_ALIGNMENT_SCHEME
    assert aligned_note_indices(song["lines"])[0] == [-1, 0, 2, 3, -1, 4]
    meter = first["melody"]["meter"]
    assert [m["beat_position"] for m in meter] == [0, 0.5, 1, 0]
    assert meter[3] == dict(numerator=6, denominator=8, beat_position=0)
    assert song["lines"][1]["melody"]["meter"][0]["beat_position"] == 1
    # Targets, spelling, and source phonemes never enter the note/meter features.
    encoded = encode_bridge_song(song["lines"])
    altered = copy.deepcopy(song["lines"])
    for line in altered:
        for key in ("syllables", "words", "sung_syllables", "note_alignment"):
            line.pop(key)
        line["text"] = "different"
    for key in ("melody_features", "meter_features", "meter_mask"):
        np.testing.assert_array_equal(encoded[key], encode_bridge_song(altered)[key])


def test_strict_mismatch_is_word_local_and_following_word_still_aligns(pronunciation):
    raw = source_record()
    raw["notes"][0]["syllable_text"] = "hea"
    raw["notes"][2]["syllable_text"] = "ven"
    raw["notes"][2]["continuation"] = True
    raw["notes"][0]["syllable_text"] = "heaven"
    song = lakh.extract_song(raw)
    strict, counts = aligned_note_indices(song["lines"])
    assert strict == [-1, -1, -1, 3, -1, 4]
    assert counts["mismatched_words"] == 1
    assert aligned_note_indices(song["lines"], "prefix")[0] == [-1, 0, -1, 3, -1, 4]


@pytest.mark.parametrize(
    "problem,match",
    [
        ("seconds", "calibrated seconds"),
        ("tempo", "Unresolved tempo"),
        ("meter", "Missing explicit meter"),
        ("meter_conflict", "Conflicting"),
        ("overlap", "overlapping"),
        ("orphan", "Melisma continuation"),
        ("spelling", "Spelling alignment"),
    ],
)
def test_bad_sources_are_rejected_without_repairs(pronunciation, problem, match):
    raw = source_record()
    if problem == "seconds":
        raw["notes"][0]["onset_seconds"] = None
    elif problem == "tempo":
        raw["seconds_status"] = "unresolved_tempo_metadata"
    elif problem == "meter":
        raw["time_signatures"] = []
    elif problem == "meter_conflict":
        raw["time_signatures"].append({**raw["time_signatures"][0], "numerator": 3})
    elif problem == "overlap":
        raw["notes"][1]["onset_seconds"] = 0
    elif problem == "orphan":
        raw["notes"][0].update(continuation=True, syllable_text="")
    elif problem == "spelling":
        raw["orthographic_alignment"] = "unresolved_note_lyric_count_mismatch"
    with pytest.raises(AnnotationError, match=match):
        lakh.extract_song(raw)


def test_unknown_meter_prefix_is_masked_and_later_signature_observed(pronunciation):
    raw = source_record()
    raw["time_signatures"] = raw["time_signatures"][1:]
    song = lakh.extract_song(raw, require_meter=False)
    assert song["lines"][0]["melody"]["meter"][:3] == [None] * 3
    encoded = encode_bridge_song(song["lines"])
    assert encoded["meter_mask"].tolist() == [False, False, False, True, True]


def test_preparation_and_metered_training_roundtrip(tmp_path, pronunciation):
    records = [source_record(i) for i in range(40)]
    for index, record in enumerate(records):
        record["notes"][1]["pitch"] = 61 + index
    bad = source_record(100)
    bad["notes"][0]["onset_seconds"] = None
    cfg = configuration(tmp_path, records + [bad])
    source_hash = file_sha256(cfg["data"]["lakh_file"])
    manifest = prepare(cfg)
    directory = Path(cfg["data"]["prepared_dir"])
    assert manifest["pronunciation"]["rules"] == ipa.REPO_RULES
    assert manifest["source"]["sha256"] == source_hash
    assert manifest["counts"]["songs_rejected"] == 1
    assert manifest["counts"]["notes_with_meter"] == 200
    assert len(manifest["songs"]) == 40
    hashes = manifest["sha256"]
    assert prepare(cfg)["sha256"] == hashes
    assert file_sha256(cfg["data"]["lakh_file"]) == source_hash
    for split in ("train", "valid", "test"):
        dataset = BridgeDataset(directory, split)
        assert dataset.examples[0]["meter_mask"].all()
    # Relocating the input archive does not invalidate materialized prepared data.
    cfg["data"]["lakh_file"] = str(tmp_path / "relocated.jsonl.gz")
    output = train_bridge(cfg, smoke_test=True)
    model = ProsodyBridge.load(output / "best")
    assert model.meter_conditioning and model.strength_labels == ipa.REPO_STRESSES
    assert model.meter_adapter[-1].weight.abs().sum() > 0
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert metrics["train"]["optimizer_steps"] > 0 and metrics["valid"]["loss"] > 0
    # New generic alignment metadata still accepts existing DALI checkpoints.
    metadata_path = output / "best" / "bridge.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["alignment_scheme"] = ALIGNMENT_SCHEME
    metadata_path.write_text(json.dumps(metadata))
    assert ProsodyBridge.load(output / "best").meter_conditioning


def test_split_preflight_groups_full_lyrics_before_ipa(tmp_path, monkeypatch):
    records = [source_record(1), source_record(2)]
    for record in records:
        record["notes"] *= 20  # The preflight reads identities without note extraction.
    cfg = configuration(tmp_path, records)
    monkeypatch.setattr(lakh, "backend", lambda: pytest.fail("preflight must not run IPA"))
    plan = prepare(cfg, check_splits=True)
    assert len({s["group"] for s in plan["songs"].values()}) == 1
    assert len({s["split"] for s in plan["songs"].values()}) == 1
    assert not (Path(cfg["data"]["prepared_dir"]) / "train.jsonl").exists()


def test_duplicate_hashes_and_nonactual_mode_fail_before_ipa(tmp_path, monkeypatch):
    cfg = configuration(tmp_path, [source_record(), source_record()])
    monkeypatch.setattr(lakh, "backend", lambda: pytest.fail("must reject before IPA"))
    with pytest.raises(ValueError, match="Duplicate MIDI hash"):
        prepare(cfg)
    cfg["data"]["prosody_mode"] = "paper"
    with pytest.raises(ValueError, match="actual"):
        prepare(cfg)


def test_config_resolves_lakh_archive_and_prepared_paths(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  lakh_file: ./source.jsonl.gz\n  prepared_dir: ./prepared\nmodel: {}\n"
    )
    cfg = load_config(path)
    assert cfg["data"]["lakh_file"] == str(tmp_path / "source.jsonl.gz")
    assert cfg["data"]["prepared_dir"] == str(tmp_path / "prepared")


@pytest.mark.parametrize("setting,value", [("max_song_lines", 0), ("alignment_policy", "invalid")])
def test_invalid_model_settings_fail_before_processing(tmp_path, pronunciation, setting, value):
    cfg = configuration(tmp_path, [source_record()])
    cfg["model"][setting] = value
    with pytest.raises(ValueError, match="limits|alignment_policy"):
        prepare(cfg)
