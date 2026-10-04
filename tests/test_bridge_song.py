"""Song coordinates, window-preserving transfer, global attention and cached decoding."""

import copy
import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from prosodia_lyricist.bridge_data import (
    BOS,
    PAD,
    SONG_FEATURE_DIM,
    SONG_SCALARS,
    BridgeDataset,
    bridge_collate,
    decode_bridge_tokens,
    encode_bridge_song,
    encode_bridge_source,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_decoding import DecoderCache
from prosodia_lyricist.bridge_eval import evaluate_bridge
from prosodia_lyricist.bridge_infer import predict_templates
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import evaluate_templates, run_bridge_epoch
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


@pytest.fixture
def song():
    lines, start = [], 0.0
    for index, n in enumerate((3, 2, 4, 1, 2)):
        lines.append({
            "syllables": [{"stress": "strong", "length": "short"}] * max(1, n - 1),
            "melody": {
                "midi_pitches": [60 + index + k for k in range(n)],
                "onset_seconds": [start + k * 0.5 for k in range(n)],
                "note_duration_seconds": [0.25 + k * 0.1 for k in range(n)],
            },
        })
        start += n * 0.5 + 0.75
    return lines


@pytest.fixture
def model():
    return ProsodyBridge(
        dict(d_model=16, num_layers=1, num_heads=2, dim_feedforward=32,
             dropout=0, max_length=8),
        bridge_scope="song", encoder_lines_per_window=2, max_window_notes=5,
        max_song_notes=64, max_song_lines=8, max_target_length=64, max_syllables=8,
        d_model=16, num_heads=2, num_layers=2, song_encoder_layers=2,
        dim_feedforward=32, dropout=0,
    )


def example(lines):
    return {**encode_bridge_song(lines), "labels": encode_bridge_targets(lines, 8)}


def test_global_features_preserve_local_inputs_and_cross_window_coordinates(song):
    encoded = encode_bridge_song(song)
    local = [encode_bridge_source(song[i:i + 2]) for i in range(0, len(song), 2)]
    np.testing.assert_array_equal(
        encoded["melody_features"], np.concatenate([w["melody_features"] for w in local])
    )
    assert encoded["line_count"] == 5
    assert encoded["note_line_ids"] == [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5]
    g = encoded["song_features"]
    assert g.shape == (12, SONG_FEATURE_DIM)
    # First note of the second window keeps its real predecessor and song pitch anchor.
    assert g[5, 0] == pytest.approx(2 / 12)
    assert g[5, 1] == 0
    assert g[5, 3] == pytest.approx(np.log2(1.25 / 0.5))
    assert g[5, 4:6].tolist() == [1, 1]
    assert g[0, 3:6].tolist() == [0, 0, 0]
    changed = copy.deepcopy(song)
    for line in changed:
        line.update(text="not a source feature", syllables=[], words=[])
        melody = line["melody"]
        melody["midi_pitches"] = [p + 7 for p in melody["midi_pitches"]]
        melody["onset_seconds"] = [t + 100 for t in melody["onset_seconds"]]
    np.testing.assert_allclose(encode_bridge_song(changed)["song_features"], g, atol=1e-6)
    for line in changed:
        melody = line["melody"]
        melody["onset_seconds"] = [t * 2 for t in melody["onset_seconds"]]
        melody["note_duration_seconds"] = [d * 2 for d in melody["note_duration_seconds"]]
    doubled = encode_bridge_song(changed)["song_features"]
    scale_column = SONG_SCALARS.index("log2_scale_seconds")
    np.testing.assert_allclose(doubled[:, scale_column], g[:, scale_column] + 1)
    np.testing.assert_allclose(
        np.delete(doubled, scale_column, axis=1), np.delete(g, scale_column, axis=1), atol=1e-6
    )


def test_duration_comparison_and_missing_or_zero_iois():
    lines = [{"melody": {
        "midi_pitches": [60, 62], "onset_seconds": [start, start + 0.5],
        "note_duration_seconds": duration,
    }} for start, duration in ((0, [0.25, 0.5]), (1, [0.5, 1.0]))]
    g = encode_bridge_song(lines, encoder_lines_per_window=1)["song_features"]
    np.testing.assert_allclose(g[:, 2], [-1, 0, 0, 1])
    # Degenerate timing remains finite and flags actual zero intervals separately from missing ones.
    lines[0]["melody"]["onset_seconds"] = [0, 0]
    g = encode_bridge_song(lines[:1])["song_features"]
    assert np.isfinite(g).all()
    assert g[:, 3:6].tolist() == [[0, 0, 0], [0, 1, 0]]
    one = {"melody": {key: value[:1] for key, value in lines[0]["melody"].items()}}
    assert np.isfinite(encode_bridge_song([one])["song_features"]).all()
    lines[1]["melody"]["onset_seconds"] = [-2, -1]
    with pytest.raises(ValueError, match="unordered"):
        encode_bridge_song(lines)


def test_window_packing_matches_independent_pretrained_encoding(song, model):
    model.eval()
    batch = bridge_collate([encode_bridge_song(song), encode_bridge_song(song[:3])])
    with torch.no_grad():
        packed = model.encode_windows(
            batch["melody_features"], batch["melody_attention_mask"].bool(),
            batch["note_line_ids"], batch["line_counts"],
        )
        for row, lines in enumerate((song, song[:3])):
            expected = []
            for i in range(0, len(lines), 2):
                window = bridge_collate([encode_bridge_source(lines[i:i + 2])])
                expected.append(model.melody_encoder.encode(
                    window["melody_features"], window["melody_attention_mask"], project=False
                ).note_embeddings[0])
            expected = torch.cat(expected)
            torch.testing.assert_close(packed[row, :len(expected)], expected, atol=1e-6, rtol=1e-5)


def test_song_gradients_freezing_global_context_and_padding(song, model):
    batch = bridge_collate([example(song), example(song[:1])])
    model.set_melody_trainable(False)
    model.train()
    model(**batch).loss.backward()
    assert all(p.grad is None for p in model.melody_encoder.parameters())
    for component in (model.adapter, model.song_features, model.song_encoder, model.decoder):
        assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0
    assert not model.melody_encoder.training
    model.zero_grad(set_to_none=True)
    model.set_melody_trainable(True)
    model(**batch).loss.backward()
    assert model.melody_encoder.input_projection[0].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        original = model(**batch)
        single = model(**bridge_collate([example(song)]))
        torch.testing.assert_close(original.logits[0], single.logits[0], atol=1e-6, rtol=1e-5)
        corrupt_padding = {key: value.clone() for key, value in batch.items()}
        for key in ("song_features", "melody_features"):
            corrupt_padding[key][~batch["melody_attention_mask"].bool()] = float("nan")
        torch.testing.assert_close(original.logits, model(**corrupt_padding).logits)
        future = {**batch, "labels": batch["labels"].clone()}
        future["labels"][0, 2] = 3 if future["labels"][0, 2] != 3 else 4
        torch.testing.assert_close(original.logits[:, :3], model(**future).logits[:, :3])
    source = bridge_collate([encode_bridge_song(song)])
    source["song_features"].requires_grad_()
    memory = model.encode(**source)
    gradient = torch.autograd.grad(memory[0, 0, 0], source["song_features"])[0]
    assert gradient[0, -1].abs().sum() > 1e-8  # Later windows can inform the very first note.


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("use_ipa_counts", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable"
    )),
])
def test_cached_logits_and_generation_match_full_decoder(
    song, model, precision, device, use_ipa_counts
):
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    model.to(device).eval()
    batch = bridge_collate([example(song), example(song[:1])])
    batch = {key: value.to(device) for key, value in batch.items()}
    source = {k: v for k, v in batch.items() if k not in ("labels", "syllable_counts")}
    context = torch.autocast(device, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    # Disabling the CPU fused encoder fast path also matches CUDA's autocast path.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.inference_mode(), context:
            memory = model.encode(**source)
            skeleton = model.make_skeleton(batch["syllable_counts"])
            tokens = torch.full_like(batch["labels"], PAD)
            tokens[:, 0] = BOS
            tokens[:, 1:] = batch["labels"][:, :-1].masked_fill(batch["labels"][:, :-1] < 0, PAD)
            full = model.decode(tokens, memory, source["melody_attention_mask"], skeleton)
            cache = DecoderCache(
                model.decoder, memory, source["melody_attention_mask"], tokens.size(1)
            )
            incremental = []
            for step in range(tokens.size(1)):
                current = {k: v[:, step:step + 1] for k, v in skeleton.items()}
                query = model.query_embeddings(tokens[:, step:step + 1], current, offset=step)
                incremental.append(model.classify(cache.step(
                    query, current["tokens"][:, 0].ne(PAD)
                )).logits)
            tolerance = 0.04 if precision == "bf16" else 1e-6
            torch.testing.assert_close(
                torch.cat(incremental, dim=1), full.logits, atol=tolerance, rtol=tolerance
            )
            generation_source = dict(source)
            if use_ipa_counts:
                generation_source["syllable_counts"] = batch["syllable_counts"]
            cached = model.generate(**generation_source, use_cache=True)
            ordinary = model.generate(**generation_source, use_cache=False)
            if precision == "fp32":
                assert torch.equal(cached.sequences, ordinary.sequences)
            assert cached.syllable_counts[0].tolist() == (
                [2, 1, 3, 1, 1] if use_ipa_counts else [3, 2, 4, 1, 2]
            )
            assert len(decode_bridge_tokens(cached.sequences[1].tolist())) == 1
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


@pytest.mark.parametrize("cudnn_enabled", [False, True])
@pytest.mark.parametrize("attention_fails", [False, True])
def test_cached_attention_backend_is_scoped_and_restored(
    model, monkeypatch, cudnn_enabled, attention_fails,
):
    from prosodia_lyricist import bridge_decoding

    model.eval()
    memory = torch.randn(2, 3, model.decoder_config["d_model"])
    query = memory[:, :1]
    keep = torch.tensor([[True, True, True], [True, False, False]])
    attention = bridge_decoding.F.scaled_dot_product_attention
    calls = []

    def checked_attention(*args, **kwargs):
        calls.append(True)
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        if attention_fails:
            raise RuntimeError("attention failed")
        return attention(*args, **kwargs)

    monkeypatch.setattr(bridge_decoding.F, "scaled_dot_product_attention", checked_attention)
    backends = [SDPBackend.MATH] + ([SDPBackend.CUDNN_ATTENTION] if cudnn_enabled else [])
    with torch.inference_mode(), sdpa_kernel(backends):
        cache = DecoderCache(model.decoder, memory, keep, max_length=2)
        if attention_fails:
            with pytest.raises(RuntimeError, match="attention failed"):
                cache.step(query, torch.ones(2, dtype=torch.bool))
        else:
            assert torch.isfinite(cache.step(query, torch.ones(2, dtype=torch.bool))).all()
        assert calls
        assert torch.backends.cuda.cudnn_sdp_enabled() == cudnn_enabled
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_song_checkpoint_roundtrip_and_feature_validation(tmp_path, song, model):
    model.eval().save(tmp_path)
    restored = ProsodyBridge.load(tmp_path).eval()
    assert restored.bridge_scope == "song" and restored.song_encoder_layers == 2
    assert restored.dataset_options == model.dataset_options
    batch = bridge_collate([example(song)])
    torch.testing.assert_close(restored(**batch).logits, model(**batch).logits, rtol=0, atol=0)
    source = bridge_collate([encode_bridge_song(song)])
    assert torch.equal(restored.generate(**source).sequences, model.generate(**source).sequences)
    path = tmp_path / "bridge.json"
    meta = json.loads(path.read_text())
    assert meta["format_version"] == 5 and meta["target_scheme"] == "line_skeleton_v5"
    meta["song_feature_scheme"] = "different_normalization"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature rules"):
        ProsodyBridge.load(tmp_path)
    with pytest.raises(ValueError, match="global melody features"):
        model.generate(**{k: v for k, v in source.items() if k != "song_features"})


def test_song_accumulation_and_limit_checks(song, model):
    joint = copy.deepcopy(model)
    separate = [bridge_collate([example(song)]), bridge_collate([example(song[:1])])]
    for net, batches, accumulation in (
        (model, separate, 3), (joint, [bridge_collate([example(song), example(song[:1])])], 1),
    ):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        run_bridge_epoch(
            net, batches, torch.device("cpu"), optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            accumulation_steps=accumulation, gradient_clip=None,
        )
    for a, b in zip(model.parameters(), joint.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    source = bridge_collate([encode_bridge_song(song)])
    model.max_notes = 4
    with pytest.raises(ValueError, match="window.*note limit"):
        model.eval().generate(**source)
    model.max_notes = 5
    model.max_target_length = 10
    scores = evaluate_templates(model, [bridge_collate([example(song), example(song[:1])])], "cpu")
    assert scores["skipped_target_limit_examples"] == 1 and scores["phrases"] == 1
    assert scores["skipped_target_limit_phrases"] == 5
    assert scores["skipped_slot_limit_phrases"] == scores["skipped_slot_limit_songs"] == 0
    model.max_target_length = 64
    model.max_syllables = 3
    scores = evaluate_templates(model, [bridge_collate([example(song), example(song[:1])])], "cpu")
    assert scores["skipped_slot_limit_songs"] == 0 and scores["phrases"] == 6
    model.max_syllables = 2
    scores = evaluate_templates(model, [bridge_collate([example(song), example(song[:1])])], "cpu")
    assert scores["skipped_slot_limit_songs"] == 1 and scores["phrases"] == 1


def test_song_dataset_evaluation_and_whole_example_skips(tmp_path, song, model):
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps({"song_id": name, "lines": lines}) + "\n"
                            for name, lines in (("full", song), ("short", song[:1]))))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "config": {"include_melody": True, "stress_source": "ipa", "max_syllables": 8},
        "songs": {name: {"split": "test", "group": name} for name in ("full", "short")},
        "sha256": {"test.jsonl": file_sha256(path)},
    }))
    dataset = BridgeDataset(tmp_path, "test", **model.dataset_options)
    assert len(dataset) == 2 and dataset.encoder_windows == 4
    assert dataset[0]["line_count"] == 5
    for override, reason in (
        ({"max_notes": 4}, "window_notes"), ({"max_song_notes": 11}, "song_notes"),
        ({"max_song_lines": 4}, "song_lines"), ({"max_target_length": 6}, "target_tokens"),
    ):
        limited = BridgeDataset(tmp_path, "test", **{**model.dataset_options, **override})
        assert len(limited) == 1 and limited.skipped == 1
        assert limited.examples[0]["line_count"] == 1
        assert limited.skipped_limits[reason] == 1
    model.save(tmp_path / "checkpoint")
    report = evaluate_bridge(tmp_path / "checkpoint", tmp_path, device="cpu", batch_size=2)
    assert report["data"]["examples"] == report["data"]["songs"] == 2
    assert report["data"]["encoder_windows"] == 4
    assert "windows" not in report["data"]
    assert report["generation"]["phrases"] == 6
    assert report["bridge_scope"] == "song"


def test_midi_song_generation_includes_window_tail(tmp_path, model):
    midi = pytest.importorskip("miditoolkit")
    source = midi.MidiFile(ticks_per_beat=480)
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60 + i, i * 960, i * 960 + 240) for i in range(5)]
    source.instruments.append(instrument)
    source.markers = [midi.Marker(f"end {i}", i * 960 + 480) for i in range(4)]
    path = tmp_path / "song.mid"
    source.dump(str(path))
    model.save(tmp_path / "checkpoint")
    lines, metadata = predict_templates(tmp_path / "checkpoint", path)
    assert len(lines) == 5 and lines[-1]["unmarked_tail"]
    assert metadata["bridge_scope"] == "song" and metadata["encoder_windows"] == 3
    assert metadata["note_counts"] == metadata["syllable_counts"] == [1] * 5
    assert [line["line_id"] for line in lines] == list(range(5))
