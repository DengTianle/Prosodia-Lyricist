"""Label-history corruption must preserve supervision, structure, and evaluation."""

import pytest
import torch
import torch.nn.functional as F

from prosodia_lyricist.bridge_data import (
    FIRST_LINE,
    PAD,
    PAIR_OFFSET,
    SLOT,
    bridge_collate,
    encode_bridge_song,
    encode_bridge_source,
    encode_bridge_targets,
)
from prosodia_lyricist.bridge_model import ProsodyBridge
from prosodia_lyricist.bridge_train import run_bridge_epoch, train_bridge


@pytest.fixture(params=["window", "song"])
def bridge_batch(request):
    torch.manual_seed(1234)
    model = ProsodyBridge(
        dict(d_model=8, num_layers=1, num_heads=2, dim_feedforward=16,
             dropout=0, max_length=16),
        bridge_scope=request.param, lines_per_window=2, max_syllables=8,
        max_song_notes=32, max_song_lines=4, max_target_length=32,
        d_model=8, num_layers=2, num_heads=2, dim_feedforward=16,
        dropout=0, song_encoder_layers=1,
    )
    pairs = [("strong", "long"), ("weak", "short"), ("strong", "short"), ("weak", "long")]
    lines = [{
        "syllables": [{"stress": s, "length": v} for s, v in pairs[:count]],
        "melody": {
            "midi_pitches": list(range(60, 60 + count)),
            "onset_seconds": list(range(index * 10, index * 10 + count)),
            "note_duration_seconds": [0.5] * count,
        },
    } for index, count in enumerate((4, 3))]
    encode = encode_bridge_song if request.param == "song" else encode_bridge_source
    batch = bridge_collate([
        {**encode(group), "labels": encode_bridge_targets(group, 8)}
        for group in (lines, lines[:1])
    ])
    return model, batch


def capture_inputs(monkeypatch, model):
    calls, original = [], model.decode

    def decode(tokens, memory, source_mask, skeleton):
        calls.append((tokens.detach().clone(), {k: v.clone() for k, v in skeleton.items()}))
        return original(tokens, memory, source_mask, skeleton)

    monkeypatch.setattr(model, "decode", decode)
    return calls


def test_full_mask_preserves_gold_targets_structure_and_gradients(bridge_batch, monkeypatch):
    model, batch = bridge_batch
    original = {key: value.clone() for key, value in batch.items()}
    calls = capture_inputs(monkeypatch, model)
    model.train()
    model(**batch)
    output = model(**batch, history_mask_probability=1.0)
    plain, skeleton = calls[0]
    masked, masked_skeleton = calls[1]
    eligible = plain.ge(PAIR_OFFSET) & plain.lt(FIRST_LINE) & skeleton["tokens"].ne(PAD)
    assert masked[eligible].eq(SLOT).all()
    assert torch.equal(masked[~eligible], plain[~eligible])
    assert masked[0, 5] == SLOT  # Last label of line 0 is feedback at the next line prefix.
    for key in skeleton:
        assert torch.equal(skeleton[key], masked_skeleton[key])
    for key in batch:
        assert torch.equal(batch[key], original[key])
    slots = skeleton["tokens"].eq(SLOT)
    gold = batch["labels"][slots] - PAIR_OFFSET
    expected_loss = (
        F.cross_entropy(output.strength_logits[slots], gold // 2)
        + F.cross_entropy(output.length_logits[slots], gold % 2)
    )
    torch.testing.assert_close(output.loss, expected_loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad[SLOT].abs().sum() > 0
    assert model.melody_encoder.input_projection[0].weight.grad.abs().sum() > 0
    assert model.strength_head.weight.grad.abs().sum() > 0
    assert model.length_head.weight.grad.abs().sum() > 0


def test_disabled_mask_and_eval_preserve_outputs_and_rng(bridge_batch):
    model, batch = bridge_batch
    for training in (True, False):
        model.train(training)
        rng = torch.random.get_rng_state()
        baseline = model(**batch)
        explicit = model(**batch, history_mask_probability=0.0 if training else 0.8)
        assert torch.equal(baseline.logits, explicit.logits)
        assert torch.equal(baseline.loss, explicit.loss)
        assert torch.equal(torch.random.get_rng_state(), rng)


def test_partial_mask_is_seeded_and_leaves_some_history(bridge_batch, monkeypatch):
    model, batch = bridge_batch
    calls = capture_inputs(monkeypatch, model)
    model.train()
    for seed in (17, 17, 18):
        torch.manual_seed(seed)
        model(**batch, history_mask_probability=0.5)
    assert torch.equal(calls[0][0], calls[1][0])
    assert not torch.equal(calls[0][0], calls[2][0])
    tokens, skeleton = calls[0]
    active = skeleton["tokens"].ne(PAD)
    assert tokens[active].eq(SLOT).any()
    assert (tokens[active].ge(PAIR_OFFSET) & tokens[active].lt(FIRST_LINE)).any()


def test_epoch_applies_mask_only_to_training(bridge_batch, monkeypatch):
    model, batch = bridge_batch
    calls = capture_inputs(monkeypatch, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    training = run_bridge_epoch(
        model, [batch], "cpu", optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        history_mask_probability=1.0,
    )
    validation = run_bridge_epoch(model, [batch], "cpu", history_mask_probability=1.0)
    assert calls[0][0].eq(SLOT).any()
    assert not calls[1][0].eq(SLOT).any()
    assert training["history_mask_probability"] == 1.0
    assert validation["history_mask_probability"] == 0.0
    assert training["tokens"] == validation["tokens"] == 11


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf"), "bad", None])
def test_invalid_probability_fails_in_model_and_before_loading_data(bridge_batch, probability):
    model, batch = bridge_batch
    with pytest.raises(ValueError, match="history_mask_probability"):
        model(**batch, history_mask_probability=probability)
    with pytest.raises(ValueError, match="history_mask_probability"):
        train_bridge({
            "data": {}, "model": {},
            "training": dict(batch_size=1, learning_rate=1e-4, melody_learning_rate=2e-6,
                             history_mask_probability=probability),
        })
