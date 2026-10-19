"""Paper losses, causal four-stream decoding, and IPA feedback regressions."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from prosodia_lyricist.data import ProsodyCollator
from prosodia_lyricist.features import TARGET_KEYS, WORD_END, encode_example, encode_source
from prosodia_lyricist.model import LOSS_NAMES, ProsodyBart


@pytest.fixture
def song(record):
    record = copy.deepcopy(record)
    record["words"] = [
        {"text": "hello", "syllable_count": 2},
        {"text": "world", "syllable_count": 1},
    ]
    return {
        "title": "a song",
        "lines": [
            record,
            {
                "text": "light",
                "words": [{"text": "light", "syllable_count": 1}],
                "syllables": [{"stress": "weak", "length": "short"}],
            },
        ],
    }


@pytest.fixture
def model(tiny_model):
    return ProsodyBart(tiny_model.bart, max_syllables=8, dropout=0)


def test_paper_source_and_targets(tokenizer, song):
    example = encode_example(song, tokenizer, 8)
    assert tokenizer.convert_ids_to_tokens(example["input_ids"]) == [
        "<s>",
        "<title>",
        "a",
        "song",
        "<sent_0>",
        "<keywords>",
        "<prosody>",
        "<weak>",
        "<strong>",
        "<strong>",
        "<sent_1>",
        "<keywords>",
        "<prosody>",
        "<weak>",
        "</s>",
    ]
    assert tokenizer.convert_ids_to_tokens(example["labels"]) == [
        "<s>",
        "hello",
        WORD_END,
        "world",
        WORD_END,
        ".",
        "light",
        WORD_END,
        ".",
        "</s>",
    ]
    assert example["syllable_labels"] == [0, 0, 2, 0, 1, 0, 0, 1, 0, 0]
    assert example["stress_labels"] == [0, 0, 1, 0, 1, 0, 0, 2, 0, 0]
    assert example["length_labels"] == [0, 0, 1, 0, 2, 0, 0, 2, 0, 0]
    secret = copy.deepcopy(song)
    for line in secret["lines"]:
        line["text"] = "secret"
        line["words"] = []
    assert encode_source(song, tokenizer, 8) == encode_source(secret, tokenizer, 8)


def test_four_losses_shift_padding_gradients_and_reload(tmp_path, tokenizer, song, model):
    long = encode_example(song, tokenizer, 8)
    short = encode_example({**song, "lines": song["lines"][:1]}, tokenizer, 8)
    batch = ProsodyCollator(tokenizer.pad_token_id)([long, short])
    model.eval()
    seen = []
    original = model.embed_target

    def capture(*args):
        seen.append([arg.detach().clone() for arg in args])
        return original(*args)

    model.embed_target = capture
    output = model(**batch)
    assert tuple(output.loss_components) == LOSS_NAMES
    assert torch.allclose(output.loss, sum(output.loss_components.values()))
    expected = model.bart.prepare_decoder_input_ids_from_labels(batch["labels"])
    assert torch.equal(seen[0][0], expected)
    for i, key in enumerate(TARGET_KEYS[1:], 1):
        assert torch.equal(seen[0][i], model.shift_features(batch[key]))
        assert seen[0][i][:, 0].eq(0).all()
    # Loss padding must be ignored in every stream, including the non-word pad classes.
    padded = {
        k: torch.nn.functional.pad(
            v,
            (0, 3),
            value=-100 if k in TARGET_KEYS else (tokenizer.pad_token_id if k == "input_ids" else 0),
        )
        for k, v in batch.items()
    }
    padded_output = model(**padded)
    for name, value in output.loss_components.items():
        assert torch.allclose(value, padded_output.loss_components[name], atol=1e-6)
    output.loss.backward()
    for module in [
        *model.prosody_heads.values(),
        model.decoder_projection,
        model.projection,
        model.syllable_embedding,
        model.stress_embedding,
        model.length_embedding,
    ]:
        assert module.weight.grad.abs().sum() > 0
    model.save(tmp_path, tokenizer)
    restored = ProsodyBart.load(tmp_path).eval()
    assert restored.loss_weights == dict.fromkeys(LOSS_NAMES, 1)
    assert torch.equal(output.logits, restored(**batch).logits)
    for name, values in output.prosody_logits.items():
        assert torch.equal(values, restored(**batch).prosody_logits[name])
    broken = dict(batch, length_labels=batch["length_labels"].clone())
    broken["length_labels"][0, 0] = -100
    with pytest.raises(ValueError, match="aligned padding"):
        model(**broken)


def test_future_prosody_does_not_leak_into_current_prediction(tokenizer, song, model):
    model.eval()
    batch = ProsodyCollator(tokenizer.pad_token_id)([encode_example(song, tokenizer, 8)])
    before = model(**batch).logits
    # Labels at event 2 affect input event 3 and later only.
    batch["syllable_labels"][0, 2] = 7
    batch["stress_labels"][0, 2] = 2
    batch["length_labels"][0, 2] = 2
    after = model(**batch).logits
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3:], after[:, 3:])


def scripted_generation(monkeypatch, model, tokenizer, sequence):
    calls = []
    original = model.embed_target

    def embed(*args):
        calls.append(tuple(int(arg[0, 0]) for arg in args))
        return original(*args)

    monkeypatch.setattr(model, "embed_target", embed)
    sequence = iter(sequence)

    def forward(**kwargs):
        values = torch.full((1, 1, len(tokenizer)), -1000.0)
        values[0, 0, next(sequence)] = 1000.0
        return SimpleNamespace(
            logits=values,
            decoder_hidden_states=[torch.zeros(1, 1, model.bart.config.d_model)],
            past_key_values=None,
        )

    monkeypatch.setattr(model.bart, "forward", forward)
    for name, head in model.prosody_heads.items():
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(-1000)
            head.bias[1 if name == "syllables" else 2] = 1000
    return calls


@pytest.mark.parametrize("correct", [True, False])
def test_correction_is_fed_back_before_next_word(monkeypatch, model, tokenizer, song, correct):
    end = tokenizer.convert_tokens_to_ids(WORD_END)
    hello, world, period = [tokenizer.convert_tokens_to_ids(w) for w in ("hello", "world", ".")]
    sequence = [tokenizer.bos_token_id, hello, end, world, end, period, tokenizer.eos_token_id]
    calls = scripted_generation(monkeypatch, model, tokenizer, sequence)
    parsed = []

    def pronounce(word):
        parsed.append(word)
        return [{"text": word, "syllables": song["lines"][0]["syllables"][:2]}]

    inputs = {k: torch.tensor([v]) for k, v in encode_source(song, tokenizer, 8).items()}
    result = model.eval().generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=7,
        pronunciation=pronounce,
        prosody_correction=correct,
    )
    # This event is embedded to predict world, after correction of hello.
    assert calls[3] == (end, 2, 1, 1) if correct else calls[3] == (end, 1, 2, 2)
    assert parsed == (["hello", "world"] if correct else [])
    assert result.explanations[0]["completed"]
    assert result.explanations[0]["lines"] == ["hello world"]
    assert result.explanations[0]["words"][0]["predicted"] == {
        "syllables": 1,
        "stress": 2,
        "length": 2,
    }
    assert result.explanations[0]["words"][0]["correction_applied"] is correct
    # Punctuation/EOS skip sampling and carry pad in all three prosody streams.
    assert result.syllable_ids[0, -2:].tolist() == [0, 0]
    assert result.stress_ids.shape == result.length_ids.shape == result.sequences.shape


def test_unpronounceable_word_does_not_silently_claim_correction(
    monkeypatch, model, tokenizer, song
):
    sequence = [
        tokenizer.bos_token_id,
        tokenizer.convert_tokens_to_ids("hello"),
        tokenizer.convert_tokens_to_ids(WORD_END),
    ]
    scripted_generation(monkeypatch, model, tokenizer, sequence)

    def failed(word):
        raise ValueError("No IPA pronunciation")

    inputs = {k: torch.tensor([v]) for k, v in encode_source(song, tokenizer, 8).items()}
    with pytest.raises(ValueError, match="No IPA pronunciation"):
        model.eval().generate(**inputs, tokenizer=tokenizer, max_new_tokens=3, pronunciation=failed)


def test_generation_actual_cache_and_batch(tokenizer, song, model):
    inputs = {k: torch.tensor([v, v]) for k, v in encode_source(song, tokenizer, 8).items()}
    result = model.eval().generate(
        **inputs,
        tokenizer=tokenizer,
        max_new_tokens=6,
        prosody_correction=False,
    )
    assert result.sequences.shape[0] == 2
    assert result.sequences.shape[1] <= 7
    assert len(result.explanations) == 2
    assert result.attention_mask.shape == result.sequences.shape


@pytest.mark.parametrize("override", [None, 0])
def test_checkpoint_trigram_filter_preserves_word_events(
    monkeypatch, model, tokenizer, song, override
):
    from prosodia_lyricist.generation import generate_templates

    model.bart.generation_config.no_repeat_ngram_size = 3
    hello = tokenizer.convert_tokens_to_ids("hello")
    world = tokenizer.convert_tokens_to_ids("world")
    end = tokenizer.convert_tokens_to_ids(WORD_END)
    step = 0

    def forward(**kwargs):
        nonlocal step
        logits = torch.full((1, 1, len(tokenizer)), -1000.0)
        logits[0, 0, hello], logits[0, 0, world] = 10, 9
        if step == 9:
            logits[0, 0, tokenizer.eos_token_id] = 1000
        step += 1
        return SimpleNamespace(
            logits=logits,
            decoder_hidden_states=[torch.zeros(1, 1, model.bart.config.d_model)],
            past_key_values=None,
        )

    monkeypatch.setattr(model.bart, "forward", forward)
    inputs = {k: torch.tensor([v]) for k, v in encode_source(song, tokenizer, 8).items()}
    result = generate_templates(
        model.eval(), tokenizer, **inputs, max_new_tokens=10,
        prosody_correction=False, no_repeat_ngram_size=override,
    )
    words = [hello, hello, hello, world if override is None else hello]
    expected = [model.bart.config.decoder_start_token_id, tokenizer.bos_token_id]
    for word in words:
        expected.extend((word, end))
    expected.append(tokenizer.eos_token_id)
    assert result.sequences[0].tolist() == expected
    events = result.sequences.eq(end)
    assert int(events.sum()) == 4
    for features in (result.syllable_ids, result.stress_ids, result.length_ids):
        assert features[events].gt(0).all()
        assert features[~events].eq(0).all()
    assert result.explanations[0]["repetition_controls"] == {
        "no_repeat_ngram_size": 3 if override is None else 0,
        "repetition_penalty": 1.0,
    }


def test_text_repetition_penalty_adjusts_positive_and_negative_logits():
    from prosodia_lyricist.generation import apply_text_repetition

    logits = torch.tensor([2.0, -2.0, 3.0])
    adjusted = apply_text_repetition(
        logits, [0, 1, 0], no_repeat_ngram_size=0, repetition_penalty=2.0
    )
    torch.testing.assert_close(adjusted, torch.tensor([1.0, -4.0, 3.0]))
    torch.testing.assert_close(logits, torch.tensor([2.0, -2.0, 3.0]))


@pytest.mark.parametrize("name,value", [
    ("no_repeat_ngram_size", -1), ("no_repeat_ngram_size", True),
    ("no_repeat_ngram_size", 1.5), ("repetition_penalty", 0),
    ("repetition_penalty", float("inf")), ("repetition_penalty", float("nan")),
])
def test_invalid_repetition_settings_are_rejected(model, name, value):
    from prosodia_lyricist.generation import resolve_repetition_settings

    with pytest.raises(ValueError, match=name):
        resolve_repetition_settings(model, **{name: value})


def test_ipa_diphthongs_secondary_stress_and_punctuation():
    from prosodia_lyricist.ipa import syllable_features

    assert syllable_features("ˌaɪ")["stress"] == "strong"
    for syllable in ("eɪ", "aɪ", "ɔɪ", "əʊ", "oʊ", "aʊ", "ɪə", "eə", "ʊə", "ɑː"):
        assert syllable_features(syllable)["length"] == "long"
    assert syllable_features("æ")["length"] == "short"


def test_explainable_training_and_inference(tmp_path, monkeypatch, annotation, tokenizer):
    from prosodia_lyricist import ipa
    from prosodia_lyricist.infer import infer
    from prosodia_lyricist.prepare import prepare
    from prosodia_lyricist.train import train

    midi = pytest.importorskip("miditoolkit")
    monkeypatch.setattr(ipa, "backend", lambda: None)
    monkeypatch.setattr(
        ipa,
        "parse_words",
        lambda text: [
            {"text": word, "syllables": [ipa.syllable_features("ˈeɪ")]} for word in text.split()
        ],
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(30):
        row = copy.deepcopy(annotation)
        row["info"].update(id=f"song-{i}", title=f"Song {i}")
        (raw / f"{i}.json").write_text(json.dumps(row))
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    config = {
        "data": {
            "dali_dir": str(raw),
            "prepared_dir": str(tmp_path / "data"),
            "language": "english",
            "min_ncc": 0,
            "max_syllables": 8,
            "valid_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 1234,
            "stress_source": "ipa",
        },
        "model": {
            "pretrained": str(tmp_path / "tokenizer"),
            "dropout": 0,
            "max_source_length": 64,
            "max_target_length": 64,
        },
        "training": {
            "output_dir": str(tmp_path / "runs"),
            "device": "cpu",
            "seed": 1234,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patience": 2,
            "warmup_ratio": 0,
            "gradient_clip": 1,
            "learning_rate": 0.001,
            "weight_decay": 0,
        },
    }
    prepare(config)
    output = train(config, smoke_test=True, local_files_only=True)
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert set(metrics["train"]["components"]) == set(LOSS_NAMES)
    assert metrics["valid"]["loss"] == pytest.approx(sum(metrics["valid"]["components"].values()))
    melody = midi.MidiFile()
    instrument = midi.Instrument(0)
    instrument.notes = [midi.Note(80, 60, 0, 480)]
    melody.instruments.append(instrument)
    melody.markers.append(midi.Marker("Phrase_0", 480))
    melody.dump(str(tmp_path / "test.mid"))
    result = infer(
        output / "best",
        tmp_path / "test.mid",
        device="cpu",
        top_k=1,
        max_new_tokens=6,
        return_explanations=True,
    )
    assert set(result["template"]) == {"lyrics", "syllables", "stresses", "lengths", "tokens"}
    assert result["prosody_correction"]


def test_byte_level_word_is_corrected_once_after_all_pieces(monkeypatch, model):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from prosodia_lyricist.features import configure_tokenizer

    vocabulary = ["<s>", "<pad>", "</s>", "<unk>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({token: i for i, token in enumerate(vocabulary)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend.decoder = decoders.ByteLevel()
    tokenizer = configure_tokenizer(
        PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
        ),
        8,
    )
    model.bart.resize_token_embeddings(len(tokenizer))
    pieces = tokenizer.encode(" hello", add_special_tokens=False)
    assert len(pieces) > 2
    end = tokenizer.convert_tokens_to_ids(WORD_END)
    # Deliberately omit the leading-space BPE on the second word: the explicit
    # boundary must still separate displayed words, matching their IPA labels.
    sequence = [tokenizer.bos_token_id, *pieces, end, *pieces[1:], end, tokenizer.eos_token_id]
    calls = scripted_generation(monkeypatch, model, tokenizer, sequence)
    parsed = []

    def pronounce(word):
        parsed.append(word)
        return [
            {
                "text": word,
                "syllables": [
                    {"stress": "weak", "length": "short"},
                    {"stress": "strong", "length": "long"},
                ],
            }
        ]

    result = model.eval().generate(
        input_ids=torch.tensor([[0, 2]]),
        length_ids=torch.tensor([[0, 0]]),
        attention_mask=torch.tensor([[1, 1]]),
        tokenizer=tokenizer,
        max_new_tokens=len(sequence),
        pronunciation=pronounce,
    )
    assert parsed == ["hello", "hello"]
    assert result.explanations[0]["lines"] == ["hello hello"]
    assert all(event[1:] == (0, 0, 0) for event in calls if event[0] != end)
    assert calls[-1] == (end, 2, 1, 1)
    record = {
        "text": "hello",
        "words": [{"text": "hello", "syllable_count": 2}],
        "syllables": pronounce("hello")[0]["syllables"],
    }
    target = encode_example(record, tokenizer, 8)
    assert target["labels"][: len(pieces) + 2] == [tokenizer.bos_token_id, *pieces, end]
    assert target["syllable_labels"][: len(pieces) + 2] == [0] * (len(pieces) + 1) + [2]


def test_cached_steps_match_teacher_forcing(tokenizer, song, model):
    from transformers.modeling_outputs import BaseModelOutput

    model.eval()
    batch = ProsodyCollator(tokenizer.pad_token_id)([encode_example(song, tokenizer, 8)])
    with torch.no_grad():
        full = model(**batch)
        encoder = model.bart.get_encoder()(
            inputs_embeds=model.embed_source(batch["input_ids"], batch["length_ids"]),
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        shifted = [model.bart.prepare_decoder_input_ids_from_labels(batch["labels"])] + [
            model.shift_features(batch[key]) for key in TARGET_KEYS[1:]
        ]
        past = None
        for step in range(batch["labels"].shape[1]):
            output = model.bart(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder.last_hidden_state),
                attention_mask=batch["attention_mask"],
                decoder_inputs_embeds=model.embed_target(*[s[:, step : step + 1] for s in shifted]),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            assert torch.allclose(output.logits[:, 0], full.logits[:, step], atol=1e-6)
