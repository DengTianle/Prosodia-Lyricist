import json
from pathlib import Path

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration

from prosodia_lyricist import evaluation, generation, ipa
from prosodia_lyricist.evaluation import conditional_perplexity, nll_statistics, reference_targets
from prosodia_lyricist.features import WORD_END, encode_source
from prosodia_lyricist.infer import infer, main
from prosodia_lyricist.model import ProsodyBart
from prosodia_lyricist.prepare import SCHEMA_VERSION
from prosodia_lyricist.report import markdown_report


@pytest.fixture
def explainable(tokenizer, monkeypatch):
    model = ProsodyBart(
        BartForConditionalGeneration(
            BartConfig(
                vocab_size=len(tokenizer),
                d_model=16,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                encoder_ffn_dim=32,
                decoder_ffn_dim=32,
                max_position_embeddings=512,
                dropout=0,
                attention_dropout=0,
                activation_dropout=0,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.eos_token_id,
            )
        ),
        max_syllables=40,
        dropout=0,
    ).eval()

    def parse(text):
        return [{"text": w, "syllables": [ipa.syllable_features("ˈeɪ")]} for w in text.split()]

    monkeypatch.setattr(ipa, "parse_words", parse)
    return model


def test_ppl_replays_all_streams_and_masks_events(explainable, tokenizer, record):
    inputs = {k: torch.tensor([v]) for k, v in encode_source(record, tokenizer, 40).items()}
    streams = reference_targets(["hello world"], tokenizer, 40, explainable=True)
    labels, *prosody = [torch.tensor([s]) for s in streams]
    score = conditional_perplexity(explainable, inputs, labels, tokenizer, prosody_labels=prosody)
    assert score["token_count"] == 4  # hello, world, period, EOS
    assert score["event_perplexity"]["token_count"] == 6
    with torch.no_grad():
        output = explainable(
            **inputs,
            labels=labels,
            syllable_labels=prosody[0],
            stress_labels=prosody[1],
            length_labels=prosody[2],
        )
    expected = nll_statistics(
        output.logits,
        labels,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        exclude_token_ids=[tokenizer.convert_tokens_to_ids(WORD_END)],
    )
    assert score["nll_sum"] == pytest.approx(expected["nll_sum"])
    explainable.loss_weights = dict.fromkeys(explainable.loss_weights, 99)
    assert (
        conditional_perplexity(explainable, inputs, labels, tokenizer, prosody_labels=prosody)
        == score
    )
    padded = [torch.nn.functional.pad(s, (0, 3), value=-100) for s in [labels, *prosody]]
    scored_padding = conditional_perplexity(
        explainable, inputs, padded[0], tokenizer, prosody_labels=padded[1:]
    )
    assert scored_padding["perplexity"] == pytest.approx(score["perplexity"], rel=1e-5)
    with pytest.raises(ValueError, match="requires three"):
        conditional_perplexity(explainable, inputs, labels, tokenizer)


def scripted_sample(monkeypatch, tokenizer, sequence):
    sequence = iter(sequence)

    def sample(logits, *, allowed=None, **kwargs):
        value = next(sequence) if logits.numel() == len(tokenizer) else 2
        assert allowed is None or value in allowed
        return value

    monkeypatch.setattr(generation, "sample", sample)


@pytest.mark.parametrize("correct", [True, False])
def test_imagine_report_cli_with_compound_checkpoint(
    tmp_path, explainable, tokenizer, monkeypatch, correct
):
    pytest.importorskip("miditoolkit")
    checkpoint = tmp_path / "tiny-explainable"
    explainable.save(checkpoint, tokenizer)
    (checkpoint / "run.json").write_text(
        json.dumps(
            {
                "data_schema_version": SCHEMA_VERSION,
                "smoke_test": True,
                "config": {
                    "data": {"stress_source": "ipa"},
                    "model": {"max_source_length": 512, "max_target_length": 128},
                },
            }
        )
    )
    end = tokenizer.convert_tokens_to_ids(WORD_END)
    sequence = [
        tokenizer.bos_token_id,
        tokenizer.convert_tokens_to_ids("hello"),
        end,
        tokenizer.convert_tokens_to_ids("world"),
        end,
        tokenizer.convert_tokens_to_ids("."),
        tokenizer.eos_token_id,
    ]
    scripted_sample(monkeypatch, tokenizer, sequence)
    midi = Path(__file__).resolve().parents[1] / "examples/imagine.mid"
    ref = tmp_path / "reference.txt"
    ref.write_text("\n".join(["hello world"] * 17) + "\n")
    prefix = tmp_path / "report"
    explanation_path = tmp_path / "explanation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--checkpoint",
            str(checkpoint),
            "--midi",
            str(midi),
            "--device",
            "cpu",
            "--top-k",
            "1",
            "--report-prefix",
            str(prefix),
            "--reference",
            str(ref),
            "--explanations",
            str(explanation_path),
        ]
        + ([] if correct else ["--no-prosody-correction"]),
    )
    main()
    report = json.loads(prefix.with_suffix(".json").read_text())
    explanation = json.loads(explanation_path.read_text())
    assert report["decoder_mode"] == "explainable"
    assert report["metrics"]["input_phrases"] == 17
    assert report["metrics"]["generated_phrases"] == 1
    assert report["generated_perplexity"]["token_count"] == 4
    assert report["reference_perplexity"]["token_count"] == 17 * 3 + 1
    assert report["decoder_explanation"] == explanation
    assert explanation["completed"] and not report["generation"]["hit_token_limit"]
    assert all(w["correction_applied"] is correct for w in explanation["words"])
    markdown = prefix.with_suffix(".md").read_text()
    assert "Decoder prosody feedback" in markdown
    assert report["reference_syllables"] == [
        [
            {"word": word, "ipa": "ˈeɪ", "stress": "strong", "length": "long"}
            for word in ("hello", "world")
        ]
    ] * 17
    assert markdown.count("Ground-truth lyrics: hello world") == 17
    assert markdown.count("Ground-truth prosody (derived from lyric IPA): 2 syllables.") == 17
    assert "`<strong,long> <strong,long>`" in markdown
    assert "Ground-truth word / IPA | Ground-truth prosody |" in markdown
    assert "hello / ˈeɪ | <strong,long>" in markdown
    # No-reference inference uses the identical source and generation sequence.
    scripted_sample(monkeypatch, tokenizer, sequence)
    without_reference = infer(
        checkpoint, midi, device="cpu", top_k=1, return_report=True, prosody_correction=correct
    )
    assert without_reference["encoded_source"] == report["encoded_source"]
    assert without_reference["generated_token_ids"] == report["generated_token_ids"]
    assert "reference_syllables" not in without_reference
    assert "Ground-truth" not in markdown_report(without_reference)


def test_truncation_after_unfinished_word_keeps_budget_warning(
    tmp_path, explainable, tokenizer, monkeypatch, record
):
    from prosodia_lyricist import infer as inference

    explainable.save(tmp_path, tokenizer)
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "data_schema_version": SCHEMA_VERSION,
                "config": {
                    "data": {"stress_source": "ipa"},
                    "model": {"max_source_length": 512, "max_target_length": 128},
                },
            }
        )
    )
    midi = Path(__file__).resolve().parents[1] / "examples/imagine.mid"
    scripted_sample(
        monkeypatch, tokenizer, [tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("hello")]
    )
    monkeypatch.setattr(evaluation, "text_prosody", lambda text: [])
    report = inference.infer(tmp_path, midi, device="cpu", max_new_tokens=2, return_report=True)
    assert report["generation"]["hit_token_limit"]
    assert report["decoder_explanation"]["truncated_word"] == "hello"
    assert report["lyrics"] == []
    assert report["generated_perplexity"]["perplexity"] is None
    assert report["generated_perplexity"]["token_count"] == 0
