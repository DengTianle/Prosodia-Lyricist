import json
import math

import pytest
import torch

from prosodia_lyricist import evaluation, infer
from prosodia_lyricist.evaluation import (
    aggregate_perplexity,
    bleu,
    conditional_perplexity,
    evaluate_prosody,
    nll_statistics,
    paper_syllable,
    prosody_bleu,
)
from prosodia_lyricist.features import encode_source
from prosodia_lyricist.prepare import SCHEMA_VERSION
from prosodia_lyricist.report import markdown_report, write_report


def test_bleu_clipping_brevity_and_fixed_order():
    assert bleu(list("abcd"), list("abcd")) == 1
    assert bleu(list("abcdabcd"), list("abcd")) == pytest.approx(math.exp(-1))
    assert bleu(list("abcd"), list("abce")) == 0
    assert bleu(list("abc"), list("abc")) == 0  # No hidden effective-order smoothing.
    assert bleu([], list("abcd")) == 0
    assert bleu(list("abcd"), []) == 0
    assert bleu(list("ab"), list("aaaa"), order=1) == pytest.approx(0.25)


def test_bleu_matches_nltk_without_special_tokens():
    nltk_bleu = pytest.importorskip("nltk.translate.bleu_score").sentence_bleu
    reference, hypothesis = list("abcabcabcd"), list("abcabcabcde")
    assert bleu(reference, hypothesis) == pytest.approx(nltk_bleu([reference], hypothesis))


@pytest.mark.parametrize(
    "value,stress,length",
    [
        ("ˌaɪ", "strong", "long"),
        ("`oʊ", "strong", "long"),
        ("'æ", "strong", "short"),
        ("ə", "weak", "short"),
        ("i:", "weak", "long"),
        ("iː", "weak", "long"),
        ("eɪ", "weak", "long"),
    ],
)
def test_supplement_ipa_rules(value, stress, length):
    result = paper_syllable({"ipa": value})
    assert (result["stress"], result["length"]) == (stress, length)


def test_joint_bleu_detects_length_difference():
    source = [{"stress": "strong", "length": "short"}] * 4
    assert prosody_bleu(source, source) == 1
    assert prosody_bleu(source, [{"stress": "strong", "length": "long"}] * 4) == 0


def test_missing_extra_and_failed_phrases_not_discarded(monkeypatch):
    source = [{"stress": "strong", "length": "short"}] * 4

    def parse(text):
        if text == "bad":
            raise ValueError("no pronunciation")
        return source

    monkeypatch.setattr(evaluation, "text_prosody", parse)
    records = [{"syllables": source}] * 2
    result = evaluate_prosody(records, ["good"])
    assert result["prosody_bleu"] == 0.5
    assert result["phrases"][1]["status"] == "missing"
    result = evaluate_prosody(records[:1], ["good", "extra"])
    assert result["prosody_bleu"] == 0.5
    result = evaluate_prosody(records, ["good", "bad"])
    assert result["prosody_bleu"] == 0.5
    assert result["pronunciation_failures"] == 1
    assert result["phrases"][1]["generated_count"] is None


def test_unknown_stress_is_unscorable(monkeypatch):
    monkeypatch.setattr(evaluation, "text_prosody", lambda _: [])
    result = evaluate_prosody([{"syllables": [{"stress": "unknown", "length": "short"}]}], [])
    assert result["prosody_bleu"] is None


def test_perplexity_masks_bos_padding_and_aggregates_by_token():
    logits = torch.zeros(2, 4, 5)
    labels = torch.tensor([[0, 3, 2, 1], [0, 4, 2, -100]])
    stats = nll_statistics(logits, labels, bos_token_id=0, pad_token_id=1)
    assert stats["token_count"] == 4  # Two ordinary targets and two EOS.
    assert aggregate_perplexity([stats])["perplexity"] == pytest.approx(5)
    assert labels[0, 0] == 0  # Masking must not mutate the caller's targets.
    result = aggregate_perplexity(
        [{"nll_sum": 2, "token_count": 1}, {"nll_sum": 0, "token_count": 3}]
    )
    assert result["perplexity"] == pytest.approx(math.exp(0.5))


def test_conditional_perplexity_uses_shifted_raw_logits(tiny_model, tokenizer, record):
    tiny_model.eval()
    inputs = {
        key: torch.tensor([value]) for key, value in encode_source(record, tokenizer, 8).items()
    }
    labels = torch.tensor([tokenizer.encode("hello world.")])
    result = conditional_perplexity(tiny_model, inputs, labels, tokenizer)
    with torch.no_grad():
        raw = tiny_model(**inputs, labels=labels)
        expected = nll_statistics(
            raw.logits,
            labels,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    assert result["nll_sum"] == pytest.approx(expected["nll_sum"])
    # Teacher forcing must not accidentally include an auxiliary weighted loss.
    tiny_model.loss_weights["word"] = 99
    assert conditional_perplexity(tiny_model, inputs, labels, tokenizer) == result


def test_inference_report_and_template_cli(tmp_path, tokenizer, tiny_model, monkeypatch):
    miditoolkit = pytest.importorskip("miditoolkit")
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    inst = miditoolkit.Instrument(0)
    inst.notes = [miditoolkit.Note(80, 60, i * 240, (i + 1) * 240) for i in range(4)]
    midi.instruments = [inst]
    midi.markers = [miditoolkit.Marker("Phrase", 720)]
    path = tmp_path / "input.mid"
    midi.dump(str(path))
    checkpoint = tmp_path / "checkpoint"
    tiny_model.save(checkpoint, tokenizer)
    (checkpoint / "run.json").write_text(
        json.dumps(
            {
                "data_schema_version": SCHEMA_VERSION,
                "config": {
                    "data": {"stress_source": "ipa"},
                    "model": {"max_source_length": 64, "max_target_length": 32},
                },
            }
        )
    )
    monkeypatch.setattr(
        infer.ProsodyBart,
        "generate",
        lambda *a, **kw: torch.tensor(
            [[tokenizer.eos_token_id] + tokenizer.encode("hello world.")]
        ),
    )
    monkeypatch.setattr(evaluation, "text_prosody", lambda _: [])
    report = infer.infer(checkpoint, path, device="cpu", top_k=1, return_report=True)
    assert report["unit"] == "whole_song"
    assert report["stress_source"] == "supplement"
    assert report["lyrics"] == ["hello world"]
    assert report["generated_perplexity"]["token_count"] == 4
    assert "self-score" in markdown_report(report)
    write_report(report, tmp_path / "report")
    assert json.loads((tmp_path / "report.json").read_text())["midi_sha256"]
    with pytest.raises(FileExistsError):
        write_report(report, tmp_path / "report")
    with pytest.raises(ValueError, match="one nonempty line"):
        infer.infer(checkpoint, path, device="cpu", reference_lines=["one", "two"])
    monkeypatch.setattr(
        evaluation.ipa,
        "parse_words",
        lambda text: [
            {"text": word, "syllables": [{"ipa": "ˈaɪ"}]} for word in text.split()
        ],
    )
    with_reference = infer.infer(
        checkpoint,
        path,
        device="cpu",
        top_k=1,
        return_report=True,
        reference_lines=["hello world"],
    )
    assert with_reference["reference_perplexity"] == report["generated_perplexity"]
    assert with_reference["encoded_source"] == report["encoded_source"]
    assert with_reference["generated_token_ids"] == report["generated_token_ids"]
    assert with_reference["reference_syllables"][0][0] == {
        "word": "hello", "ipa": "ˈaɪ", "stress": "strong", "length": "long"
    }
    reference_md = markdown_report(with_reference)
    assert "Ground-truth lyrics: hello world" in reference_md
    assert "Ground-truth prosody (derived from lyric IPA): 2 syllables." in reference_md
    assert "`<strong,long> <strong,long>`" in reference_md
    assert "Ground-truth word / IPA | Ground-truth prosody |" in reference_md
    assert "hello / ˈaɪ | <strong,long>" in reference_md
    assert "Ground-truth" not in markdown_report(report)
    monkeypatch.setattr(
        "sys.argv",
        ["infer", "--midi", str(path), "--template-only", "--report-prefix", str(tmp_path / "tpl")],
    )
    infer.main()
    assert "metrics" not in json.loads((tmp_path / "tpl.json").read_text())
