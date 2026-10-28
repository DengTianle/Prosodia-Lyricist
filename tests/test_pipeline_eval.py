import json
import math
from types import SimpleNamespace

import pytest
import torch

from prosodia_lyricist import evaluation, infer, midi_eval, pipeline_eval
from prosodia_lyricist.melody_checkpoint import file_sha256
from prosodia_lyricist.prepare import SCHEMA_VERSION


def phrase(count, *, stress="strong", text="hello"):
    return {"text": text, "syllables": [{"stress": stress, "length": "short"}] * count}


@pytest.fixture
def decoder(tmp_path, tokenizer, tiny_model):
    checkpoint = tmp_path / "xai"
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
    return infer.TemplateDecoder(checkpoint, device="cpu")


@pytest.fixture
def songs(tmp_path):
    songs = []
    for name in ("first_song", "second_song"):
        midi, lyrics = tmp_path / f"{name}.mid", tmp_path / f"{name}.lyrics.txt"
        midi.write_bytes(b"placeholder")
        lyrics.write_text("hello(1)\nworld\n")
        songs.append(midi_eval.EvalSong(name, midi, lyrics))
    return songs


@pytest.fixture
def generation(monkeypatch, decoder):
    calls = []
    tokenizer = decoder.tokenizer

    def generate(**inputs):
        assert not torch.is_grad_enabled()
        assert "labels" not in inputs
        calls.append(inputs)
        return SimpleNamespace(
            sequences=torch.tensor(
                [
                    [tokenizer.bos_token_id]
                    + tokenizer.encode("hello world.", add_special_tokens=False)
                    + [tokenizer.eos_token_id]
                ]
            ),
            explanations=[{"text": "hello world.", "hit_token_limit": False}],
        )

    monkeypatch.setattr(decoder.model, "generate", generate)
    monkeypatch.setattr(evaluation, "text_prosody", lambda *a, **kw: phrase(4)["syllables"])
    monkeypatch.setattr(midi_eval, "text_prosody", lambda *a, **kw: phrase(4)["syllables"])
    # Match the pronunciation interface used by wordwise reference target encoding.
    monkeypatch.setattr(
        infer.ipa,
        "parse_words",
        lambda text, **kw: [{"text": word, "syllables": []} for word in text.split()],
    )
    return calls


def test_positional_accuracy_bleu_and_micro_aggregation():
    first = pipeline_eval.compare_prosody([phrase(4)], [phrase(4)])
    second = pipeline_eval.compare_prosody([phrase(2, stress="weak")], [phrase(8)])
    score = pipeline_eval.aggregate_relation([first, second])
    assert score["prosody_bleu"] == 0.5
    assert score["accuracy"]["combined"] == 1 / 3
    assert score["counts"]["count_mismatch_phrases"] == 1
    extra = pipeline_eval.compare_prosody([phrase(4), phrase(4)], [phrase(4)])
    assert extra["prosody_bleu"] == 0.5
    # Real MIDI and bridge templates have no lyric text.
    assert pipeline_eval.compare_prosody([phrase(4)], [{"syllables": phrase(4)["syllables"]}])


def test_whole_song_shared_generation_and_reference_independence(
    decoder,
    songs,
    generation,
    monkeypatch,
):
    checksum = file_sha256(decoder.checkpoint / "weights.pt")
    predictors = {"actual": lambda song: [phrase(4)], "bridge": lambda song: [phrase(4)]}
    report = pipeline_eval.evaluate_pipeline(songs, predictors, decoder, top_k=1)
    assert len(generation) == 4  # One complete song per method, never per reference phrase.
    assert report["common_song_ids"] == [s.song_id for s in songs]
    method = report["per_song"][0]["methods"]["bridge"]
    assert method["lyrics_template"]["prosody_bleu"] == 1
    assert method["lyrics_reference"]["prosody_bleu"] == 0.5  # Missing second lyric phrase.
    assert method["reference_perplexity"]["token_count"] == 5  # words + periods + EOS
    assert method["generated_perplexity"]["token_count"] == 4
    assert report["per_song"][0]["ignored_numeric_markers"] == 1
    original = method["generated_token_ids"], method["encoded_source"]
    songs[0].lyrics.write_text("changed\nreference\nwith\nmore lines\n")
    other = pipeline_eval.evaluate_pipeline(songs[:1], predictors, decoder, top_k=1)
    changed = other["per_song"][0]["methods"]["bridge"]
    assert (changed["generated_token_ids"], changed["encoded_source"]) == original
    assert changed["reference_perplexity"]["token_count"] != 5
    assert file_sha256(decoder.checkpoint / "weights.pt") == checksum
    assert all(not model.training for model in (decoder.model, decoder.model.bart))


def test_common_subsets_and_ppl_coverage_are_separate(decoder, songs, generation, monkeypatch):
    def restricted(song):
        if song.song_id == "second_song":
            raise ValueError("note limit")
        return [phrase(4)]

    def score(encoded, lines, **kwargs):
        # Actual and bridge have different length labels but the same count.
        if encoded["length_ids"].count(1):
            raise ValueError("reference too long")
        return {"nll_sum": 2.0, "token_count": 2, "perplexity": math.e}

    long = phrase(4)
    long["syllables"] = [{"stress": "strong", "length": "long"}] * 4
    monkeypatch.setattr(decoder, "score_reference", score)
    predictors = {"actual": lambda song: [long], "bridge": restricted}
    report = pipeline_eval.evaluate_pipeline(songs, predictors, decoder, top_k=1)
    assert report["common_song_ids"] == ["first_song"]
    assert report["common_reference_perplexity_song_ids"] == []
    assert report["methods"]["actual"]["evaluated_songs"] == 2
    assert report["methods"]["bridge"]["failed_songs"] == 1
    assert report["methods"]["actual"]["reference_perplexity"]["scored_songs"] == 0
    assert report["methods"]["bridge"]["reference_perplexity"]["perplexity"] == math.e
    assert "note limit" in pipeline_eval.markdown_summary(report)


def test_seed_is_stable_across_method_and_song_order(decoder, songs, generation):
    predictors = {"actual": lambda song: [phrase(4)], "bridge": lambda song: [phrase(4)]}
    first = pipeline_eval.evaluate_pipeline(songs, predictors, decoder, top_k=1)
    second = pipeline_eval.evaluate_pipeline(songs[::-1], predictors, decoder, top_k=1)

    def seeds(report):
        return {s["song_id"]: s["seed"] for s in report["per_song"]}

    assert seeds(first) == seeds(second)
    for song in first["per_song"]:
        assert song["methods"]["actual"]["generation"]["seed"] == song["seed"]
        assert song["methods"]["bridge"]["generation"]["seed"] == song["seed"]


def test_source_target_limits_and_resource_failures(decoder, songs, generation, monkeypatch):
    with pytest.raises(ValueError, match="target token limit"):
        pipeline_eval.evaluate_pipeline(
            songs, {"actual": lambda song: [phrase(4)]}, decoder, max_new_tokens=32
        )
    report = pipeline_eval.evaluate_pipeline(
        songs,
        {"actual": lambda song: [phrase(4)] * 10},
        decoder,
        top_k=1,
    )
    assert not generation
    assert report["methods"]["actual"]["failed_songs"] == 2
    assert report["methods"]["actual"]["lyrics_reference"]["prosody_bleu"] is None
    with pytest.raises(ValueError, match="Reference exceeds"):
        decoder.score_reference(
            infer.encode_source({"lines": [phrase(4)]}, decoder.tokenizer, 8), ["hello"] * 20
        )

    def broken(*args, **kwargs):
        raise RuntimeError("missing IPA resources")

    monkeypatch.setattr(midi_eval, "text_prosody", broken)
    with pytest.raises(RuntimeError, match="missing IPA resources"):
        pipeline_eval.evaluate_pipeline(songs, {"actual": lambda song: [phrase(4)]}, decoder)


def test_pronunciation_failures_penalized(decoder, songs, generation, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("unknown word")

    monkeypatch.setattr(evaluation, "text_prosody", failed)
    report = pipeline_eval.evaluate_pipeline(
        songs[:1],
        {"actual": lambda song: [phrase(4)]},
        decoder,
        top_k=1,
    )
    method = report["per_song"][0]["methods"]["actual"]
    assert method["pronunciation_failures"] == 1
    assert method["lyrics_reference"]["accuracy"]["combined"] == 0
    assert method["lyrics_template"]["prosody_bleu"] == 0
    assert method["generated_prosody"][0]["error"] == "unknown word"


def test_cli_loads_decoder_once_writes_reports_and_journal(
    tmp_path,
    decoder,
    songs,
    generation,
    monkeypatch,
    capsys,
):
    loads = []

    def load(*args, **kwargs):
        loads.append(1)
        return decoder

    monkeypatch.setattr(pipeline_eval, "TemplateDecoder", load)
    monkeypatch.setattr(
        pipeline_eval.MidiPredictor,
        "__call__",
        lambda self, song: [{**phrase(4), "length_rule": "onset_bucket_8_quarters"}],
    )
    pronounced = [{**s, "word": "hello", "ipa": "æ"} for s in phrase(4)["syllables"]]
    monkeypatch.setattr(evaluation, "text_prosody", lambda *a, **k: pronounced)
    monkeypatch.setattr(midi_eval, "text_prosody", lambda *a, **k: pronounced)
    output = tmp_path / "report.json"
    options = [
        "--checkpoint",
        str(decoder.checkpoint),
        "--data-dir",
        str(tmp_path),
        "--methods",
        "actual",
        "--output",
        str(output),
        "--no-progress",
    ]
    report = pipeline_eval.main(options)
    assert loads == [1]
    assert report["songs"] == 2
    assert len(output.with_suffix(".songs.jsonl").read_text().splitlines()) == 2
    assert output.with_suffix(".md").is_file()
    assert len(report["readable_reports"]["song_ids"]) == 2  # Default 3, capped at set size.
    assert output.with_suffix(".details").joinpath("index.md").is_file()
    assert "Detailed song reports and lyrics" in output.with_suffix(".md").read_text()
    assert json.loads(output.read_text())["checkpoint_weights_sha256"]
    assert "Lyrics/ref BLEU" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pipeline_eval.main(options)
